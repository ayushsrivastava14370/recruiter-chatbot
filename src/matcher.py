"""Explainable hybrid candidate ranking.

score = w_skills * skill_coverage + w_semantic * semantic_similarity + w_exp * experience_fit

- skill_coverage: weighted overlap of JD skills found in the CV
  (must-have skills count 2x, nice-to-have 1x).
- semantic_similarity: calibrated cosine similarity between JD and CV
  embeddings - captures related experience that keyword matching misses.
- experience_fit: min(1, candidate_years / required_years).

Every number is shown in the UI, so a recruiter can see *why* a candidate ranks
where they do. No LLM is involved in scoring, so rankings are deterministic.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .config import DEFAULT_WEIGHTS
from .embeddings import Embedder
from .extractor import CandidateProfile, JobProfile
from .llm import GeminiClient
from .skills import skill_in_text

LISTED_ONLY_CREDIT = 0.7  # credit for a skill that appears only in a skills list, with no supporting evidence

# cosine ranges used to calibrate raw similarity into 0..1 per backend
_CALIBRATION = {"gemini": (0.55, 0.85), "tfidf": (0.0, 0.35)}


class MatchResult(BaseModel):
    rank: int = 0
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    total_score: float  # 0-100
    skill_score: float  # 0-1
    semantic_score: float  # 0-1
    experience_score: float  # 0-1
    candidate_years: float
    required_years: float
    matched_must: list[str] = Field(default_factory=list)
    missing_must: list[str] = Field(default_factory=list)
    matched_nice: list[str] = Field(default_factory=list)
    missing_nice: list[str] = Field(default_factory=list)
    extra_skills: list[str] = Field(default_factory=list)
    recommendation: str = ""
    listed_only: list[str] = Field(default_factory=list)  # matched, but only in a skills list (verify in interview)
    flags: list[str] = Field(default_factory=list)  # parsing / integrity warnings for this candidate

    def explanation(self) -> str:
        parts = [
            f"{self.candidate_name} scores {self.total_score:.0f}/100 for {self.job_title} ({self.recommendation}).",
            f"Must-have skills matched: {len(self.matched_must)}/{len(self.matched_must) + len(self.missing_must)}"
            + (f" (missing: {', '.join(self.missing_must)})" if self.missing_must else "."),
            f"Experience: {self.candidate_years:g} yrs vs {self.required_years:g} required.",
        ]
        return " ".join(parts)


def _recommendation(score: float, missing_must: int) -> str:
    if score >= 75 and missing_must <= 1:
        return "Strong fit - shortlist"
    if score >= 60:
        return "Good fit - consider"
    if score >= 45:
        return "Partial fit - review gaps"
    return "Low fit"


def _has(skill: str, cand: CandidateProfile) -> bool:
    return skill in cand.skills or skill_in_text(skill, cand.text)


def rank_candidates(
    job: JobProfile,
    candidates: list[CandidateProfile],
    llm: GeminiClient | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[list[MatchResult], str]:
    """Rank candidates for a job. Returns (results, embedding_backend_label)."""
    if not candidates:
        return [], ""
    w = {k: max(0.0, float(v)) for k, v in (weights or DEFAULT_WEIGHTS).items()}
    if sum(w.values()) == 0:  # all sliders at zero -> fall back to defaults
        w = dict(DEFAULT_WEIGHTS)
    if not (job.must_have or job.nice_to_have):  # JD without detectable skills
        w["skills"] = 0.0
        if w["semantic"] + w["experience"] == 0:
            w["semantic"] = 1.0
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    embedder = Embedder(llm)
    matrix = embedder.fit_transform([job.text] + [c.text for c in candidates])
    sims = matrix[1:] @ matrix[0]
    lo, hi = _CALIBRATION[embedder.backend]

    results: list[MatchResult] = []
    for cand, sim in zip(candidates, sims):
        matched_must = [s for s in job.must_have if _has(s, cand)]
        missing_must = [s for s in job.must_have if s not in matched_must]
        matched_nice = [s for s in job.nice_to_have if _has(s, cand)]
        missing_nice = [s for s in job.nice_to_have if s not in matched_nice]

        # Evidence-weighted coverage: a skill backed by experience/projects earns full credit; a skill that
        # only appears in a bare skills list earns LISTED_ONLY_CREDIT (counters keyword stuffing).
        listed_only = set(cand.listed_only_skills)
        credit = lambda s: LISTED_ONLY_CREDIT if s in listed_only else 1.0  # noqa: E731
        denom = 2 * len(job.must_have) + len(job.nice_to_have)
        earned = 2 * sum(credit(s) for s in matched_must) + sum(credit(s) for s in matched_nice)
        skill_score = earned / denom if denom else 0.0
        semantic = float(np.clip((sim - lo) / (hi - lo), 0.0, 1.0))
        exp_score = 1.0 if job.min_years <= 0 else float(min(1.0, cand.years_experience / job.min_years))

        total = 100 * (w["skills"] * skill_score + w["semantic"] * semantic + w["experience"] * exp_score)
        jd_skills = set(job.must_have) | set(job.nice_to_have)
        results.append(
            MatchResult(
                candidate_id=cand.id,
                candidate_name=cand.name,
                job_id=job.id,
                job_title=job.title,
                total_score=round(total, 1),
                skill_score=round(skill_score, 3),
                semantic_score=round(semantic, 3),
                experience_score=round(exp_score, 3),
                candidate_years=cand.years_experience,
                required_years=job.min_years,
                matched_must=matched_must,
                missing_must=missing_must,
                matched_nice=matched_nice,
                missing_nice=missing_nice,
                extra_skills=[s for s in cand.skills if s not in jd_skills][:10],
                recommendation=_recommendation(total, len(missing_must)),
                listed_only=[s for s in matched_must + matched_nice if s in cand.listed_only_skills],
                flags=list(cand.warnings),
            )
        )
    # Ties broken by must-have coverage, then experience, then name (stable & deterministic)
    results.sort(key=lambda r: (-r.total_score, -len(r.matched_must), -r.candidate_years, r.candidate_name))
    for i, r in enumerate(results, 1):
        r.rank = i
    return results, embedder.label
