"""In-memory workspace: holds parsed candidates, jobs, policies, the pipeline
tracker and the RAG index. The Streamlit UI keeps one Workspace per session."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from . import config
from .extractor import CandidateProfile, JobProfile, parse_candidate, parse_job
from .guardrails import classify_document
from .llm import GeminiClient
from .matcher import MatchResult, rank_candidates
from .parser import SUPPORTED_EXTENSIONS, DocumentError, extract_text, extract_text_from_bytes
from .rag import Document, RecruitmentIndex


ROLE_ABBREVIATIONS = {
    "ml": "machine learning", "ai": "artificial intelligence", "genai": "generative ai", "nlp": "nlp",
    "ds": "data scientist", "da": "data analyst", "de": "data engineer", "ba": "business analyst",
    "be": "backend", "back-end": "backend", "back end": "backend", "fe": "frontend", "sde": "software engineer",
    "swe": "software engineer", "mle": "machine learning engineer",
}
GENERIC_TITLE_WORDS = {"engineer", "developer", "role", "position", "senior", "junior", "lead", "the", "and", "of",
                       "manager", "specialist", "associate", "i", "ii", "iii", "llms", "llm"}


class Workspace:
    def __init__(self, llm: GeminiClient):
        self.llm = llm
        self.candidates: dict[str, CandidateProfile] = {}
        self.jobs: dict[str, JobProfile] = {}
        self.policies: list[Document] = []
        self.pipeline: pd.DataFrame = pd.DataFrame()
        self.rankings: dict[str, list[MatchResult]] = {}
        self.ranking_backend: str = ""
        self._index: RecruitmentIndex | None = None
        self.errors: list[str] = []

    # ------------------------------------------------------------- loading
    def load_sample_data(self, use_llm: bool = False) -> None:
        for path in sorted(config.CV_DIR.glob("*")):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._safe(lambda p=path: self.add_candidate_text(extract_text(p), p.name, use_llm))
        for path in sorted(config.JD_DIR.glob("*")):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._safe(lambda p=path: self.add_job_text(extract_text(p), p.name, use_llm))
        for path in sorted(config.POLICY_DIR.glob("*")):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.policies.append(Document(f"policy-{path.stem}", "policy", path.stem.replace("_", " ").title(), extract_text(path), path.name))
        if config.PIPELINE_CSV.exists():
            self.pipeline = pd.read_csv(config.PIPELINE_CSV)
        self.invalidate()

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # keep loading other files
            self.errors.append(str(exc))

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()

    def add_candidate_text(self, text: str, filename: str, use_llm: bool = False) -> CandidateProfile:
        # Re-uploading the SAME file name = "re-analyse": the old profile is replaced (e.g. after a parser fix).
        reanalysed = next((c for c in self.candidates.values() if c.source_file == filename), None)
        if reanalysed:
            self.candidates.pop(reanalysed.id)
        fp = self._fingerprint(text)
        for c in self.candidates.values():  # exact duplicate under a DIFFERENT file name
            if self._fingerprint(c.raw_text or c.text) == fp or self._fingerprint(c.text) == fp:
                raise DocumentError(f"'{filename}' is a duplicate of the CV already loaded for {c.name} - skipped. "
                                    "(To re-analyse a CV, upload it again with its original file name.)")
        profile = parse_candidate(text, filename, self.llm, use_llm)
        if reanalysed:
            profile.id = reanalysed.id
            profile.warnings.append(f"Re-analysed: replaced the earlier profile from '{filename}'.")
            self.candidates[profile.id] = profile
            self.invalidate()
            return profile
        # Same person re-applying with an updated CV (same email) -> replace the old version
        same = next((c for c in self.candidates.values() if profile.email and c.email.lower() == profile.email.lower()), None)
        if same:
            profile.id = same.id
            profile.warnings.append(f"Replaced an earlier CV for this candidate ({same.source_file}).")
        else:
            base, n = profile.id, 2
            while profile.id in self.candidates:
                profile.id = f"{base}-{n}"
                n += 1
            if profile.id != base:
                profile.warnings.append("Another candidate has the same name - use the full name / check the email to tell them apart.")
        self.candidates[profile.id] = profile
        self.invalidate()
        return profile

    def add_job_text(self, text: str, filename: str, use_llm: bool = False) -> JobProfile:
        fp = self._fingerprint(text)
        for j in self.jobs.values():
            if self._fingerprint(j.text) == fp:
                raise DocumentError(f"'{filename}' is a duplicate of the JD '{j.title}' - skipped.")
        job = parse_job(text, filename, self.llm, use_llm)
        if job.id in self.jobs:  # same title, different content -> keep both
            job.id = f"{job.id}-{len(self.jobs) + 1}"
            job.title = f"{job.title} ({filename})"
        self.jobs[job.id] = job
        self.invalidate()
        return job

    def add_upload(self, data: bytes, filename: str, kind: str, use_llm: bool) -> CandidateProfile | JobProfile:
        text = extract_text_from_bytes(data, filename)
        detected, conf = classify_document(text)
        if detected != kind and detected != "unknown" and conf >= 0.7:
            want = "Job descriptions" if detected == "jd" else "Candidate CVs"
            raise DocumentError(f"'{filename}' looks like a {'job description' if detected == 'jd' else 'CV'} "
                                f"(confidence {conf:.0%}), not a {'CV' if kind == 'cv' else 'job description'}. "
                                f"Please upload it in the '{want}' box.")
        return self.add_candidate_text(text, filename, use_llm) if kind == "cv" else self.add_job_text(text, filename, use_llm)

    def remove_candidate(self, cid: str) -> None:
        self.candidates.pop(cid, None); self.invalidate()

    def remove_job(self, jid: str) -> None:
        self.jobs.pop(jid, None); self.invalidate()

    def invalidate(self) -> None:
        self._index = None
        self.rankings = {}

    # ------------------------------------------------------------- analysis
    def rank(self, job_id: str, weights: dict[str, float] | None = None) -> list[MatchResult]:
        cache_key = f"{job_id}|{sorted((weights or {}).items())}"
        if cache_key not in self.rankings:
            results, backend = rank_candidates(self.jobs[job_id], list(self.candidates.values()), self.llm, weights)
            self.rankings[cache_key] = results
            self.ranking_backend = backend
        return self.rankings[cache_key]

    def match_for(self, job_id: str, cand_id: str, weights: dict[str, float] | None = None) -> MatchResult | None:
        return next((r for r in self.rank(job_id, weights) if r.candidate_id == cand_id), None)

    @property
    def index(self) -> RecruitmentIndex:
        if self._index is None:
            docs = [Document(c.id, "cv", c.name, c.text, c.source_file) for c in self.candidates.values()]
            docs += [Document(j.id, "jd", j.title, j.text, j.source_file) for j in self.jobs.values()]
            docs += self.policies
            for _, row in self.pipeline.iterrows():  # one retrievable record per pipeline row
                body = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in row.items())
                docs.append(Document(f"pipe-{row.get('candidate_name', '')}", "pipeline", f"Pipeline - {row.get('candidate_name', '')}", body, "pipeline.csv"))
            self._index = RecruitmentIndex(docs, self.llm, self.pipeline)
        return self._index

    # ------------------------------------------------------------- lookups
    def find_candidates(self, text: str, threshold: int = 88) -> list[CandidateProfile]:
        """All candidates referenced in free text. Full-name hits win; otherwise fuzzy first/last name
        matches (tolerates typos like 'Ananya Roa'). More than one result = ambiguous reference."""
        low = text.lower().replace("'s", "")
        full = [c for c in self.candidates.values() if c.name.lower() in low]
        if full:
            return full
        words = re.findall(r"[a-z]+", low)
        scored = []
        for c in self.candidates.values():
            parts = [p for p in c.name.lower().split() if len(p) >= 3]
            best = max((fuzz.ratio(p, w) for p in parts for w in words), default=0)
            if best >= threshold:
                scored.append((best, c))
        if not scored:
            return []
        top = max(s for s, _ in scored)
        return [c for s, c in sorted(scored, key=lambda x: -x[0]) if s == top]

    def pipeline_stage(self, name: str) -> str | None:
        if self.pipeline.empty or "candidate_name" not in self.pipeline:
            return None
        rows = self.pipeline[self.pipeline["candidate_name"].str.lower() == name.lower()]
        return None if rows.empty else str(rows.iloc[0].get("stage", ""))

    def find_candidate(self, text: str, threshold: int = 85) -> CandidateProfile | None:
        """Find a candidate mentioned anywhere in free text (full or first name)."""
        low = text.lower()
        best, best_score = None, 0
        for c in self.candidates.values():
            full = c.name.lower()
            if full in low:
                return c
            for part in full.split():
                if len(part) < 3:
                    continue
                score = max((fuzz.ratio(part, w) for w in low.replace("'s", "").split()), default=0)
                if score > best_score:
                    best, best_score = c, score
        return best if best_score >= threshold else None

    def find_job(self, text: str, threshold: int = 70) -> JobProfile | None:
        """Match a role mentioned in free text. Expands recruiter shorthand (ML, DA, SDE...) and scores by the
        share of distinctive title keywords present (fuzzy, typo-tolerant). Ties -> None (caller falls back
        to the active role) rather than guessing."""
        if not self.jobs:
            return None
        msg = " " + text.lower() + " "
        for short, full in ROLE_ABBREVIATIONS.items():
            msg = re.sub(rf"(?<![a-z]){short}(?![a-z])", full, msg)
        words = re.findall(r"[a-z]+", msg)
        scores = []
        for j in self.jobs.values():
            keys = [k for k in re.findall(r"[a-z]+", j.title.lower()) if k not in GENERIC_TITLE_WORDS and len(k) > 1]
            if not keys:
                continue
            hit = sum(1 for k in keys if any(fuzz.ratio(k, w) >= 85 for w in words))
            scores.append((hit / len(keys), hit, j))
        if not scores:
            return None
        scores.sort(key=lambda x: (-x[0], -x[1]))
        best = scores[0]
        if best[1] == 0 or best[0] * 100 < min(threshold, 50):
            return None
        if len(scores) > 1 and (scores[1][0], scores[1][1]) == (best[0], best[1]):
            return None  # ambiguous
        return best[2]
