"""Structured candidate search: experience / skill / location filters answered from parsed profiles.

Why this exists: semantic retrieval (RAG) returns the top-k most *similar* passages, which is the wrong
tool for filters such as "candidates with 4.5 years" or "who knows RAG and Docker" - a matching CV may
simply not be in the top-k. Filters are therefore evaluated deterministically over EVERY profile.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extractor import CandidateProfile
from .skills import extract_skills

_N = r"(\d{1,2}(?:\.\d{1,2})?)"
_Y = r"\s*\+?\s*(?:years?|yrs?)\b"


@dataclass
class Criteria:
    min_years: float | None = None
    max_years: float | None = None
    strict_min: bool = False
    strict_max: bool = False
    exact_years: float | None = None
    skills: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)

    @property
    def has_years(self) -> bool:
        return self.min_years is not None or self.max_years is not None or self.exact_years is not None

    def describe(self) -> str:
        parts = []
        if self.exact_years is not None:
            parts.append(f"~{self.exact_years:g} years of experience")
        if self.min_years is not None:
            parts.append(f"{'more than' if self.strict_min else 'at least'} {self.min_years:g} years")
        if self.max_years is not None:
            parts.append(f"{'less than' if self.strict_max else 'at most'} {self.max_years:g} years")
        if self.skills:
            parts.append("skills: " + ", ".join(self.skills))
        if self.locations:
            parts.append("location: " + ", ".join(self.locations))
        return "; ".join(parts)


_ASKS_FOR_PEOPLE = re.compile(
    r"\b(who|which|any|anyone|someone|candidates?|people|profiles?|list|find|show|search|filter|want|need|looking for)\b", re.I
)
_RELATION = re.compile(r"\b(has|have|having|with|knows?|know|experienced|skilled|proficient|from|based|located|in)\b", re.I)


def parse_criteria(message: str, known_cities: list[str]) -> Criteria | None:
    """Return filter criteria if the message is a structured candidate search, else None."""
    low = message.lower()
    if re.search(r"\b(notice|ctc|salary|lpa|stage|interview date)\b", low):
        return None  # pipeline-attribute questions are answered from the pipeline table by RAG
    c = Criteria()
    if m := re.search(rf"between\s+{_N}\s*(?:and|-|to)\s*{_N}{_Y}", low) or re.search(rf"\b{_N}\s*(?:-|to)\s*{_N}{_Y}", low):
        c.min_years, c.max_years = sorted((float(m.group(1)), float(m.group(2))))
    else:
        if m := re.search(rf"(more than|over|above|greater than|>)\s*{_N}{_Y}", low):
            c.min_years, c.strict_min = float(m.group(2)), True
        elif m := re.search(rf"(at least|atleast|minimum(?: of)?|min\.?|>=)\s*{_N}{_Y}", low) or re.search(rf"\b{_N}\s*\+\s*(?:years?|yrs?)\b", low):
            c.min_years = float(m.groups()[-1])
        if m := re.search(rf"(less than|under|below|fewer than|<)\s*{_N}{_Y}", low):
            c.max_years, c.strict_max = float(m.group(2)), True
        elif m := re.search(rf"(at most|up to|upto|maximum(?: of)?|max\.?|<=)\s*{_N}{_Y}", low):
            c.max_years = float(m.group(2))
        if c.min_years is None and c.max_years is None:
            if m := re.search(rf"\b{_N}{_Y}(?!\s*(?:old|ago|of age))", low):
                c.exact_years = float(m.group(1))
    if re.search(r"\b(fresher|freshers|fresh graduate|entry[- ]level|no experience)\b", low):
        c.max_years = 1.0
    c.skills = extract_skills(message)
    c.locations = [city for city in known_cities if re.search(rf"\b{re.escape(city.lower())}\b", low)]

    if c.has_years:
        return c
    if (c.skills or c.locations) and _ASKS_FOR_PEOPLE.search(low) and _RELATION.search(low):
        return c
    return None


def _years_ok(y: float, c: Criteria) -> bool:
    if c.exact_years is not None and abs(y - c.exact_years) > 0.25:
        return False
    if c.min_years is not None and (y <= c.min_years if c.strict_min else y < c.min_years):
        return False
    if c.max_years is not None and (y >= c.max_years if c.strict_max else y > c.max_years):
        return False
    return True


def _has_skill(cand: CandidateProfile, skill: str) -> str | None:
    if skill in cand.skills:
        return skill
    if skill in cand.implied_skills:
        return f"{skill} (implied)"
    return None


def search_candidates(cands: list[CandidateProfile], c: Criteria) -> tuple[list[tuple[CandidateProfile, list[str]]], list[CandidateProfile]]:
    """Return (matches with the skills that matched, closest-by-experience suggestions when nothing matches)."""
    matches = []
    for cand in cands:
        if c.has_years and not _years_ok(cand.years_experience, c):
            continue
        if c.locations and not any(loc.lower() in cand.location.lower() for loc in c.locations):
            continue
        hits = [_has_skill(cand, s) for s in c.skills]
        if c.skills and not all(hits):
            continue
        matches.append((cand, [h for h in hits if h]))
    matches.sort(key=lambda x: (-x[0].years_experience, x[0].name))

    closest: list[CandidateProfile] = []
    if not matches and c.has_years:
        target = c.exact_years if c.exact_years is not None else (c.min_years if c.min_years is not None else c.max_years)
        closest = sorted(cands, key=lambda x: (abs(x.years_experience - target), x.name))[:3]
    return matches, closest


def format_results(c: Criteria, matches, closest, total: int) -> str:
    lines = [f"### Candidates matching: {c.describe()}\n"]
    if matches:
        lines += ["| Candidate | Experience | How determined | Current title | Location | Matching skills |", "|---|---|---|---|---|---|"]
        for cand, hits in matches:
            lines.append(f"| **{cand.name}** | {cand.years_experience:g} yrs | {cand.experience_source or '-'} | "
                         f"{cand.current_title or '-'} | {cand.location or '-'} | {', '.join(hits) or '-'} |")
        lines.append(f"\n{len(matches)} of {total} candidates match.")
    else:
        lines.append(f"No candidate matches exactly (searched all {total} profiles).")
        if closest:
            lines.append("\n**Closest by experience:**")
            lines += [f"- **{x.name}**: {x.years_experience:g} yrs ({x.current_title or 'n/a'})" for x in closest]
    lines.append("\n_Answered deterministically from the parsed profiles of every candidate, including uploads (not semantic search). "
                 "Exact-year searches allow ±0.25 years._")
    return "\n".join(lines)
