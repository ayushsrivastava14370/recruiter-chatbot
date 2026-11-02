"""Turn raw CV / JD text into structured profiles.

Two-stage, "verify then trust" extraction:
1. Deterministic rules (regex + skill taxonomy) - always run, instant, offline.
2. Optional Gemini structured extraction (JSON schema). LLM-proposed skills are
   kept ONLY if they can be found in the source text, so the profile never
   contains hallucinated skills.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from .guardrails import (
    UNTRUSTED_DATA_RULE,
    classify_document,
    detect_prompt_injection,
    strip_injected_text,
    text_without_skills_section,
)
from .llm import GeminiClient, LLMError
from .skills import extract_skills, normalize_skill, skill_in_text


# ----------------------------------------------------------------- data models
class CandidateProfile(BaseModel):
    id: str
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    current_title: str = ""
    years_experience: float = 0.0
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    source_file: str = ""
    text: str = ""
    parsed_with: str = "rules"
    raw_text: str = ""
    evidenced_skills: list[str] = Field(default_factory=list)  # skills mentioned outside the bare SKILLS list
    warnings: list[str] = Field(default_factory=list)
    injection_flags: list[str] = Field(default_factory=list)

    @property
    def listed_only_skills(self) -> list[str]:
        """Skills that appear only in a skills list, with no supporting experience/project text."""
        return [s for s in self.skills if s not in self.evidenced_skills]


class JobProfile(BaseModel):
    id: str
    title: str
    department: str = ""
    location: str = ""
    min_years: float = 0.0
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    source_file: str = ""
    text: str = ""
    parsed_with: str = "rules"
    warnings: list[str] = Field(default_factory=list)


# LLM output schemas (kept flat & simple - works well with Gemini JSON mode)
class _LLMCandidate(BaseModel):
    name: str
    email: Optional[str] = None
    location: Optional[str] = None
    current_title: Optional[str] = None
    years_experience: Optional[float] = None
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class _LLMJob(BaseModel):
    title: str
    department: Optional[str] = None
    location: Optional[str] = None
    min_years: Optional[float] = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------- helpers
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)(?:\s+of)?(?:\s+\w+){0,3}\s+experience", re.IGNORECASE)
RANGE_RE = re.compile(
    r"(?:[A-Za-z]{3,9}\.?\s+)?((?:19|20)\d{2})\s*(?:-|to|–|—)\s*(?:[A-Za-z]{3,9}\.?\s+)?((?:19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"^\s*([A-Z][A-Za-z /&-]{2,40}):?\s*$")
KNOWN_HEADINGS = ("experience", "employment", "work history", "education", "skills", "summary", "profile", "projects",
                  "certification", "requirements", "qualifications", "responsibilities", "nice to have", "preferred",
                  "must have", "about", "benefits", "academic", "achievements")
EDU_KEYWORDS = ("b.tech", "btech", "b.e", "bachelor", "master", "m.tech", "mtech", "m.sc", "msc", "b.sc", "bsc", "mba", "phd", "ph.d", "b.com", "mca", "bca", "bba", "b.a.", "m.a.", "degree", "diploma")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "doc"


def _sections(text: str) -> dict[str, str]:
    """Split a document into {HEADING: body} using standalone heading lines."""
    sections: dict[str, list[str]] = {"_header": []}
    current = "_header"
    for line in text.splitlines():
        m = SECTION_RE.match(line)
        stripped = line.strip().rstrip(":").strip()
        is_heading = m and len(line.split()) <= 6 and (
            stripped.isupper() or line.strip().endswith(":") or any(k in stripped.lower() for k in KNOWN_HEADINGS)
        )
        if is_heading:
            current = m.group(1).strip().upper()
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _find_section(sections: dict[str, str], *keywords: str) -> str:
    for heading, body in sections.items():
        if any(k in heading for k in keywords):
            return body
    return ""


def _years_from_ranges(text: str) -> float:
    """Union of date ranges (in years) - avoids double counting overlapping jobs."""
    this_year = date.today().year
    intervals = []
    for start, end in RANGE_RE.findall(text):
        s = int(start)
        e = this_year if not end[0].isdigit() else int(end)
        if 1970 <= s <= e <= this_year:
            intervals.append((s, e))
    if not intervals:
        return 0.0
    intervals.sort()
    total, (cs, ce) = 0, intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return float(total)


LOCATION_RE = re.compile(r"^[A-Z][A-Za-z .'-]{1,30},\s*[A-Z][A-Za-z .'-]{1,30}$")


def _header_location(lines: list[str]) -> str:
    """Real CVs put 'City, Country | email | phone' in the header - pick the 'City, Country' part."""
    for line in lines[:6]:
        for part in re.split(r"\s*[|•·]\s*", line):
            part = part.strip()
            if LOCATION_RE.match(part) and not EMAIL_RE.search(part) and not re.search(r"\d", part):
                return part
    return ""


def _field(text: str, label: str) -> str:
    m = re.search(rf"^\s*{label}\s*[:|-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------- candidates
def parse_candidate_rules(text: str, source_file: str = "") -> CandidateProfile:
    raw_text = text
    injection = detect_prompt_injection(raw_text)
    if injection:  # remove AI-directed instructions BEFORE anything is extracted or scored
        text = strip_injected_text(raw_text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else "Unknown Candidate"
    if len(name.split()) > 5 or EMAIL_RE.search(name):  # first line isn't a name
        name = _field(text, "name") or "Unknown Candidate"

    sections = _sections(text)
    experience_text = _find_section(sections, "EXPERIENCE", "EMPLOYMENT", "WORK HISTORY")
    education_text = _find_section(sections, "EDUCATION", "ACADEMIC")
    cert_text = _find_section(sections, "CERTIFICATION")

    explicit = [float(y) for y in YEARS_RE.findall(text)]
    years = max(explicit) if explicit else _years_from_ranges(experience_text or text)

    education = [l.strip("-* ").strip() for l in education_text.splitlines() if any(k in l.lower() for k in EDU_KEYWORDS)]
    if not education:
        education = [l.strip("-* ").strip() for l in lines if any(re.search(rf"\b{re.escape(k)}\b", l.lower()) for k in EDU_KEYWORDS)][:3]
    certs = [l.strip("-* ").strip() for l in cert_text.splitlines() if l.strip()]

    title = _field(text, "current role") or _field(text, "title") or (lines[1] if len(lines) > 1 and not EMAIL_RE.search(lines[1]) else "")
    email_m = EMAIL_RE.search(text)
    phone_m = PHONE_RE.search(text)

    warnings: list[str] = []
    if not 0 <= years <= 50:
        warnings.append(f"Unrealistic experience value ({years:g} yrs) ignored - please verify manually.")
        years = 0.0
    if name == "Unknown Candidate":
        warnings.append("Candidate name could not be detected.")
    if not email_m:
        warnings.append("No email address found - emails cannot be addressed automatically.")
    skills = extract_skills(text)
    if not skills:
        warnings.append("No known skills detected - ranking will rely on semantic similarity only.")
    kind, conf = classify_document(text)
    if kind == "jd" and conf >= 0.6:
        warnings.append("This document looks like a JOB DESCRIPTION, not a CV. Did you upload it in the wrong box?")
    if injection:
        warnings.append("⚠️ Possible prompt-injection text found (instructions aimed at an AI screener). It is ignored by the scoring.")

    return CandidateProfile(
        id=slugify(name),
        name=name,
        email=email_m.group(0) if email_m else "",
        phone=phone_m.group(0).strip() if phone_m else "",
        location=_field(text, "location") or _header_location(lines),
        current_title=title[:80],
        years_experience=years,
        education=education[:4],
        skills=skills,
        certifications=certs[:6],
        source_file=source_file,
        text=text,  # sanitised text used for scoring, retrieval and prompts
        raw_text=raw_text if injection else "",  # original kept for human review when something was stripped
        evidenced_skills=extract_skills(text_without_skills_section(text)),
        warnings=warnings,
        injection_flags=injection,
    )


def parse_candidate(text: str, source_file: str = "", llm: GeminiClient | None = None, use_llm: bool = False) -> CandidateProfile:
    profile = parse_candidate_rules(text, source_file)
    text = profile.text  # sanitised (injected instructions removed)
    if not (use_llm and llm and llm.available):
        return profile
    try:
        data: _LLMCandidate = llm.generate(
            "Extract the candidate profile from this CV. Only include information explicitly stated in the CV. "
            "For years_experience, compute total professional (non-internship) experience.\n\nCV:\n" + text[:12000],
            system="You are a precise CV parser. Never invent information. " + UNTRUSTED_DATA_RULE,
            schema=_LLMCandidate,
            temperature=0,
        )
    except LLMError:
        return profile

    # Grounding check: keep only LLM skills that are actually in the CV text.
    verified = [normalize_skill(s) for s in data.skills if skill_in_text(normalize_skill(s), text) or skill_in_text(s, text)]
    merged = list(dict.fromkeys(profile.skills + verified))
    profile.skills = merged
    if data.name and profile.name == "Unknown Candidate":
        profile.name = data.name
        profile.id = slugify(data.name)
    profile.current_title = data.current_title or profile.current_title
    profile.location = data.location or profile.location
    if data.years_experience is not None and 0 <= data.years_experience <= 50:
        profile.years_experience = float(data.years_experience)
    profile.education = data.education or profile.education
    profile.certifications = data.certifications or profile.certifications
    profile.parsed_with = "rules + gemini (verified)"
    return profile


# --------------------------------------------------------------------- jobs
def parse_job_rules(text: str, source_file: str = "") -> JobProfile:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = _field(text, "job title") or _field(text, "position") or _field(text, "role") or (lines[0] if lines else "Untitled Role")
    sections = _sections(text)
    must_text = "\n".join(body for h, body in sections.items() if any(k in h for k in ("MUST", "REQUIRED", "REQUIREMENT", "QUALIFICATION")) and "PREFERRED" not in h)
    nice_text = "\n".join(body for h, body in sections.items() if any(k in h for k in ("NICE", "PREFERRED", "BONUS", "GOOD TO HAVE", "DESIRABLE")))

    must = extract_skills(must_text) if must_text else []
    nice = [s for s in extract_skills(nice_text) if s not in must]
    if not must:  # no clear sections -> treat every skill as must-have
        must = extract_skills(text)
    years = [float(y) for y in YEARS_RE.findall(text)] or [
        float(y) for y in re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", text, re.IGNORECASE)
    ]
    years = [y for y in years if 0 <= y <= 30]
    warnings: list[str] = []
    if not must and not nice:
        warnings.append("No known skills detected in this JD - skill weight is redistributed to semantic similarity and experience.")
    if not years:
        warnings.append("No minimum experience found - experience fit is treated as satisfied for everyone.")
    kind, conf = classify_document(text)
    if kind == "cv" and conf >= 0.6:
        warnings.append("This document looks like a CV, not a job description. Did you upload it in the wrong box?")
    return JobProfile(
        id=slugify(title),
        title=title[:100],
        department=_field(text, "department"),
        location=_field(text, "location"),
        min_years=min(years) if years else 0.0,
        must_have=must,
        nice_to_have=nice,
        source_file=source_file,
        text=text,
        warnings=warnings,
    )


def parse_job(text: str, source_file: str = "", llm: GeminiClient | None = None, use_llm: bool = False) -> JobProfile:
    job = parse_job_rules(text, source_file)
    if not (use_llm and llm and llm.available):
        return job
    try:
        data: _LLMJob = llm.generate(
            "Extract the structured job requirements from this job description. Use short skill names "
            "(e.g. 'Python', 'Docker'). Put a skill in must_have_skills only if the JD says it is required.\n\nJD:\n" + text[:12000],
            system="You are a precise job-description parser. Never invent requirements. " + UNTRUSTED_DATA_RULE,
            schema=_LLMJob,
            temperature=0,
        )
    except LLMError:
        return job
    must = [normalize_skill(s) for s in data.must_have_skills if skill_in_text(normalize_skill(s), text) or skill_in_text(s, text)]
    nice = [normalize_skill(s) for s in data.nice_to_have_skills if skill_in_text(normalize_skill(s), text) or skill_in_text(s, text)]
    if must:
        job.must_have = list(dict.fromkeys(must + [s for s in job.must_have if s not in nice]))
    job.nice_to_have = [s for s in dict.fromkeys(nice + job.nice_to_have) if s not in job.must_have]
    job.title = data.title or job.title
    job.department = data.department or job.department
    job.location = data.location or job.location
    if data.min_years is not None:
        job.min_years = float(data.min_years)
    job.parsed_with = "rules + gemini (verified)"
    return job
