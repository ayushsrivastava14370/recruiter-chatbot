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
from .skills import extract_skills, implied_skills, normalize_skill, skill_in_text


# ----------------------------------------------------------------- data models
class CandidateProfile(BaseModel):
    id: str
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    current_title: str = ""
    years_experience: float = 0.0
    experience_source: str = ""  # how years_experience was determined (shown in the UI)
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    source_file: str = ""
    text: str = ""
    parsed_with: str = "rules"
    raw_text: str = ""
    evidenced_skills: list[str] = Field(default_factory=list)  # skills mentioned outside the bare SKILLS list
    implied_skills: list[str] = Field(default_factory=list)  # e.g. RAG -> NLP (credited at a reduced rate)
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


# ---------------------------------------------------------------- experience
# 1) What the CV STATES: "4.5+ years of professional experience", "Total Experience: 6 yrs",
#    "experience of 3 years", "5 years 6 months of experience"
_NUM = r"(\d{1,2}(?:\.\d{1,2})?)"
STATED_YEARS_PATTERNS = [
    re.compile(_NUM + r"\s*\+?\s*(?:years?|yrs?)\b(?:\s*(?:and\s*)?(\d{1,2})\s*months?)?\+?(?:\s+(?:of|in))?(?:\s+[\w/&().,-]+){0,5}?\s+(?:experience|exp)\b", re.IGNORECASE),
    re.compile(r"\b(?:total|overall|professional|work|industry)?\s*experience\s*(?:of|:|-|–|—)?\s*(?:about|around|over|nearly)?\s*" + _NUM + r"\s*\+?\s*(?:years?|yrs?)\b(?:\s*(?:and\s*)?(\d{1,2})\s*months?)?", re.IGNORECASE),
]

# 2) What the EMPLOYMENT DATES add up to (month precision). Supports: "Jan 2021", "January 2021",
#    "Jan'21", "07/2021", "2021-07", "2021", and open ends "Present / Current / Till date / Now / Ongoing".
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE = (r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?[\s,'’-]*(?:19|20)?\d{2}"
         r"|\d{1,2}[/.-](?:19|20)\d{2}|(?:19|20)\d{2}[/.-]\d{1,2}(?!\d)|(?:19|20)\d{2})")
_OPEN_END = r"present|current|currently|now|till date|to date|ongoing|today|date"
RANGE_RE = re.compile(rf"({_DATE})\s*(?:-|–|—|to|till|until|~)\s*({_DATE}|{_OPEN_END})", re.IGNORECASE)


def _stated_years(text: str) -> float:
    found = []
    for pat in STATED_YEARS_PATTERNS:
        for m in pat.finditer(text):
            years = float(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0)
            if 0 < years <= 50:
                found.append(round(years, 1))
    return max(found) if found else 0.0  # the headline/summary figure is normally the largest


def _parse_month(token: str, is_end: bool) -> int | None:
    """Return an absolute month index (year*12 + month-1) for a date token."""
    t = token.strip().lower()
    today = date.today()
    if re.fullmatch(_OPEN_END, t):
        return today.year * 12 + today.month - 1
    if m := re.match(r"([a-z]{3})[a-z]*\.?[\s,'’-]*((?:19|20)?\d{2})$", t):
        mon = _MONTHS.get(m.group(1)[:3])
        yr = int(m.group(2)) + (2000 if len(m.group(2)) == 2 else 0)
        return yr * 12 + mon - 1 if mon else None
    if m := re.fullmatch(r"(\d{1,2})[/.-]((?:19|20)\d{2})", t):
        mon, yr = int(m.group(1)), int(m.group(2))
        return yr * 12 + mon - 1 if 1 <= mon <= 12 else None
    if m := re.fullmatch(r"((?:19|20)\d{2})[/.-](\d{1,2})", t):
        yr, mon = int(m.group(1)), int(m.group(2))
        return yr * 12 + mon - 1 if 1 <= mon <= 12 else None
    if re.fullmatch(r"(?:19|20)\d{2}", t):
        # year only: "2015 - 2018" conventionally means 3 years -> start Jan 2015, end Dec 2017 (inclusive)
        return int(t) * 12 - (1 if is_end else 0)
    return None


def _years_from_ranges(text: str) -> float:
    """Union of employment date ranges in MONTHS (overlapping jobs not double counted).
    Internship lines are excluded - 'total experience' conventionally means professional roles."""
    today_idx = date.today().year * 12 + date.today().month - 1
    intervals = []
    for line in text.splitlines():
        if re.search(r"\bintern(ship)?\b|\btrainee\b|\bapprentice", line, re.IGNORECASE):
            continue
        for start, end in RANGE_RE.findall(line):
            s, e = _parse_month(start, False), _parse_month(end, True)
            if s is None or e is None or not (1970 * 12 <= s <= e <= today_idx):
                continue
            intervals.append((s, e + 1))  # inclusive end month
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
    return round(total / 12, 1)


_SUMMARY_YEARS = re.compile(_NUM + r"\s*\+?\s*(?:years?|yrs?)\b(?!\s*(?:old|ago|of age))", re.IGNORECASE)


def _resolve_experience(text: str, summary: str, experience: str, education: str) -> tuple[float, str, str]:
    """Decide total experience. Priority (most to least trustworthy):
    1. a figure the candidate STATES ("4.5+ years of experience", or "N+ years" in the summary)
    2. the union of employment date ranges in the EXPERIENCE section (month precision, no internships)
    Education dates are never counted. Returns (years, source, warning)."""
    stated = _stated_years(text)
    if not stated and summary:
        vals = [float(v) for v in _SUMMARY_YEARS.findall(summary) if 0 < float(v) <= 50]
        stated = max(vals) if vals else 0.0
    dates_text = experience or (text.replace(education, "") if education else text)
    ranged = _years_from_ranges(dates_text)

    if stated:
        warning = ""
        if ranged and abs(stated - ranged) >= 1.5:
            warning = (f"CV states {stated:g} years of experience but employment dates add up to ~{ranged:g} years "
                       "(internships excluded) - please verify.")
        return stated, "stated in CV", warning
    if ranged:
        return ranged, "computed from employment dates (internships excluded)", ""
    return 0.0, "not found", "No professional experience found (internships are excluded) - verify manually."


_INSTITUTION = re.compile(r"\b(university|college|institute|school|academy|iit|nit|iiit)\b", re.IGNORECASE)


def _education_entries(education_text: str) -> list[str]:
    """Pair each degree line with its institution line (CVs put them on adjacent lines in either order)."""
    lines = [l.strip("-*• ").strip() for l in education_text.splitlines() if l.strip()]
    entries, used = [], set()
    for i, line in enumerate(lines):
        if not any(k in line.lower() for k in EDU_KEYWORDS) or i in used:
            continue
        entry = line
        for j in (i - 1, i + 1):
            if 0 <= j < len(lines) and j not in used and _INSTITUTION.search(lines[j]) and not any(
                k in lines[j].lower() for k in EDU_KEYWORDS
            ):
                entry = f"{line} - {lines[j]}" if j > i else f"{line} - {lines[j]}"
                used.add(j)
                break
        used.add(i)
        entries.append(entry)
    return entries


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
    if name.isupper():  # "AYUSH KUMAR SRIVASTAVA" -> "Ayush Kumar Srivastava"
        name = name.title()

    sections = _sections(text)
    experience_text = _find_section(sections, "EXPERIENCE", "EMPLOYMENT", "WORK HISTORY")
    education_text = _find_section(sections, "EDUCATION", "ACADEMIC")
    cert_text = _find_section(sections, "CERTIFICATION")
    summary_text = _find_section(sections, "SUMMARY", "PROFILE", "OBJECTIVE", "ABOUT") or sections.get("_header", "")

    years, exp_source, exp_warning = _resolve_experience(text, summary_text, experience_text, education_text)

    education = _education_entries(education_text)
    if not education:
        education = [l.strip("-* ").strip() for l in lines if any(re.search(rf"\b{re.escape(k)}\b", l.lower()) for k in EDU_KEYWORDS)][:3]
    certs = [l.strip("-* ").strip() for l in cert_text.splitlines() if l.strip()]

    title = _field(text, "current role") or _field(text, "title") or (lines[1] if len(lines) > 1 and not EMAIL_RE.search(lines[1]) else "")
    email_m = EMAIL_RE.search(text)
    phone_m = PHONE_RE.search(text)

    warnings: list[str] = [exp_warning] if exp_warning else []
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
        experience_source=exp_source,
        education=education[:4],
        skills=skills,
        certifications=certs[:6],
        source_file=source_file,
        text=text,  # sanitised text used for scoring, retrieval and prompts
        raw_text=raw_text if injection else "",  # original kept for human review when something was stripped
        evidenced_skills=extract_skills(text_without_skills_section(text)),
        implied_skills=implied_skills(skills),
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
            "For years_experience: if the CV states a total (e.g. '4.5+ years') return that number; otherwise sum the "
            "durations of all non-internship roles up to today (" + date.today().strftime("%B %Y") + ").\n\nCV:\n" + text[:12000],
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
    profile.implied_skills = implied_skills(merged)
    if data.name and profile.name == "Unknown Candidate":
        profile.name = data.name
        profile.id = slugify(data.name)
    profile.current_title = data.current_title or profile.current_title
    profile.location = data.location or profile.location
    # Experience: the LLM may only FILL A GAP, never overwrite a value stated in the CV or computed from dates.
    llm_years = data.years_experience
    if llm_years is not None and 0 <= llm_years <= 50:
        if profile.experience_source == "not found" and llm_years > 0:
            profile.years_experience = round(float(llm_years), 1)
            profile.experience_source = "estimated by Gemini (no explicit figure or dates found)"
            profile.warnings = [w for w in profile.warnings if not w.startswith("No professional experience found")]
            profile.warnings.append("Experience was estimated by AI - please verify manually.")
        elif profile.years_experience and abs(llm_years - profile.years_experience) >= 1.5:
            profile.warnings.append(
                f"AI estimate ({llm_years:g} yrs) differs from the CV ({profile.years_experience:g} yrs, "
                f"{profile.experience_source}); the CV value is used."
            )
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
