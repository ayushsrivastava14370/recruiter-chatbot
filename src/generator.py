"""LLM-powered content generation: candidate summaries, interview questions and
recruitment emails. Every generator has a deterministic template fallback so the
app keeps working without an API key.

All prompts pass the structured match analysis (matched / missing skills) plus
the raw CV, and instruct the model to use only that information.
"""
from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from . import config
from .extractor import CandidateProfile, JobProfile
from .guardrails import UNTRUSTED_DATA_RULE, redact_pii, wrap_document
from .llm import GeminiClient, LLMError
from .matcher import MatchResult


def _prep(text: str) -> str:
    """Data minimisation: contact details are not needed by the LLM, so they are masked."""
    return redact_pii(text) if config.REDACT_PII else text


GROUNDING_RULES = (
    "Use ONLY facts from the CANDIDATE CV, JOB DESCRIPTION and MATCH ANALYSIS below. "
    "Do not invent employers, numbers, degrees or skills. If something is unknown, say so. "
    "Never mention or infer protected characteristics (age, gender, ethnicity, religion, marital status)."
)

GROUNDING_RULES += " " + UNTRUSTED_DATA_RULE

# Email types that don't make sense for a candidate's current pipeline stage
_STAGE_CONFLICTS = {
    "rejection": {"offer", "hiring manager round"},
    "interview_invite": {"rejected", "offer"},
    "assessment": {"rejected", "offer"},
    "offer_call": {"rejected", "applied", "hr screening"},
    "shortlist_update": {"rejected"},
}


def stage_conflict(email_type: str, stage: str | None) -> str | None:
    """Warn when an email contradicts the pipeline (e.g. a rejection to someone at Offer stage)."""
    if stage and stage.lower() in _STAGE_CONFLICTS.get(email_type, set()):
        return (f"This candidate is currently at the **{stage}** stage in the pipeline tracker - "
                f"a '{EMAIL_TYPES[email_type]}' email may be a mistake. Please double-check before sending.")
    return None


def validate_email_details(details: dict) -> list[str]:
    """Sanity-check recruiter-entered logistics: past dates, weekends, missing link."""
    import pandas as pd

    warnings = []
    raw = (details.get("date") or "").strip()
    if raw:
        try:
            d = pd.to_datetime(raw, dayfirst=True).date()
            if d < date.today():
                warnings.append(f"The date {d:%d %b %Y} is in the past.")
            elif d.weekday() >= 5:
                warnings.append(f"{d:%A %d %b} is a weekend - our policy schedules interviews Monday-Friday.")
        except (ValueError, TypeError, OverflowError):
            warnings.append(f"Could not understand the date '{raw}' - it will be used exactly as typed.")
    link = (details.get("link") or "").strip()
    if link and not link.startswith(("http://", "https://")):
        warnings.append("The meeting link doesn't start with http(s):// - check it.")
    return warnings


EMAIL_TYPES = {
    "interview_invite": "Interview invitation",
    "assessment": "Technical assessment instructions",
    "shortlist_update": "Shortlisted - next steps",
    "rejection": "Polite rejection",
    "offer_call": "Offer discussion call",
}


class InterviewQuestion(BaseModel):
    category: str = Field(description="One of: Technical, Skill-gap probe, Behavioural, Role-specific scenario")
    question: str
    rationale: str = Field(description="Why this question for this candidate, referencing their CV or a gap")
    good_answer_signals: str = Field(description="What a strong answer would include")


class InterviewQuestionSet(BaseModel):
    questions: list[InterviewQuestion]


class EmailDraft(BaseModel):
    subject: str
    body: str


def _context(cand: CandidateProfile, job: JobProfile | None, match: MatchResult | None) -> str:
    parts = ["CANDIDATE CV:\n" + wrap_document("cv", cand.name, _prep(cand.text[:8000]))]
    if job:
        parts.append("JOB DESCRIPTION:\n" + wrap_document("jd", job.title, job.text[:5000]))
    if match:
        parts.append(
            "MATCH ANALYSIS:\n"
            f"- Overall score: {match.total_score}/100 ({match.recommendation})\n"
            f"- Matched must-have: {', '.join(match.matched_must) or 'none'}\n"
            f"- Missing must-have: {', '.join(match.missing_must) or 'none'}\n"
            f"- Matched nice-to-have: {', '.join(match.matched_nice) or 'none'}\n"
            f"- Missing nice-to-have: {', '.join(match.missing_nice) or 'none'}\n"
            f"- Experience: {match.candidate_years:g} years (required {match.required_years:g})"
        )
    return "\n\n".join(parts)


# ------------------------------------------------------------------ summary
def candidate_summary(cand: CandidateProfile, job: JobProfile | None, match: MatchResult | None, llm: GeminiClient | None) -> str:
    if llm and llm.available:
        prompt = (
            f"{GROUNDING_RULES}\n\n{_context(cand, job, match)}\n\n"
            "Write a recruiter-ready candidate summary in Markdown with these sections:\n"
            "**Snapshot** (2 sentences: current role, years of experience, core domain)\n"
            "**Key strengths** (3-5 bullets, each tied to concrete CV evidence)\n"
            + ("**Fit for the role** (relevant matches) and **Gaps / risks** (missing skills, with how critical they are)\n" if job else "")
            + "**Recommendation** (one line" + (", based on the match analysis" if match else "") + ")\n"
            "Keep it under 220 words."
        )
        try:
            return llm.generate(prompt, system="You are an experienced technical recruiter.", temperature=0.3)
        except LLMError as exc:
            return f"⚠️ {exc}\n\n" + _summary_template(cand, job, match)
    return _summary_template(cand, job, match)


def _summary_template(cand: CandidateProfile, job: JobProfile | None, match: MatchResult | None) -> str:
    lines = [
        f"**Snapshot** - {cand.name}, {cand.current_title or 'professional'} with ~{cand.years_experience:g} years of experience"
        + (f", based in {cand.location}." if cand.location else "."),
        f"\n**Skills found in CV:** {', '.join(cand.skills) or 'none detected'}",
    ]
    if cand.education:
        lines.append(f"\n**Education:** {'; '.join(cand.education)}")
    if cand.certifications:
        lines.append(f"\n**Certifications:** {'; '.join(cand.certifications)}")
    if match and job:
        lines.append(f"\n**Fit for {job.title}:** {match.total_score:.0f}/100 - {match.recommendation}")
        lines.append(f"\n- Matched must-haves: {', '.join(match.matched_must) or 'none'}")
        lines.append(f"- Gaps: {', '.join(match.missing_must + match.missing_nice) or 'none'}")
    lines.append("\n_Template summary (offline mode). Add a Gemini API key for an AI-written narrative._")
    return "\n".join(lines)


# -------------------------------------------------------- interview questions
def interview_questions(
    cand: CandidateProfile, job: JobProfile, match: MatchResult | None, llm: GeminiClient | None, n: int = 8
) -> list[InterviewQuestion]:
    if llm and llm.available:
        prompt = (
            f"{GROUNDING_RULES}\n\n{_context(cand, job, match)}\n\n"
            f"Create {n} interview questions for this candidate for the {job.title} role:\n"
            "- ~40% Technical questions that dig into projects and skills claimed in the CV (reference them specifically)\n"
            "- ~25% Skill-gap probes for the MISSING skills in the match analysis (assess transferable ability)\n"
            "- ~20% Role-specific scenario questions based on the JD responsibilities\n"
            "- ~15% Behavioural questions (STAR format)\n"
            "Each question must be specific to THIS candidate, not generic."
        )
        try:
            result: InterviewQuestionSet = llm.generate(
                prompt, system="You are a senior hiring manager designing a structured interview.", temperature=0.4, schema=InterviewQuestionSet
            )
            return result.questions
        except LLMError:
            pass
    return _questions_template(cand, job, match)


def _questions_template(cand: CandidateProfile, job: JobProfile, match: MatchResult | None) -> list[InterviewQuestion]:
    qs: list[InterviewQuestion] = []
    matched = (match.matched_must + match.matched_nice) if match else cand.skills
    missing = match.missing_must if match else []
    for s in matched[:3]:
        qs.append(InterviewQuestion(
            category="Technical",
            question=f"Walk me through a project where you used {s}. What was your specific contribution and what trade-offs did you make?",
            rationale=f"{s} is required for {job.title} and appears in the CV.",
            good_answer_signals="Concrete project, measurable outcome, awareness of alternatives.",
        ))
    for s in missing[:2]:
        qs.append(InterviewQuestion(
            category="Skill-gap probe",
            question=f"This role needs {s}. What exposure have you had to it, and how would you get productive with it in your first month?",
            rationale=f"{s} is a must-have that was not found in the CV.",
            good_answer_signals="Honest self-assessment, related experience, a realistic learning plan.",
        ))
    qs.append(InterviewQuestion(
        category="Role-specific scenario",
        question=f"Imagine your first 90 days as {job.title}. Based on the job description, what would you prioritise and why?",
        rationale="Tests understanding of the role's responsibilities.",
        good_answer_signals="Prioritisation, stakeholder awareness, quick wins vs long-term work.",
    ))
    qs.append(InterviewQuestion(
        category="Behavioural",
        question="Tell me about a time you disagreed with a technical decision. What did you do and what was the outcome? (STAR)",
        rationale="Assesses collaboration and communication.",
        good_answer_signals="Clear situation, respectful challenge, data-driven, reflection on outcome.",
    ))
    return qs


# ------------------------------------------------------------------ emails
def draft_email(
    email_type: str,
    cand: CandidateProfile,
    job: JobProfile | None,
    llm: GeminiClient | None,
    match: MatchResult | None = None,
    details: dict | None = None,
) -> EmailDraft:
    details = details or {}
    d = _default_details(email_type, details)
    role = job.title if job else "the role"
    if llm and llm.available:
        detail_lines = "\n".join(f"- {k}: {v}" for k, v in d.items() if v)
        prompt = (
            f"Draft a '{EMAIL_TYPES.get(email_type, email_type)}' email from {config.RECRUITER_NAME} "
            f"({config.RECRUITER_TITLE}, {config.COMPANY_NAME}) to candidate {cand.name} for the {role} position.\n"
            f"Use these exact logistics (do not change dates/times/links):\n{detail_lines}\n\n"
            "Guidelines: warm, professional, concise (120-200 words), clear call to action, plain text (no markdown). "
            "Format the body like a real email: greeting line, 2-4 short paragraphs separated by blank lines "
            "(logistics as one item per line), then a sign-off with the recruiter's name, title and company on separate lines. "
            "You may personalise with ONE genuine detail from the candidate's CV below. "
            + ("For rejections, be kind and do not list specific deficiencies. " if email_type == "rejection" else "")
            + "Do not invent salary or any information not given.\n\n"
            "CANDIDATE CV (for personalisation only):\n" + wrap_document("cv", cand.name, _prep(cand.text[:3000]))
        )
        try:
            return llm.generate(prompt, system="You are a professional recruitment coordinator.", temperature=0.5, schema=EmailDraft)
        except LLMError:
            pass
    return _email_template(email_type, cand, role, d)


def _default_details(email_type: str, details: dict) -> dict:
    nxt = date.today() + timedelta(days=3)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    base = {
        "date": details.get("date") or nxt.strftime("%A, %d %B %Y"),
        "time": details.get("time") or "11:00 AM IST",
        "duration": details.get("duration") or ("90 minutes" if email_type == "assessment" else "45 minutes"),
        "format": details.get("format") or "Video call (Google Meet)",
        "interviewers": details.get("interviewers") or "Hiring Manager and a Senior Engineer",
        "link": details.get("link") or "https://meet.example.com/abc-defg-hij",
        "deadline": details.get("deadline") or (date.today() + timedelta(days=5)).strftime("%d %B %Y"),
        "assessment_platform": details.get("assessment_platform") or "HackerRank (link will be sent separately)",
        "notes": details.get("notes") or "",
    }
    if email_type == "assessment":
        base.pop("interviewers"); base.pop("format")
    elif not details.get("assessment_platform"):
        base.pop("assessment_platform")  # only mention an assessment when one is actually part of this email
    if email_type in ("rejection", "shortlist_update"):
        for k in ("time", "duration", "format", "interviewers", "link", "assessment_platform", "date"):
            base.pop(k, None)
        if email_type == "rejection":
            base.pop("deadline", None)
    return base


def _email_template(email_type: str, cand: CandidateProfile, role: str, d: dict) -> EmailDraft:
    first = cand.name.split()[0]
    sign = f"\n\nBest regards,\n{config.RECRUITER_NAME}\n{config.RECRUITER_TITLE}, {config.COMPANY_NAME}"
    if email_type == "interview_invite":
        return EmailDraft(
            subject=f"Interview Invitation - {role} at {config.COMPANY_NAME}",
            body=(f"Dear {first},\n\nThank you for your interest in the {role} position at {config.COMPANY_NAME}. "
                  f"We were impressed by your profile and would like to invite you to an interview.\n\n"
                  f"Date: {d['date']}\nTime: {d['time']}\nDuration: {d['duration']}\nFormat: {d['format']}\n"
                  f"Interviewers: {d['interviewers']}\nLink: {d['link']}\n\n"
                  f"Please confirm your availability by replying to this email by {d['deadline']}. "
                  "If the slot doesn't work, let us know a few alternatives." + sign),
        )
    if email_type == "assessment":
        return EmailDraft(
            subject=f"Technical Assessment - {role} at {config.COMPANY_NAME}",
            body=(f"Dear {first},\n\nAs the next step for the {role} position, we invite you to complete a technical assessment.\n\n"
                  f"Platform: {d['assessment_platform']}\nDuration: {d['duration']}\nComplete by: {d['deadline']}\n\n"
                  "Instructions:\n1. Use a stable internet connection and a laptop/desktop.\n"
                  "2. The timer starts when you open the test; it cannot be paused.\n"
                  "3. You may use official documentation, but not AI assistants or help from others.\n"
                  "4. Write clean, commented code - readability is assessed.\n\n"
                  "If you need accommodations or face technical issues, reply to this email." + sign),
        )
    if email_type == "shortlist_update":
        return EmailDraft(
            subject=f"Update on your application - {role}",
            body=(f"Dear {first},\n\nGood news - you have been shortlisted for the {role} position at {config.COMPANY_NAME}. "
                  f"Our team will reach out with the next steps shortly. Please share your availability for the coming week by {d['deadline']}." + sign),
        )
    if email_type == "offer_call":
        return EmailDraft(
            subject=f"Next steps - {role} at {config.COMPANY_NAME}",
            body=(f"Dear {first},\n\nCongratulations on completing our interview process for the {role} position! "
                  f"We would like to schedule a call to discuss the offer.\n\nProposed: {d['date']} at {d['time']} ({d['duration']})\n"
                  f"Link: {d['link']}\n\nPlease confirm if this works for you." + sign),
        )
    return EmailDraft(
        subject=f"Your application for {role} at {config.COMPANY_NAME}",
        body=(f"Dear {first},\n\nThank you for taking the time to apply for the {role} position at {config.COMPANY_NAME}. "
              "After careful consideration, we have decided to move forward with other candidates whose experience more closely "
              "matches the current requirements of the role.\n\nWe appreciate your interest and encourage you to apply for future "
              "openings that match your profile. We wish you every success." + sign),
    )
