"""Deterministic answers for factual questions that have an exact answer in structured data.

Design rule: if a question has an EXACT answer in the parsed profiles or the pipeline table
(a candidate's years, skills, gaps, stage, CTC, notice period; "who is at Offer stage"; "how many
candidates"), answer it directly from that data. Semantic retrieval (RAG) is reserved for open-ended
questions (policies, process, "why"/"how" questions), where similarity search is the right tool.
"""
from __future__ import annotations

import re

import pandas as pd

from .extractor import CandidateProfile
from .skills import extract_skills, skill_in_text
from .workspace import Workspace

GREETING_RE = re.compile(r"^\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|thanks|thank you|ok|okay|cool)\b[\s!.?]*$", re.I)
HELP_TEXT = (
    "👋 Hi! I'm RecruitAI. I can help you with:\n"
    "- **Rank** candidates for a role: *Rank the top 5 for the ML Engineer role*\n"
    "- **Filter**: *candidates with more than 4 years*, *who knows RAG and Docker?*, *candidates in Bengaluru*\n"
    "- **Candidate facts**: *How many years of experience does Aarav have?*, *Does Neha know Power BI?*, *What are Arjun's gaps for ML?*\n"
    "- **Pipeline**: *Who is at the Technical Interview stage?*, *Which candidates have a 90-day notice period?*\n"
    "- **Summaries, interview questions and emails**: *Summarise Ananya*, *Interview questions for Arjun*, *Draft an invite for Sara*\n"
    "- **Policies**: *What is the salary band for the Data Analyst role?*"
)

_DOMAIN_TERMS = re.compile(
    r"\b(candidates?|cvs?|resumes?|applicants?|profiles?|roles?|jobs?|jds?|positions?|openings?|vacanc\w*|skills?|experience|"
    r"years?|yrs|fresher\w*|senior|junior|lead|engineers?|analysts?|developers?|scientists?|interns?|interview\w*|stages?|"
    r"hir\w+|recruit\w*|salary|salaries|compensation|ctc|lpa|pay|band|notice|offers?|assessments?|tests?|pass mark|panel|"
    r"process|polic\w+|referr\w+|sourc\w+|linkedin|naukri|pipeline|shortlist\w*|reject\w*|rank\w*|score\w*|match\w*|fit|"
    r"gaps?|compare|summar\w*|emails?|mail|invit\w*|questions?|onboard\w*|joining|leave|benefits?|manager|team|department|"
    r"location|city|remote|hybrid|upload\w*|pdf|docx|files?|documents?|data|tracker|status|screening|approv\w*|feedback)\b",
    re.I,
)


def is_off_topic(message: str, ws: Workspace) -> bool:
    """True when the message has nothing to do with recruitment (no domain word, candidate, skill or role)."""
    if _DOMAIN_TERMS.search(message) or extract_skills(message):
        return False
    if ws.find_candidates(message) or ws.find_job(message):
        return False
    return True


OFF_TOPIC_REPLY = ("I only help with recruitment tasks on the supplied data - ranking and filtering candidates, candidate facts, "
                   "pipeline status, summaries, interview questions, emails and hiring policies. Type **hi** to see examples.")

_PEOPLE_QUESTION = re.compile(r"\b(who|whom|which|candidates?|people|anyone|list|show|how many|count|names?)\b", re.I)

_PIPE_FIELDS = {
    "stage": r"\b(stage|status|progress|round)\b|\bwhere (is|are)\b.*\b(process|pipeline|hiring|funnel)\b",
    "expected_ctc_lpa": r"\b(ctc|salary expectation|expected salary|expected ctc|expectation|lpa|package)\b",
    "notice_period_days": r"\b(notice)\b",
    "recruiter": r"\b(recruiter|who is handling|owner)\b",
    "next_step": r"\b(next step|next round|next stage|upcoming|scheduled|interview date|when is)\b",
    "applied_date": r"\b(applied on|application date|when did .* apply)\b",
    "source": r"\b(source|sourced|came from|referral|referred)\b",
    "notes": r"\b(notes?|feedback|remarks?|comments?)\b",
}


def _pipeline_row(ws: Workspace, name: str) -> pd.Series | None:
    if ws.pipeline.empty:
        return None
    rows = ws.pipeline[ws.pipeline["candidate_name"].str.lower() == name.lower()]
    return None if rows.empty else rows.iloc[0]


# ----------------------------------------------------------------- one candidate
def candidate_fact(message: str, cand: CandidateProfile, ws: Workspace, job, weights) -> str | None:
    """Answer a factual question about ONE candidate from structured data, or None if not a fact question."""
    low = message.lower()
    asked_skills = extract_skills(message)

    # 1. Skill check: "Does Aarav know PyTorch?", "Has Neha used Power BI?"
    if asked_skills and re.search(r"\b(know|knows|have|has|had|use|used|uses|worked|experience (with|in|on)|skilled|familiar|proficient|can)\b", low):
        lines = [f"**{cand.name}** - skill check:"]
        evidence_text = cand.text
        for s in asked_skills:
            if s in cand.skills:
                how = "listed in the skills section only - verify in interview" if s in cand.listed_only_skills else "backed by experience/projects in the CV"
                lines.append(f"- ✅ **{s}**: yes ({how})")
            elif s in cand.implied_skills:
                lines.append(f"- 💡 **{s}**: implied by related skills in the CV (not stated explicitly)")
            elif skill_in_text(s, evidence_text):
                lines.append(f"- ✅ **{s}**: mentioned in the CV")
            else:
                lines.append(f"- ❌ **{s}**: not found in the CV")
        return "\n".join(lines) + "\n\n_Source: parsed CV profile._"

    # 2. Gaps for a role
    if re.search(r"\b(gaps?|missing|lacks?|lacking|weakness(es)?|shortfall|doesn'?t have|does not have)\b", low):
        if not job:
            return f"For which role should I check {cand.name}'s gaps? Available: " + ", ".join(j.title for j in ws.jobs.values())
        m = ws.match_for(job.id, cand.id, weights)
        return (f"**{cand.name}** vs **{job.title}** - score {m.total_score:.0f}/100 ({m.recommendation})\n\n"
                f"- ❌ Missing must-have: {', '.join(m.missing_must) or 'none'}\n"
                f"- ➖ Missing nice-to-have: {', '.join(m.missing_nice) or 'none'}\n"
                f"- ✅ Matched must-have: {', '.join(m.matched_must) or 'none'}"
                + (f"\n- 💡 Implied (80% credit): {', '.join(m.implied)}" if m.implied else "")
                + (f"\n- 🔎 Listed only (verify): {', '.join(m.listed_only)}" if m.listed_only else "")
                + f"\n- Experience: {m.candidate_years:g} yrs vs {m.required_years:g} required\n\n_Source: computed match analysis._")

    # 3. Pipeline attributes
    for col, pat in _PIPE_FIELDS.items():
        if re.search(pat, low):
            row = _pipeline_row(ws, cand.name)
            if row is None:
                return (f"**{cand.name}** is not in the recruitment pipeline tracker yet (no stage, CTC, notice period or "
                        "next-step data). Their CV profile is loaded, so I can rank, summarise or draft emails for them.")
            if col == "next_step":
                return f"**{cand.name}** - next step: **{row['next_step']}** on **{row['next_step_date']}** (stage: {row['stage']}). [PIPELINE]"
            label = col.replace("_", " ").replace("lpa", "(LPA)").replace("days", "(days)")
            return (f"**{cand.name}** - {label}: **{row[col]}**  \n"
                    f"Role: {row['applied_role']} · stage: {row['stage']} · next: {row['next_step']} ({row['next_step_date']}) [PIPELINE]")

    # 4. Profile attributes
    if re.search(r"\b(how (many|much)|total|years?)\b.*\b(experience|years?|exp)\b|\bexperience\b.*\b(have|has|does)\b|\bhow experienced\b", low):
        return (f"**{cand.name}** has **{cand.years_experience:g} years** of professional experience "
                f"(_{cand.experience_source or 'source unknown'}_).  \nCurrent title: {cand.current_title or 'n/a'}")
    if re.search(r"\b(where (is|does)|location|based|city|live)\b", low):
        return f"**{cand.name}** is based in **{cand.location or 'an unknown location (not in CV)'}**."
    if re.search(r"\b(email|e-mail|contact|phone|mobile|number)\b", low):
        return f"**{cand.name}** - email: {cand.email or 'not in CV'} · phone: {cand.phone or 'not in CV'}"
    if re.search(r"\b(education|degree|qualification|graduat|college|university|studied)\b", low):
        return f"**{cand.name}** - education: " + ("; ".join(cand.education) if cand.education else "not found in CV")
    if re.search(r"\b(certif)", low):
        return f"**{cand.name}** - certifications: " + ("; ".join(cand.certifications) if cand.certifications else "none listed")
    if re.search(r"\b(current (role|title|job|position)|designation|title|working as|what does .* do)\b", low):
        return f"**{cand.name}** - current title: **{cand.current_title or 'n/a'}**"
    if re.search(r"\b(skills?|tech stack|technologies)\b", low):
        return (f"**{cand.name}** - {len(cand.skills)} skills: {', '.join(cand.skills)}"
                + (f"\n\nImplied: {', '.join(cand.implied_skills)}" if cand.implied_skills else ""))
    return None


# ----------------------------------------------------------------- whole pipeline
def pipeline_query(message: str, ws: Workspace) -> str | None:
    """Exact filters over the pipeline tracker: stage, source, notice period, recruiter, role."""
    if ws.pipeline.empty or not _PEOPLE_QUESTION.search(message):
        return None  # e.g. "what's the assessment pass mark?" is a policy question, not a pipeline filter
    low = message.lower()
    df = ws.pipeline.copy()
    conds, desc = [], []

    stages = sorted(df["stage"].dropna().unique(), key=len, reverse=True)
    stage_hits = [s for s in stages if re.search(rf"\b{re.escape(s.lower())}\b", low)]
    if "assessment" in low and not stage_hits:
        stage_hits = [s for s in stages if "assessment" in s.lower()]
    if stage_hits:
        conds.append(df["stage"].isin(stage_hits)); desc.append("stage: " + ", ".join(stage_hits))

    sources = df["source"].dropna().unique()
    src_hits = [s for s in sources if re.search(rf"\b{re.escape(s.lower())}", low) or (s.lower() == "referral" and "referr" in low)]
    if src_hits:
        conds.append(df["source"].isin(src_hits)); desc.append("source: " + ", ".join(src_hits))

    if "notice" in low:
        nums = [int(n) for n in re.findall(r"(\d{1,3})\s*-?\s*days?", low)] or [int(n) for n in re.findall(r"\b(\d{1,3})\b", low)]
        if nums:
            n = nums[0]
            if re.search(r"\b(less than|under|below|within|at most|up to|<)\b", low):
                conds.append(df["notice_period_days"] <= n); desc.append(f"notice ≤ {n} days")
            elif re.search(r"\b(more than|over|above|at least|>)\b", low):
                conds.append(df["notice_period_days"] >= n); desc.append(f"notice ≥ {n} days")
            else:
                conds.append(df["notice_period_days"] == n); desc.append(f"notice = {n} days")
        elif re.search(r"\b(immediate|immediately|serving|no notice)\b", low):
            conds.append(df["notice_period_days"] == 0); desc.append("immediate joiners")

    recruiters = df["recruiter"].dropna().unique()
    rec_hits = [r for r in recruiters if r.lower() in low or r.split()[0].lower() in low.split()]
    if rec_hits and re.search(r"\b(recruiter|handled|handling|owned|assigned|managed|with)\b", low):
        conds.append(df["recruiter"].isin(rec_hits)); desc.append("recruiter: " + ", ".join(rec_hits))

    if not conds:
        return None
    job = ws.find_job(message)
    if job is not None:
        conds.append(df["applied_role"].str.lower() == job.title.lower()); desc.append(f"role: {job.title}")

    mask = conds[0]
    for c in conds[1:]:
        mask &= c
    res = df[mask]
    head = f"### Pipeline: {'; '.join(desc)}\n\n**{len(res)}** candidate(s) found [PIPELINE]\n"
    if res.empty:
        return head + "\nNo candidates in the tracker match this filter."
    rows = ["| Candidate | Role | Stage | Next step | Date | Notice (days) | Source | Recruiter |", "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        rows.append(f"| **{r['candidate_name']}** | {r['applied_role']} | {r['stage']} | {r['next_step']} | {r['next_step_date']} | "
                    f"{r['notice_period_days']} | {r['source']} | {r['recruiter']} |")
    return head + "\n" + "\n".join(rows) + "\n\n_Answered exactly from the pipeline tracker (all rows checked)._"


def count_query(message: str, ws: Workspace) -> str | None:
    low = message.lower()
    if not re.search(r"\b(how many|number of|count|total)\b", low) or not re.search(r"\b(candidates?|cvs?|resumes?|applicants?|people|profiles?|roles?|jobs?|jds?|openings?)\b", low):
        return None
    if re.search(r"\b(roles?|jobs?|jds?|openings?|positions?)\b", low) and not re.search(r"\bcandidates?\b", low):
        return f"There are **{len(ws.jobs)}** open roles loaded: " + ", ".join(j.title for j in ws.jobs.values()) + "."
    in_pipe = set(ws.pipeline["candidate_name"]) if not ws.pipeline.empty else set()
    not_in = [c.name for c in ws.candidates.values() if c.name not in in_pipe]
    return (f"There are **{len(ws.candidates)}** candidate profiles loaded (CVs), and **{len(ws.pipeline)}** rows in the pipeline tracker."
            + (f"\n\nNot yet in the pipeline tracker: {', '.join(not_in)}." if not_in else ""))
