"""Chat orchestrator: routes a recruiter's message to the right tool.

Routing is rule-based (regex intents + fuzzy entity matching). That makes it
deterministic, instant and free (no extra LLM call per message). Anything that
is not an action falls through to grounded RAG question answering.

Intents: rank | summary | questions | email | compare | qa
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .generator import EMAIL_TYPES, candidate_summary, draft_email, interview_questions, stage_conflict
from .guardrails import JAILBREAK_REFUSAL, check_protected_query, detect_jailbreak, protected_refusal
from .workspace import Workspace

STOPWORDS = {"draft", "write", "send", "generate", "summarise", "summarize", "create", "rank", "interview", "email", "invite",
             "invitation", "questions", "question", "tell", "who", "what", "for", "the", "an", "a", "please", "can", "give",
             "rejection", "assessment", "offer", "profile", "summary", "compare", "show", "me", "role", "engineer", "analyst",
             "i", "hi", "hello", "make", "prepare", "list"}

INTENT_PATTERNS: list[tuple[str, str]] = [
    ("questions", r"\b(interview questions?|questions? (for|to ask)|what (should|can) i ask)"),
    ("email", r"\b(email|e-mail|mail|invite|invitation|draft|write to|reject|rejection|assessment instruction)"),
    ("rank", r"\b(rank|ranking|shortlist|top \d+|best (candidates?|fit|match)|who (is|are) the best|match(ing)? candidates|suitable candidates)"),
    ("compare", r"\b(compare|comparison|versus|vs\.?)\b"),
    ("summary", r"\b(summar|profile of|overview of|tell me about|who is)"),
]


@dataclass
class AssistantReply:
    text: str
    intent: str
    payload: dict[str, Any] = field(default_factory=dict)


def detect_intent(message: str) -> str:
    low = message.lower()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, low):
            return intent
    return "qa"


def detect_email_type(message: str) -> str:
    low = message.lower()
    if re.search(r"reject|regret|unsuccessful|not selected", low):
        return "rejection"
    if re.search(r"assessment|test|assignment|coding challenge|take-home", low):
        return "assessment"
    if re.search(r"offer", low):
        return "offer_call"
    if re.search(r"shortlist|next step|update", low):
        return "shortlist_update"
    return "interview_invite"


class RecruiterAssistant:
    def __init__(self, ws: Workspace, weights: dict[str, float] | None = None):
        self.ws = ws
        self.weights = weights

    def _job(self, message: str, active_job_id: str | None):
        job = self.ws.find_job(message)
        if job is None and active_job_id in self.ws.jobs:
            job = self.ws.jobs[active_job_id]
        if job is None and len(self.ws.jobs) == 1:
            job = next(iter(self.ws.jobs.values()))
        return job

    def handle(self, message: str, history: list[dict] | None = None, active_job_id: str | None = None) -> AssistantReply:
        ws, llm = self.ws, self.ws.llm
        message = (message or "").strip()[:2000]  # guard against empty / huge pastes
        if not message:
            return AssistantReply("Please type a question or request.", "empty")
        if not ws.candidates and not ws.jobs:
            return AssistantReply("No data loaded yet. Upload CVs and a job description from the sidebar first.", "empty")

        # Guardrail 1: direct prompt injection / jailbreak / secret extraction typed into the chat
        if (hit := detect_jailbreak(message)):
            return AssistantReply(JAILBREAK_REFUSAL, "security", {"security_event": f"Jailbreak / prompt-extraction attempt blocked: '{hit}'"})
        # Guardrail 2: refuse to filter/rank/judge on protected characteristics
        if (category := check_protected_query(message)):
            return AssistantReply(protected_refusal(category), "guardrail", {"security_event": f"Discriminatory request blocked ({category})"})

        intent = detect_intent(message)
        job = self._job(message, active_job_id)

        # Ambiguity: "Rohan" when there are two Rohans -> ask, don't guess
        matches = ws.find_candidates(message)
        if len(matches) > 1 and intent in ("summary", "questions", "email"):
            return AssistantReply("I found more than one matching candidate: " + ", ".join(c.name for c in matches)
                                  + ". Which one do you mean? Please use the full name.", "clarify")
        cand = matches[0] if matches else None

        # Pronoun follow-ups ("draft an invite for her") -> last candidate discussed.
        # Only used when the message actually contains a reference word, so an unknown name is never silently replaced.
        if cand is None and history and re.search(r"\b(her|him|his|she|he|them|they|their|this candidate|that candidate|same candidate)\b", message.lower()):
            for m in reversed(history):
                if m.get("candidate_id") in ws.candidates:
                    cand = ws.candidates[m["candidate_id"]]
                    break

        if intent == "rank":
            if not job:
                return AssistantReply("Which role should I rank candidates for? Available: " + ", ".join(j.title for j in ws.jobs.values()), "rank")
            if not ws.candidates:
                return AssistantReply("There are no candidates loaded yet - upload some CVs first.", "rank")
            n = int(m.group(1)) if (m := re.search(r"top (\d+)", message.lower())) else 5
            n = max(1, min(n, len(ws.candidates)))
            results = ws.rank(job.id, self.weights)[:n]
            lines = [f"### Top {len(results)} candidates for **{job.title}**\n",
                     "| # | Candidate | Score | Matched must-haves | Gaps (must-have) | Recommendation |",
                     "|---|---|---|---|---|---|"]
            for r in results:
                flag = " ⚠️" if ws.candidates[r.candidate_id].injection_flags else ""
                lines.append(f"| {r.rank} | {r.candidate_name}{flag} | **{r.total_score:.0f}** | {', '.join(r.matched_must) or '-'} | "
                             f"{', '.join(r.missing_must) or 'None'} | {r.recommendation} |")
            if any(ws.candidates[r.candidate_id].injection_flags for r in results):
                lines.append("\n> ⚠️ = CV contains hidden/instruction-like text aimed at AI screeners. It was ignored by the scoring - review manually.")
            lines.append(f"\n_Score = skills {int((self.weights or {}).get('skills', .5)*100)}% + semantic + experience. "
                         f"Semantic similarity via {ws.ranking_backend}. Open the **Match & Rank** tab for the full breakdown._")
            return AssistantReply("\n".join(lines), "rank", {"job_id": job.id})

        if intent in ("summary", "questions", "email") and cand is None:
            names = ", ".join(c.name for c in list(ws.candidates.values())[:15])
            unknown = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", message)
            unknown = [u for u in unknown if u.lower() not in STOPWORDS and not ws.find_job(u, 90)]
            prefix = f"I couldn't find a candidate called **{unknown[-1]}** in the data. " if unknown else "Which candidate do you mean? "
            return AssistantReply(f"{prefix}Available candidates: {names}.", intent)

        match = ws.match_for(job.id, cand.id, self.weights) if (job and cand) else None
        integrity = ""
        if cand and cand.injection_flags:
            integrity = ("> 🛡️ **Integrity warning:** this CV contains text that tries to instruct the AI "
                         f"(e.g. _\"{cand.injection_flags[0][:90]}...\"_). It was treated as data and ignored.\n\n")

        if intent == "summary":
            text = candidate_summary(cand, job, match, llm)
            header = f"### {cand.name}" + (f" - fit for {job.title}: {match.total_score:.0f}/100" if match else "")
            return AssistantReply(f"{integrity}{header}\n\n{text}", "summary", {"candidate_id": cand.id})

        if intent == "questions":
            if not job:
                return AssistantReply("For which role? " + ", ".join(j.title for j in ws.jobs.values()), "questions")
            qs = interview_questions(cand, job, match, llm)
            lines = [f"### Interview questions for {cand.name} - {job.title}\n"]
            for i, q in enumerate(qs, 1):
                lines.append(f"**{i}. [{q.category}]** {q.question}\n   - _Why:_ {q.rationale}\n   - _Look for:_ {q.good_answer_signals}")
            return AssistantReply("\n".join(lines), "questions", {"candidate_id": cand.id})

        if intent == "email":
            etype = detect_email_type(message)
            draft = draft_email(etype, cand, job, llm, match)
            warn = stage_conflict(etype, ws.pipeline_stage(cand.name))
            text = ((f"> ⚠️ {warn}\n\n" if warn else "")
                    + f"### ✉️ {EMAIL_TYPES[etype]} draft - {cand.name}\n**To:** {cand.email or '⚠️ no email address in CV - add it manually'}\n"
                    f"**Subject:** {draft.subject}\n\n```text\n{draft.body}\n```\n"
                    "_Edit dates/links in the **Email Drafts** tab before sending. A human must review every email._")
            return AssistantReply(text, "email", {"candidate_id": cand.id, "draft": draft.model_dump()})

        if intent == "compare" and job:
            mentioned = [c for c in ws.candidates.values() if any(p.lower() in message.lower() for p in c.name.split() if len(p) > 2)]
            if len(mentioned) >= 2:
                rows = ["| Candidate | Score | Years | Matched must-haves | Missing must-haves |", "|---|---|---|---|---|"]
                for c in mentioned:
                    r = ws.match_for(job.id, c.id, self.weights)
                    if r:
                        rows.append(f"| {c.name} | {r.total_score:.0f} | {r.candidate_years:g} | {', '.join(r.matched_must) or '-'} | {', '.join(r.missing_must) or 'None'} |")
                return AssistantReply(f"### Comparison for {job.title}\n\n" + "\n".join(rows), "compare")

        # ---- Default: grounded Q&A over all data
        extra = ""
        if job and re.search(r"\b(fit|match|best|score|rank|suitable)", message.lower()):
            extra = "\n".join(r.explanation() for r in ws.rank(job.id, self.weights)[:8])
        ans = ws.index.answer(message, history, extra_context=extra)
        payload = {"sources": [(lbl, c.doc.title, c.doc.doc_type, round(s, 3), c.text) for lbl, c, s in ans.sources]}
        if cand:
            payload["candidate_id"] = cand.id
        return AssistantReply(ans.answer, "qa", payload)
