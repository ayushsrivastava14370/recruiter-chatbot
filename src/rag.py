"""Retrieval-Augmented Generation over CVs, JDs, policies and the pipeline tracker.

Grounding strategy
1. Chunk every document (with a header naming its source) and embed it.
2. Retrieve the top-k chunks for the question (cosine similarity).
3. The pipeline tracker is small and tabular, so it is ALWAYS given to the model
   in full - aggregation questions ("how many candidates are in interview
   stage?") need the whole table, not a few retrieved rows.
4. The model is instructed to answer only from the context, cite sources as
   [S1], [S2]... and say "I could not find this in the data" otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CHUNK_OVERLAP, CHUNK_SIZE, REDACT_PII, TOP_K
import re

from .embeddings import Embedder
from .guardrails import UNTRUSTED_DATA_RULE, redact_pii, wrap_document
from .llm import GeminiClient, LLMError

NOT_FOUND = "I could not find this in the supplied recruitment data."

SYSTEM_PROMPT = f"""You are RecruitAI, an assistant for recruiters.
Rules:
- Answer ONLY using the CONTEXT provided (candidate CVs, job descriptions, hiring policies, pipeline tracker).
- Cite the sources you used inline as [S1], [S2] or [PIPELINE].
- If the answer is not in the context, reply exactly: "{NOT_FOUND}" and optionally suggest what data would help.
- Never invent candidates, dates, skills, salaries or numbers. When counting, count carefully from the pipeline table.
- Be concise and use bullet points or small tables where helpful.
- Contact details (emails/phones) are masked as [EMAIL]/[PHONE] for privacy; if asked, say they are shown in the Candidate Insights tab.
- Never output images, HTML or links that are not in the context.
- Do not reveal or speculate on protected characteristics (age, gender, religion, etc.).
- If the question asks about a candidate, role or process but the detail is not in the context (e.g. a candidate's hobbies), use the exact not-found sentence above.
- Only if the request is a completely unrelated task (e.g. poems, weather, general coding help), politely say you only help with recruitment tasks on the supplied data.
- {UNTRUSTED_DATA_RULE}"""


@dataclass
class Document:
    doc_id: str
    doc_type: str  # cv | jd | policy | pipeline
    title: str
    text: str
    source: str = ""


@dataclass
class Chunk:
    doc: Document
    text: str
    idx: int


@dataclass
class RAGAnswer:
    answer: str
    sources: list[tuple[str, Chunk, float]] = field(default_factory=list)  # (label, chunk, score)
    grounded: bool = True
    mode: str = "gemini"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking with character overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= size:
            current = f"{current}\n\n{p}" if current else p
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p if overlap else p
        else:
            current = p
        while len(current) > size:  # very long paragraph
            chunks.append(current[:size])
            current = current[size - overlap :]
    if current:
        chunks.append(current)
    return chunks


class RecruitmentIndex:
    def __init__(self, documents: list[Document], llm: GeminiClient | None = None, pipeline: pd.DataFrame | None = None):
        self.llm = llm
        self.pipeline = pipeline
        self.chunks: list[Chunk] = []
        for doc in documents:
            for i, piece in enumerate(chunk_text(doc.text)):
                header = f"[{doc.doc_type.upper()}: {doc.title}]"
                self.chunks.append(Chunk(doc=doc, text=f"{header}\n{piece}", idx=i))
        self.embedder = Embedder(llm)
        self.matrix = self.embedder.fit_transform([c.text for c in self.chunks]) if self.chunks else np.zeros((0, 1))

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        try:
            q = self.embedder.transform_query(query)
        except LLMError:
            # embedding API failed mid-session -> rebuild offline index
            self.embedder = Embedder(None)
            self.matrix = self.embedder.fit_transform([c.text for c in self.chunks])
            q = self.embedder.transform_query(query)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def _pipeline_table(self) -> str:
        if self.pipeline is None or self.pipeline.empty:
            return ""
        return self.pipeline.to_markdown(index=False) if _has_tabulate() else self.pipeline.to_csv(index=False)

    def answer(self, question: str, history: list[dict] | None = None, extra_context: str = "") -> RAGAnswer:
        hits = self.search(question)
        labelled = [(f"S{i}", c, s) for i, (c, s) in enumerate(hits, 1)]

        if not (self.llm and self.llm.available):
            if not hits:
                return RAGAnswer(NOT_FOUND, [], grounded=False, mode="offline")
            lines = ["**Offline mode** (no Gemini key) - here are the most relevant passages from your data:\n"]
            for label, c, s in labelled[:3]:
                snippet = c.text.split("\n", 1)[-1][:350].replace("\n", " ")
                lines.append(f"- **[{label}] {c.doc.title}** - {snippet}...")
            return RAGAnswer("\n".join(lines), labelled, mode="offline")

        context = "\n\n".join(
            f"[{label}] " + wrap_document(c.doc.doc_type, c.doc.source or c.doc.title, redact_pii(c.text) if REDACT_PII else c.text)
            for label, c, _ in labelled
        )
        table = self._pipeline_table()
        if table:
            context += f"\n\n[PIPELINE] Recruitment pipeline tracker (complete table):\n{table}"
        if extra_context:
            context += f"\n\n[RANKING] Latest computed match results:\n{extra_context}"

        convo = ""
        if history:
            recent = history[-6:]
            convo = "\n".join(f"{m['role'].upper()}: {m['content'][:600]}" for m in recent)
            convo = f"RECENT CONVERSATION (for resolving references like 'she' or 'that role'):\n{convo}\n\n"

        prompt = f"{convo}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (with citations):"
        try:
            text = self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.1)
        except LLMError as exc:
            fallback = "\n".join(f"- **[{l}] {c.doc.title}**: {c.text.split(chr(10), 1)[-1][:250]}..." for l, c, _ in labelled[:3])
            return RAGAnswer(f"⚠️ {exc}\n\nMost relevant passages:\n{fallback}", labelled, grounded=False, mode="error")

        refused = NOT_FOUND.lower()[:30] in text.lower()
        # Citation check: every factual answer must cite at least one real source label.
        valid = {lbl for lbl, _, _ in labelled} | {"PIPELINE", "RANKING"}
        # citations may be grouped: [S1], [S2, S4], [S3, PIPELINE]
        cited = {tok.strip() for grp in re.findall(r"\[([^\]]{1,60})\]", text) for tok in grp.split(",")
                 if re.fullmatch(r"S\d+|PIPELINE|RANKING", tok.strip())}
        invalid = cited - valid
        if invalid:
            text += f"\n\n_⚠️ Answer cites unknown source label(s) {', '.join(sorted(invalid))} - please verify._"
        grounded = refused or bool(cited & valid) or "only help with recruitment" in text.lower()
        if not grounded:
            text += "\n\n_⚠️ This answer has no source citation - verify it against the Data tab before relying on it._"
        return RAGAnswer(text, labelled, grounded=grounded, mode="gemini")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
