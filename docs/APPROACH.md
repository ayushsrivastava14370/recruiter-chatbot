# RecruitAI: Technical Approach

## 1. Problem framing
Recruiters spend most of their time on repetitive work: reading CVs against a JD, shortlisting, writing summaries, preparing interview questions, sending emails and answering status questions. The goal is an assistant that automates these tasks **without sacrificing accuracy or trust**. Every output must be traceable to the supplied data.

## 2. Design principles
1. **Deterministic where possible, generative where valuable.** Extraction of facts (skills, years, contact details) and scoring are rule- and math-based, so they're reproducible and explainable. The LLM is used for language tasks: summaries, questions, emails and natural-language Q&A.
2. **Verify, then trust.** Any fact the LLM extracts (e.g. a skill) is kept only if it can be located in the source text.
3. **Ground every answer.** RAG with citations, plus an explicit refusal when the data has no answer.
4. **Graceful degradation.** No key, a rate limit or a model outage never breaks the app. Each feature has an offline fallback.

## 3. Pipeline

### 3.1 Ingestion & parsing
- `pypdf` / `python-docx` extract the text, which is then normalised.
- **Rule-based parser:** regex for email, phone, location and "X years of experience". Section detection drives education and certifications. Years of experience fall back to the **union of employment date ranges**, so overlapping jobs aren't double-counted.
- **Skill taxonomy:** 70+ canonical skills with aliases (e.g. `k8s → Kubernetes`, `FAISS → Vector Databases`). Word-boundary regexes handle tokens like `C++` and `Node.js`, and stop `Java` from matching `JavaScript`.
- **JD parser:** splits *Must have / Requirements* from *Nice to have / Preferred* sections and reads the minimum years.
- **Optional Gemini extraction:** structured JSON output via a Pydantic `response_schema`, merged with the rule-based output after the grounding check.

### 3.2 Matching & ranking
`score = 0.5·skill_coverage + 0.3·semantic_similarity + 0.2·experience_fit`. The weights are adjustable in the UI.

| Component | Definition | Why |
|---|---|---|
| Skill coverage | (2·matched_must + matched_nice) / (2·#must + #nice) | Recruiters think in must-haves; fully explainable |
| Semantic similarity | cosine(JD, CV) with Gemini `gemini-embedding-001`, calibrated to 0-1 | Captures related experience that keywords miss (e.g. "BERT fine-tuning" vs "NLP") |
| Experience fit | min(1, years / required years) | Simple, transparent; no penalty for seniority |

The output includes matched and missing must-have and nice-to-have skills, extra skills, a recommendation band and a plain-English explanation. **No LLM is used in scoring**, so the ranking is stable and auditable, which matters for fair hiring.

### 3.3 Generation (summaries, questions, emails)
- Prompts contain the CV, the JD **and the computed match analysis**, so the LLM writes about real strengths and real gaps.
- Interview questions use a JSON schema (`category, question, rationale, good_answer_signals`). The mix is about 40% technical (tied to CV projects), 25% gap probes, 20% role scenarios and 15% behavioural (STAR).
- Emails: the recruiter's logistics (date, time, link, deadline) are passed verbatim and the model is told not to change them. Rejections are kind and never list deficiencies.

### 3.4 Retrieval-Augmented Q&A
- Paragraph-aware chunking (900 chars, 150 overlap) of CVs, JDs, policies, plus one record per pipeline row. Each chunk is prefixed with its source.
- Top-6 cosine retrieval (Gemini embeddings with `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types).
- The **complete pipeline table is always included**. Aggregations ("how many are at the interview stage?") need every row, which top-k retrieval can't guarantee.
- The system prompt requires `[S#]` citations and the exact refusal string when the answer isn't present. The sources are shown in the UI for verification.

### 3.5 Chat orchestration
A lightweight router (regex intents + RapidFuzz entity matching on candidate names and job titles) picks one of: rank, summary, questions, email, compare, Q&A. It also resolves follow-ups like "draft an invite for her" using the last candidate discussed. This avoids an extra LLM call per message, which saves latency and free-tier quota and is deterministic.

## 4. Model selection

| Need | Choice | Alternatives considered |
|---|---|---|
| Generation | **Gemini Flash-Lite** (`gemini-flash-lite-latest`, free tier) with automatic failover to Flash models | GPT-4o-mini (paid), local Llama via Ollama (heavy for reviewers to install). Flash-Lite gives native JSON schema output, a 1M-token context, about 2 s latency, the highest free quota and no cost. During development the full Flash models returned 503/504 under load and `gemini-2.5-flash` was retired for new keys (404), which is why there is failover plus `-latest` aliases. |
| Embeddings | **gemini-embedding-001** (768-d) | sentence-transformers MiniLM (local, but needs a ~2 GB PyTorch install); OpenAI embeddings (paid) |
| Offline fallback | **TF-IDF (scikit-learn)** | BM25. TF-IDF is already a dependency and good enough as a fallback |
| Vector store | **NumPy cosine** | FAISS/Chroma. With fewer than 1k chunks, exact search is instant; swapping in a vector DB is a one-class change |

## 5. Reliability engineering
- Disk cache for all LLM and embedding calls (SHA-256 of model + prompt), which makes repeat demos instant and deterministic.
- Exponential backoff on 429/5xx errors, plus automatic fallback across free Gemini models.
- 69 offline pytest tests: parsing of every file format, every edge-case file, every security control, skill extraction (negation, boundaries), expected top candidates per role, gap reporting, retrieval, intent routing and the end-to-end assistant.

## 6. Security
See `docs/INTERVIEW_QA.md` section D and the in-app 🛡️ Security tab. In short: injected instructions in CVs are detected and **stripped before scoring**; documents are fenced as untrusted data; jailbreaks are refused before any LLM call; model output is sanitised against exfiltration and XSS; PII is masked before it leaves the machine; uploads are validated (magic bytes, size, pages, zip bombs); and there is per-session rate limiting plus an optional access code.

## 7. Responsible AI
- Only synthetic people, companies and contact details are used.
- Prompts forbid using or inferring protected characteristics. Ranking uses only skills, experience and text similarity.
- Human in the loop: emails are drafts to edit and approve. Rankings show their reasoning, so recruiters can challenge them.

## 8. Limitations & roadmap
OCR for scanned CVs; LLM-assisted taxonomy expansion; persistent DB with authentication; bias audits (e.g. score parity across synthetic demographic variants); ATS and calendar integration; evaluation set for RAG faithfulness (e.g. RAGAS).
