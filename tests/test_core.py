"""Offline tests (no API key required): parsing, skill extraction, ranking, RAG and routing."""
from __future__ import annotations

import pytest

from src.assistant import RecruiterAssistant, detect_email_type, detect_intent
from src.extractor import parse_candidate_rules, parse_job_rules
from src.llm import GeminiClient
from src.parser import extract_text
from src.rag import chunk_text
from src.skills import extract_skills
from src.workspace import Workspace
from src import config


@pytest.fixture(scope="module")
def ws() -> Workspace:
    w = Workspace(GeminiClient(api_key=""))  # force offline mode
    w.load_sample_data()
    return w


def test_skill_extraction_is_grounded():
    skills = extract_skills("Built RAG apps with LangChain, FAISS and PyTorch; deployed on k8s. Ran A/B testing.")
    assert {"RAG", "LangChain", "Vector Databases", "PyTorch", "Kubernetes"} <= set(skills)
    assert "Software Testing" not in skills  # 'A/B testing' is not software testing
    assert "Java" not in extract_skills("JavaScript developer")  # word boundaries


def test_all_formats_parse():
    for path in list(config.CV_DIR.glob("*")) + list(config.JD_DIR.glob("*")):
        assert len(extract_text(path)) > 200, path


def test_candidate_profile_fields():
    text = extract_text(config.CV_DIR / "aarav_mehta.pdf")
    p = parse_candidate_rules(text, "aarav_mehta.pdf")
    assert p.name == "Aarav Mehta"
    assert p.email == "aarav.mehta@example.com"
    assert p.years_experience == 6
    assert "PyTorch" in p.skills and "RAG" in p.skills


def test_job_sections():
    job = parse_job_rules(extract_text(config.JD_DIR / "ml_engineer_nlp.txt"))
    assert job.min_years == 3
    assert {"Python", "NLP", "LLMs", "PyTorch", "Docker"} <= set(job.must_have)
    assert "RAG" in job.nice_to_have and "RAG" not in job.must_have


@pytest.mark.parametrize("job_id,expected_top", [
    ("machine-learning-engineer-nlp-llms", "Aarav Mehta"),
    ("data-analyst", "Neha Iyer"),
    ("backend-engineer-python", "Ananya Rao"),
])
def test_ranking_top_candidate(ws, job_id, expected_top):
    results = ws.rank(job_id)
    assert results[0].candidate_name == expected_top
    assert results == sorted(results, key=lambda r: -r.total_score)
    assert all(0 <= r.total_score <= 100 for r in results)


def test_gaps_are_reported(ws):
    r = ws.match_for("data-analyst", "ishita-banerjee")
    assert "Power BI" in r.missing_must and "Python" in r.missing_must


def test_chunking_overlap():
    chunks = chunk_text("A" * 2000 + "\n\n" + "B" * 300, size=900, overlap=100)
    assert all(len(c) <= 1100 for c in chunks) and len(chunks) >= 3


def test_retrieval_finds_policy(ws):
    hits = ws.index.search("salary band compensation for data analyst")
    assert any(c.doc.doc_type == "policy" for c, _ in hits[:3])


@pytest.mark.parametrize("msg,intent", [
    ("Rank the top 5 candidates for the ML role", "rank"),
    ("Generate interview questions for Arjun", "questions"),
    ("Draft an interview invitation for Neha", "email"),
    ("Summarise Ananya Rao", "summary"),
    ("Which candidates have a 90 day notice period?", "qa"),
])
def test_intents(msg, intent):
    assert detect_intent(msg) == intent


def test_email_type_detection():
    assert detect_email_type("send a rejection to Kabir") == "rejection"
    assert detect_email_type("assessment instructions for Sara") == "assessment"
    assert detect_email_type("invite Neha for interview") == "interview_invite"


def test_assistant_offline_end_to_end(ws):
    bot = RecruiterAssistant(ws)
    assert "Aarav Mehta" in bot.handle("Rank top 3 candidates for Machine Learning Engineer").text
    reply = bot.handle("Draft an interview invite for Ananya Rao for the Backend Engineer role")
    assert reply.intent == "email" and "Ananya" in reply.text
    assert bot.handle("Interview questions for Sara Thomas", active_job_id="machine-learning-engineer-nlp-llms").intent == "questions"
