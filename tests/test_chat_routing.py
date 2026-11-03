"""Chat routing regression suite (from the full-app audit).

Every factual / filter question with an exact answer must be answered deterministically from
structured data (profiles, pipeline) - never left to top-k semantic retrieval, which can miss rows.
"""
from __future__ import annotations

import pytest

from src.assistant import RecruiterAssistant
from src.llm import GeminiClient
from src.parser import _clean
from src.workspace import Workspace
from tests.test_experience_and_formats import DESIGN_TEMPLATE_CV

ML = "machine-learning-engineer-nlp-llms"


@pytest.fixture(scope="module")
def bot() -> RecruiterAssistant:
    ws = Workspace(GeminiClient(api_key=""))
    ws.load_sample_data()
    ws.add_candidate_text(_clean(DESIGN_TEMPLATE_CV), "priya_cv.pdf")  # an uploaded CV that is NOT in the pipeline
    return RecruiterAssistant(ws)


def ask(bot, q, job=ML):
    return bot.handle(q, [], job)


@pytest.mark.parametrize("q,intent,must_contain,must_not", [
    # filters over ALL profiles (including uploads)
    ("i want a candidate that has 4.5 year of experience", "filter", ["Priya Verma Test"], ["Meera"]),
    ("candidates with more than 5 years of experience", "filter", ["Vikram Joshi", "Aarav Mehta"], ["Ananya"]),
    ("show candidates with less than 2 years", "filter", ["Kabir Singh", "Fatima Sheikh"], ["Aarav"]),
    ("candidates between 3 and 5 years", "filter", ["Ananya Rao", "Priya Verma Test"], ["Vikram"]),
    ("who knows RAG?", "filter", ["Aarav Mehta", "Arjun Desai", "Priya Verma Test"], ["Neha"]),
    ("anyone with Kubernetes experience in Hyderabad?", "filter", ["Ananya Rao"], ["Aarav"]),
    ("list freshers", "filter", ["Fatima Sheikh"], ["Aarav"]),
    ("candidates in Bengaluru", "filter", ["Aarav Mehta", "Meera Nair"], ["Ananya"]),
    # exact pipeline filters
    ("Which candidates have a 90-day notice period?", "pipeline", ["Arjun Desai", "Vikram Joshi"], ["Aarav"]),
    ("How many candidates are at the Technical Interview stage and who are they?", "pipeline", ["**2**", "Aarav Mehta", "Arjun Desai"], []),
    ("Who are the candidates in HR screening?", "pipeline", ["Rohan Kulkarni", "Meera Nair"], []),
    ("which candidates came from referrals?", "pipeline", ["Aarav Mehta", "Ishita Banerjee", "Vikram Joshi"], []),
    ("how many candidates do we have in total?", "count", ["**13**", "Priya Verma Test"], []),
    # exact candidate facts
    ("how many years of experience does Priya Verma Test have?", "fact", ["4.5 years", "stated in CV"], []),
    ("Does Aarav know PyTorch?", "fact", ["✅", "PyTorch"], []),
    ("Does Neha know Kubernetes?", "fact", ["❌", "Kubernetes"], []),
    ("What is Neha's current stage?", "fact", ["Offer"], []),
    ("what is the expected CTC of Ananya Rao?", "fact", ["27"], []),
    ("what stage is Priya Verma Test at?", "fact", ["not in the recruitment pipeline"], []),
    ("what are Arjun's skill gaps for the ML role?", "fact", ["Missing must-have", "PyTorch"], []),
    ("where is Dev Malhotra based?", "fact", ["Gurugram"], []),
    # not hijacked: policy questions go to RAG
    ("what's the assessment pass mark?", "qa", [], ["### Pipeline"]),
    ("What is the salary band for the Data Analyst role?", "qa", [], ["### Pipeline"]),
    # actions
    ("Rank the top 5 candidates for the Machine Learning Engineer role", "rank", ["Aarav Mehta"], []),
    ("rank candidates with 3+ years for the ML role", "rank", ["filtered: at least 3 years"], ["Fatima"]),
    ("Compare Aarav and Arjun for the ML Engineer role", "compare", ["Aarav Mehta", "Arjun Desai"], []),
    # conversation hygiene
    ("hi", "help", ["RecruitAI"], []),
    ("Write me a poem about the weather", "off_topic", ["only help with recruitment"], []),
    ("Summarise John Smith", "summary", ["John Smith"], []),
])
def test_routing(bot, q, intent, must_contain, must_not):
    r = ask(bot, q)
    assert r.intent == intent, (q, r.intent, r.text[:200])
    for s in must_contain:
        assert s in r.text, (q, s, r.text[:400])
    for s in must_not:
        assert s not in r.text, (q, s, r.text[:400])
