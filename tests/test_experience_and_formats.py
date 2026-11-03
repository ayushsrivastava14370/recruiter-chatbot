"""Regression tests for real-world CV formats (experience, headings, names, implied skills).

Written after a real resume was mis-read as "2 yrs" instead of "4.5+ yrs" because of:
- the phrase "4.5+ years at <Company>" (no word "experience"),
- letter-spaced headings ("P R O F E S S I O N A L  E X P E R I E N C E") from a design template,
- an LLM estimate overwriting the value stated in the CV.
All CV text below is synthetic.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.extractor import _years_from_ranges, parse_candidate_rules
from src.llm import GeminiClient
from src.matcher import rank_candidates
from src.parser import _clean
from src.workspace import Workspace

DESIGN_TEMPLATE_CV = """PRIYA VERMA TEST
Senior Engineer — Full Stack & Generative AI
priya.test@example.com | +91 90000 00999 | Pune, India
P R O F E S S I O N A L S U M M A R Y
Senior Engineer with 4.5+ years at Example Corp spanning backend APIs and production RAG systems.
T E C H N I C A L S K I L L S
Python, FastAPI, LangChain, Qdrant, AutoGen
P R O F E S S I O N A L E X P E R I E N C E
Senior Engineer June 2022 – Present
Example Corp — Pune, India
• Built hybrid RAG pipelines in Python with Qdrant and multi-agent workflows on AutoGen.
Software Engineer Intern March 2022 – June 2022
Example Corp — Pune, India
• Built CRUD APIs.
E D U C A T I O N
Example Institute of Technology, Pune Aug 2018 – Jun 2022
Bachelor of Technology, Computer Science
Example Public School 2015 – 2017
"""


def months_since(year: int, month: int) -> float:
    t = date.today()
    return round(((t.year * 12 + t.month - 1) - (year * 12 + month - 1) + 1) / 12, 1)


def test_letter_spaced_headings_are_restored():
    cleaned = _clean(DESIGN_TEMPLATE_CV)
    assert "PROFESSIONAL EXPERIENCE" in cleaned and "TECHNICAL SKILLS" in cleaned and "EDUCATION" in cleaned


def test_design_template_cv_full_profile():
    p = parse_candidate_rules(_clean(DESIGN_TEMPLATE_CV))
    assert p.name == "Priya Verma Test"  # ALL-CAPS name normalised
    assert p.years_experience == 4.5 and p.experience_source == "stated in CV"
    assert p.location == "Pune, India"
    assert any("Bachelor of Technology" in e and "Institute" in e for e in p.education)  # degree + institution paired
    assert {"AI Agents", "Vector Databases", "RAG", "FastAPI"} <= set(p.skills)
    assert "NLP" in p.implied_skills  # RAG -> LLMs -> NLP


@pytest.mark.parametrize("text,expected", [
    ("Senior Engineer with 4.5+ years at Example Corp", 4.5),
    ("4.5+ years of professional experience", 4.5),
    ("Total Experience: 6 years 6 months", 6.5),
    ("Experience of 3 years in analytics", 3.0),
    ("Overall experience - 7 yrs", 7.0),
    ("ML engineer with 6 years of experience building NLP products", 6.0),
])
def test_stated_experience_phrasings(text, expected):
    cv = f"Test Person\nEngineer\ntest@example.com\n\nSUMMARY\n{text}\n\nEXPERIENCE\nEngineer | Co | Jan 2025 - Present\n"
    assert parse_candidate_rules(cv).years_experience == expected


@pytest.mark.parametrize("line,expected", [
    ("Engineer | Co | 07/2020 - 06/2023", 3.0),
    ("Engineer | Co | 2019-03 to 2021-02", 2.0),
    ("Engineer | Co | Jan 2018 – Dec 2019", 2.0),
    ("Engineer | Co | January 2016 — December 2016", 1.0),
    ("Engineer | Co | Mar'19 - Feb'21", 2.0),
    ("Engineer | Co | 2015 - 2018", 3.0),
])
def test_date_range_formats(line, expected):
    assert _years_from_ranges(line) == expected


def test_open_ended_ranges_and_internships_excluded():
    text = "Engineer | Co | Jan 2024 - Till date\nIntern | Co | Jun 2023 - Dec 2023\nTrainee | Co | 2022 - 2023"
    assert _years_from_ranges(text) == months_since(2024, 1)


def test_overlapping_jobs_not_double_counted():
    text = "Engineer | A | Jan 2018 - Dec 2020\nConsultant | B | Jan 2019 - Dec 2019"
    assert _years_from_ranges(text) == 3.0


def test_education_dates_never_counted():
    cv = ("Test Person\nEngineer\ntest@example.com\n\nEXPERIENCE\nEngineer | Co | Jan 2023 - Dec 2024\n\n"
          "EDUCATION\nB.Tech, Example University | Aug 2018 - Jun 2022\nSchool | 2015 - 2017\n")
    p = parse_candidate_rules(cv)
    assert p.years_experience == 2.0 and "employment dates" in p.experience_source


def test_mismatch_between_stated_and_dates_is_flagged():
    cv = "Test Person\nEngineer\ntest@example.com\n\nSUMMARY\n10 years of experience.\n\nEXPERIENCE\nEngineer | Co | Jan 2023 - Dec 2024\n"
    p = parse_candidate_rules(cv)
    assert p.years_experience == 10 and any("please verify" in w for w in p.warnings)


def test_fresher_with_only_internship():
    cv = "Test Person\nGraduate\ntest@example.com\n\nEXPERIENCE\nML Intern | Co | Jan 2026 - Jun 2026\n"
    p = parse_candidate_rules(cv)
    assert p.years_experience == 0 and any("No professional experience" in w for w in p.warnings)


def test_reupload_same_filename_reanalyses_and_delete_works():
    ws = Workspace(GeminiClient(api_key=""))
    ws.load_sample_data()
    n = len(ws.candidates)
    p1 = ws.add_candidate_text(_clean(DESIGN_TEMPLATE_CV), "my_cv.pdf")
    p2 = ws.add_candidate_text(_clean(DESIGN_TEMPLATE_CV), "my_cv.pdf")  # same file again -> replaced, not rejected
    assert len(ws.candidates) == n + 1 and p1.id == p2.id and any("Re-analysed" in w for w in p2.warnings)
    ws.remove_candidate(p2.id)
    assert len(ws.candidates) == n and p2.id not in ws.candidates


def test_implied_skill_is_credited_but_marked():
    ws = Workspace(GeminiClient(api_key=""))
    ws.load_sample_data()
    p = ws.add_candidate_text(_clean(DESIGN_TEMPLATE_CV), "my_cv.pdf")
    res, _ = rank_candidates(ws.jobs["machine-learning-engineer-nlp-llms"], [p])
    assert "NLP" in res[0].matched_must and "NLP" in res[0].implied
