"""Central configuration. Values come from (in priority order):
1. Explicit overrides (e.g. API key typed into the UI sidebar)
2. Streamlit secrets (used on Streamlit Community Cloud)
3. Environment variables / .env file
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CV_DIR = DATA_DIR / "cvs"
JD_DIR = DATA_DIR / "jds"
POLICY_DIR = DATA_DIR / "policies"
PIPELINE_CSV = DATA_DIR / "pipeline.csv"
CACHE_DIR = ROOT_DIR / ".cache"

load_dotenv(ROOT_DIR / ".env")


def _secret(name: str, default: str = "") -> str:
    """Read a setting from env vars, falling back to Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:  # st.secrets raises if no secrets file exists
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


GEMINI_API_KEY = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")
# Primary generation model + fallbacks tried in order if the primary is unavailable.
# Default: Flash-Lite - lowest latency and highest free-tier limits; quality is ample for extraction,
# summarisation and drafting. Switch to full Flash in the sidebar when it is not overloaded.
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-flash-lite-latest")
# Tried in order when a model is busy (503), rate-limited (429) or retired (404).
GEMINI_FALLBACK_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-2.5-flash-lite"]
GEMINI_EMBED_MODEL = _secret("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768

# Default weights for the hybrid match score (must sum to 1; UI lets the user tune them).
DEFAULT_WEIGHTS = {"skills": 0.5, "semantic": 0.3, "experience": 0.2}

# Privacy: mask emails/phone numbers before sending document text to the LLM (data minimisation)
REDACT_PII = _secret("REDACT_PII", "true").lower() != "false"

# Security: optional access code for the hosted demo, per-session LLM rate limit, upload limits
APP_PASSWORD = _secret("APP_PASSWORD", "")
RATE_LIMIT_PER_MIN = int(_secret("RATE_LIMIT_PER_MIN", "15") or 15)
MAX_UPLOAD_FILES = 20

# RAG settings
CHUNK_SIZE = 900  # characters
CHUNK_OVERLAP = 150
TOP_K = 6

# Recruiter / company identity used in email drafts (synthetic).
COMPANY_NAME = _secret("COMPANY_NAME", "Nimbus Analytics")
RECRUITER_NAME = _secret("RECRUITER_NAME", "Priya Sharma")
RECRUITER_TITLE = _secret("RECRUITER_TITLE", "Talent Acquisition Partner")
