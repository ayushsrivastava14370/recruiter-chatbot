@echo off
REM One-click setup + launch for Windows
if not exist .venv (
    python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
)
if not exist .env copy .env.example .env
.venv\Scripts\streamlit run app.py
