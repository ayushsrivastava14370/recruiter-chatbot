#!/usr/bin/env bash
# One-click setup + launch for macOS / Linux
set -e
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
.venv/bin/streamlit run app.py
