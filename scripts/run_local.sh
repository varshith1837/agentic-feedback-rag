#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
if [ ! -f runtime/revenue.sqlite ]; then
  python -m revenue_risk.cli prepare
  python -m revenue_risk.cli extract --provider rules
fi
exec python -m streamlit run app/streamlit_app.py
