@echo off
cd /d "%~dp0\.."
if not exist .venv (py -3.11 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if not exist .env (copy .env.example .env >nul)
if not exist runtime\revenue.sqlite (
  python -m revenue_risk.cli prepare
  python -m revenue_risk.cli extract --provider rules
)
python -m streamlit run app/streamlit_app.py
