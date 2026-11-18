#!/bin/sh
set -eu
if [ ! -f /app/runtime/revenue.sqlite ]; then
  echo 'Preparing database from data/brazil.zip ...'
  python -m revenue_risk.cli prepare
  echo 'Creating clearly labelled offline starter annotations ...'
  python -m revenue_risk.cli extract --provider rules
fi
exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
