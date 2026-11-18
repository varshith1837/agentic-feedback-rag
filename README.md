# Agentic RAG Feedback Analytics Platform

> An autonomous RAG agent that analyzes 99K+ e-commerce customer reviews, extracts structured complaint signals via LLM, builds a feedback knowledge graph, and answers business questions with **anti-hallucination verification** — every number is SQL-proven, every citation is source-checked.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-GPT--OSS--120B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What This Project Does

This platform processes the **Olist Brazilian E-Commerce dataset** (99,441 orders, 42,687 written reviews) through a full AI-powered analytics pipeline:

1. **Structured LLM Extraction** — Groq-hosted GPT-OSS 120B extracts complaint categories, severity, sentiment, and verbatim Portuguese evidence from raw customer reviews using strict Pydantic schema validation.

2. **Autonomous RAG Agent** — A bounded multi-tool agent that translates natural-language business questions into parameterized SQL queries. The agent selects from 6 fixed tools but **cannot write arbitrary SQL** — eliminating injection and hallucination risks.

3. **Anti-Hallucination Verification Engine** — Every financial number is mathematically validated against deterministic SQL aggregations. Every citation is verified against source reviews within the active filter scope. Unverified claims are dropped, not displayed.

4. **Feedback Knowledge Graph** — NetworkX-powered graph connecting reviews → orders → sellers → product categories → customer states, enabling relationship exploration and pattern discovery.

5. **Interactive Streamlit Dashboard** — Revenue exposure charts, graph visualization, cited Q&A chat interface with expandable provenance (SQL results + original Portuguese reviews).

---

## 🏗️ Architecture

```
Customer Reviews (Portuguese)
        │
        ▼
   ┌─────────────┐     ┌──────────────────────┐
   │  prep.py    │────▶│  SQLite / DuckDB     │
   │  (Ingest)   │     │  orders, items,      │
   └─────────────┘     │  reviews, sample     │
        │              └──────┬───────────────┘
        ▼                     │
   ┌─────────────┐            │
   │ extract.py  │◄───────────┘
   │ Groq API    │
   │ (GPT-OSS    │──── Pydantic Validation
   │  120B)      │──── Verbatim Quote Check
   └─────────────┘──── SHA-256 Request Cache
        │
        ▼
   extractions + mentions + llm_cache
        │
        ▼
   ┌─────────────────────────────────────┐
   │         Autonomous RAG Agent        │
   │  ┌───────────┐  ┌───────────────┐   │
   │  │ Bounded   │  │ 6 Fixed SQL   │   │
   │  │ Planner   │─▶│ Tools (no     │   │
   │  │ (≤4 calls)│  │ arbitrary SQL)│   │
   │  └───────────┘  └───────┬───────┘   │
   │                         │           │
   │  ┌──────────────────────▼────────┐  │
   │  │  Anti-Hallucination Verifier  │  │
   │  │  • Numeric SQL cross-check    │  │
   │  │  • Scoped citation check      │  │
   │  │  • Optional NLI paraphrase    │  │
   │  └───────────────────────────────┘  │
   └─────────────────┬───────────────────┘
                     ▼
            Streamlit Dashboard
            (Charts + Graph + Chat)
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **LLM Provider** | Groq API (GPT-OSS 120B / configurable) |
| **Data Validation** | Pydantic v2 (strict schemas, `extra='forbid'`) |
| **Database** | SQLite (default) / DuckDB (optional) |
| **Agent Framework** | Custom bounded planner (no LangChain dependency) |
| **Semantic Search** | SentenceTransformers (multilingual-e5-small) — optional |
| **NLI Verification** | DeBERTa-v3 cross-encoder — optional |
| **Graph** | NetworkX |
| **Frontend** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest (13 unit/integration tests) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — download `brazil.zip` and place it in `data/`

### Installation

```bash
# Clone the repository
git clone https://github.com/varshith1837/agentic-feedback-rag.git
cd agentic-feedback-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Prepare the Database

```bash
# Ingest the Olist dataset and create stratified sample
python -m revenue_risk.cli prepare --zip data/brazil.zip --sample-size 4000

# Extract complaint signals using LLM
python -m revenue_risk.cli extract --provider groq --batch-size 4 --pause 4

# Check extraction status
python -m revenue_risk.cli status
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```
Open **http://localhost:8501** in your browser.

### Ask Questions via CLI

```bash
python -m revenue_risk.cli ask "Which product categories are most exposed to delivery delays?" --provider groq
python -m revenue_risk.cli ask "Which states have the highest late-delivery revenue at risk?" --provider groq
python -m revenue_risk.cli ask "Is damaged item risk getting worse over time?" --provider groq
```

### Docker

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
docker compose up --build
```

---

## 🛠️ Agent Tools

The agent can select from **6 fixed, parameterized SQL tools** (no arbitrary SQL execution):

| Tool | Description |
|---|---|
| `list_aspects` | Returns the 11 fixed issue categories with mention counts |
| `revenue_at_risk` | Revenue exposure by aspect, category, seller, state, or month |
| `trend` | Monthly risk exposure over time |
| `compare` | Side-by-side comparison of two date ranges (delta in BRL) |
| `top_examples` | Source Portuguese reviews within filtered scope |
| `semantic_search` | Multilingual dense retrieval (optional, requires extra install) |

### 11 Issue Categories

`delivery_delay` · `product_not_received` · `damaged_item` · `wrong_or_missing_item` · `product_quality` · `not_as_described` · `seller_service` · `refund_or_return` · `packaging` · `price_value` · `other`

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest -q

# Run the 30-question evaluation suite
python -m revenue_risk.cli evaluate --provider rules

# Test out-of-scope refusal
python -m revenue_risk.cli ask "What do customers think about the mobile app?"
```

---

## 📁 Project Structure

```
agentic-feedback-rag/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard (charts, graph, chat)
├── revenue_risk/
│   ├── config.py                 # Paths, issue taxonomy, safety limits
│   ├── db.py                     # SQLite / DuckDB adapter
│   ├── prep.py                   # Olist ZIP ingestion, order joins, sampling
│   ├── schemas.py                # Pydantic contracts (extraction, plans, filters)
│   ├── llm.py                    # Groq/Ollama JSON client, retry, rate-limit backoff, cache
│   ├── extract.py                # Resumable LLM extraction with quote validation
│   ├── tools.py                  # 6 parameterized SQL tools (allowlisted)
│   ├── graph.py                  # NetworkX neighborhood renderer
│   ├── agent.py                  # Bounded planner + grounded answer rendering
│   ├── verify.py                 # Numeric, citation, and optional NLI verification
│   ├── evaluation.py             # Independent gold calculations + metrics
│   └── cli.py                    # CLI: prepare, extract, ask, evaluate, status
├── tests/                        # pytest unit & integration tests
├── eval/
│   ├── questions.json            # 30 frozen evaluation prompts
│   └── hand_label_60.csv         # Manual labeling template (60 reviews)
├── scripts/                      # Boot scripts (Docker, local, Windows)
├── docs/                         # Assumptions & limitations documentation
├── Dockerfile                    # Container image
├── compose.yaml                  # Docker Compose configuration
├── requirements.txt
├── pyproject.toml
└── .env.example                  # Environment template (never commit .env)
```

---

## ⚙️ Configuration

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | *(required for LLM)* |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-120b` |
| `OLLAMA_MODEL` | Local Ollama model | `qwen2.5:7b` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `RISK_DB` | Database path (`.sqlite` or `.duckdb`) | `runtime/revenue.sqlite` |

---

## 🔒 Safety & Reliability

- **No arbitrary SQL execution** — The LLM selects tools and parameters; SQL is pre-written and parameterized
- **Pydantic strict mode** — All LLM outputs validated with `extra='forbid'`, type checks, and range constraints
- **Verbatim quote validation** — Extracted Portuguese evidence must exist in the source review text
- **SHA-256 request caching** — Identical LLM requests are served from cache (no duplicate API calls)
- **Automatic rate-limit backoff** — Reads Groq's `retry-after` headers and pauses intelligently
- **Resumable extraction** — Pipeline crashes never lose progress; rerun to continue from where it stopped
- **Out-of-scope refusal** — Questions about unsupported attributes (mobile app, churn, future revenue) are refused

---

## 🗺️ Roadmap

- [ ] FastAPI backend for production deployment
- [ ] Vector database integration (Pinecone / Qdrant) for scalable semantic search
- [ ] LangGraph-based multi-step agent workflows
- [ ] Real-time streaming extraction pipeline
- [ ] Authentication & multi-tenant support
- [ ] Cloud deployment (GCP / AWS)

---

## 📊 Dataset

This project uses the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0). Download `brazil.zip` from Kaggle and place it in `data/` before running `prepare`.

---

## 📄 License

This project is for educational and portfolio purposes. The Olist dataset has its own [license terms](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
