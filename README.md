# Telecom RAG

A Retrieval-Augmented Generation (RAG) system for telecom technical support and customer care, powered by Google Gemini and Chroma vector stores.

## Project Structure

```text
telecom_rag/
├── pyproject.toml              # Project dependencies and script entrypoints
├── .env                        # API keys and environment overrides
├── data/                       # Raw source data
│   ├── faq.csv                 # General policies and FAQs
│   ├── tickets.db              # Resolved SQLite customer support tickets
│   └── telecom_guide.pdf       # User guides and technical manuals
├── chroma_db/                  # Persistent Chroma vector databases
└── src/
    └── telecom_rag/
        ├── __init__.py         # Package entrypoint & CLI runner
        ├── rag_chain.py        # RAG pipeline orchestration
        ├── config/             # Centralized configuration
        │   ├── __init__.py
        │   └── config.py       # Paths, models, chunking parameters
        ├── retriever/          # Multi-source retriever
        │   ├── __init__.py
        │   └── retriever.py    # Merged retriever across collections
        └── ingest/             # Data ingestion scripts
            ├── __init__.py
            ├── ingest_faq.py
            ├── ingest_tickets.py
            └── ingest_pdf.py
```

## Quick Start

### 1. Environment Setup

Create a `.env` file in the root directory (see `.env.example`):
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Optional model overrides:
```bash
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-3.6-flash
```

### 2. Ingest Data

Populate the Chroma vector stores from the raw data files:
```bash
uv run python -m telecom_rag.ingest.ingest_faq
uv run python -m telecom_rag.ingest.ingest_tickets
uv run python -m telecom_rag.ingest.ingest_pdf
```

### 3. Run Queries

Query the RAG system directly from the CLI:
```bash
uv run --frozen telecom-rag "How do I activate roaming?"
```

Or from Python:
```python
from telecom_rag.rag_chain import build_chain

chain = build_chain()
answer = chain.invoke("How do I fix no service on my phone?")
print(answer)
```

