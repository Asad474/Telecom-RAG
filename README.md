# Telecom RAG

A Retrieval-Augmented Generation (RAG) system for telecom technical support and customer care, powered by Google Gemini and Chroma vector stores.

The system answers customer questions by retrieving context from three sources — FAQs, resolved support tickets, and PDF guides — and feeding it to Gemini for a grounded answer.

## Features

- **Multi-source retrieval** — merges results from three independent Chroma collections (`faq`, `tickets`, `telecom_pdf`) into one context block.
- **Grounded answers** — the LLM answers only from retrieved context and defers to human support (`611`) when uncertain.
- **Interactive chat UI** — a Streamlit app with sample questions and conversation history.
- **CLI and Python API** — query the pipeline programmatically.
- **Reproducible environment** — managed with `uv` and locked via `uv.lock`.

## Project Structure

```text
telecom_rag/
├── pyproject.toml              # Dependencies and [project.scripts] entrypoints
├── uv.lock                     # Locked dependency versions
├── .python-version             # Pinned Python version (3.12)
├── .env.example                # Template for environment variables
├── README.md
├── data/                       # Raw source data
│   ├── faq.csv                 # General policies and FAQs
│   ├── tickets.db              # Resolved SQLite customer support tickets
│   └── telecom_guide.pdf       # User guides and technical manuals
├── chroma_db/                  # Persistent Chroma vector stores (created by ingestion)
└── src/
    └── telecom_rag/
        ├── __init__.py         # Package entrypoint & CLI runner (`main`)
        ├── app.py              # Streamlit chat UI
        ├── cli.py              # Console script wrapper for Streamlit (`chat`)
        ├── rag_chain.py        # RAG pipeline orchestration (`build_chain`)
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

## Requirements

- **Python 3.12+** — see `.python-version`. (3.14 is intentionally avoided: PyTorch, a dependency of `sentence-transformers`, does not yet publish 3.14 wheels.)
- **[uv](https://docs.astral.sh/uv/)** — dependency management and script runner.
- **A Google Gemini API key** — from [Google AI Studio](https://aistudio.google.com/apikey).

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

This creates `.venv/`, installs all dependencies, installs the `telecom_rag` package itself (editable), and registers the console scripts from `pyproject.toml`.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional — raises HuggingFace rate limits; not needed for public models
HF_TOKEN=your_huggingface_token_here

# Optional model overrides
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.0-flash
```

`HF_TOKEN` is **optional** — the default embedding model (`all-MiniLM-L6-v2`) is public. Without a token you may see a warning about unauthenticated HuggingFace Hub requests; embedding still works.

`LLM_MODEL` must be a real Gemini model. Valid examples: `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`. There is **no** `gemini-3.6-flash`.

### 3. Ingest data

Populate the Chroma vector stores from the raw data files. Run **once**, or whenever source files change:

```bash
uv run ingest-faq
uv run ingest-tickets
uv run ingest-guides
```

Equivalent module form (no scripts required):

```bash
uv run python -m telecom_rag.ingest.ingest_faq
uv run python -m telecom_rag.ingest.ingest_tickets
uv run python -m telecom_rag.ingest.ingest_pdf
```

Each script writes to `chroma_db/` under its own collection. Re-running overwrites that collection — safe to repeat.

**Verify ingestion:**

```bash
uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
for name in ('faq', 'tickets', 'telecom_pdf'):
    try:
        print(f'{name}: {client.get_collection(name).count()} records')
    except Exception as e:
        print(f'{name}: not found ({e})')
"
```

Expected output (counts vary with data):

```
faq: 52 records
tickets: 40 records
telecom_pdf: 120 records
```

### 4. Run the chat UI

```bash
uv run telecom-chat
```

Equivalent direct invocation:

```bash
uv run streamlit run src/telecom_rag/app.py
```

Open <http://localhost:8501>.

### 5. Query from the CLI or Python

**CLI:**

```bash
uv run telecom-rag "How do I activate roaming?"
```

**Python:**

```python
from telecom_rag.rag_chain import build_chain

chain = build_chain()
print(chain.invoke("How do I fix no service on my phone?"))
```

Build the chain **once** and reuse it — each `build_chain()` call reloads the embedding model and reopens the Chroma stores, which is slow (5–15 s per call).

## How It Works

```text
question
   │
   ├──► faq     retriever (top-k)  ─┐
   ├──► tickets retriever (top-k)  ─┼──► merged docs ──► formatted context ─┐
   └──► guides  retriever (top-k)  ─┘                                       │
                                                                           ▼
question ─────────────────────────────────────────────► prompt ──► Gemini ──► answer
```

1. **Ingestion** — each source is loaded, optionally chunked (PDFs only; FAQs and tickets are 1 row = 1 doc), embedded with `sentence-transformers/all-MiniLM-L6-v2`, and stored in Chroma under its own collection.
2. **Retrieval** — `build_retriever()` queries all three collections independently and concatenates the results. There is no cross-source re-ranking; order is `faq → tickets → guides`.
3. **Generation** — merged docs are formatted with source tags (`[FAQ]`, `[TICKETS]`, `[GUIDES]`), inserted into the system prompt, and sent to Gemini with `temperature=0`.
4. **Grounded output** — the system prompt instructs the model to answer only from context and suggest calling `611` when it cannot.