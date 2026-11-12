"""
Central configuration for Telecom RAG.
Reads environment variables with sensible defaults.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


PROJECT_ROOT = _find_project_root()

# Load environment variables from .env in project root
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Directory paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

    # Data files
    FAQ_CSV_PATH: Path = DATA_DIR / "faq.csv"
    TICKETS_DB_PATH: Path = DATA_DIR / "tickets.db"
    PDF_PATH: Path = DATA_DIR / "telecom_guide.pdf"

    # Chroma collections
    COLLECTION_FAQ: str = "faq"
    COLLECTION_TICKETS: str = "tickets"
    COLLECTION_GUIDES: str = "telecom_pdf"

    # Models
    EMBED_MODEL: str = os.getenv(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    # Ingestion / chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retriever Top-K
    K_FAQ: int = int(os.getenv("K_FAQ", "3"))
    K_TICKETS: int = int(os.getenv("K_TICKETS", "3"))
    K_GUIDES: int = int(os.getenv("K_GUIDES", "3"))


settings = Settings()

