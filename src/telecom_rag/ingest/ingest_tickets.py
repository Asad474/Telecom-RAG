"""
Ingests data/tickets.db into 'tickets' chroma collection.
Run once (or whenever db changes): python -m telecom_rag.ingest.ingest_tickets
"""

import os
import sqlite3
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from telecom_rag.config import settings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_ticket_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * from tickets where status = 'resolved'").fetchall()
    conn.close()

    docs = []
    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}\n"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "ticket",
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "status": row["status"],
                },
            )
        )

    return docs


def main():
    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(str(settings.TICKETS_DB_PATH))
    print(f"  {len(docs)} resolved tickets loaded.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{settings.COLLECTION_TICKETS}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=settings.COLLECTION_TICKETS,
        persist_directory=str(settings.CHROMA_DIR),
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()

