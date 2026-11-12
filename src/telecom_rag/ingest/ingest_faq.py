"""
Ingests data/faq.csv into 'faq' chroma collection.
Run once (or whenever csv changes): python -m telecom_rag.ingest.ingest_faq
"""

import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from telecom_rag.config import settings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "faq",
                    "category": row["category"],
                    "faq_id": str(row["id"]),
                },
            )
        )

    return docs


def main():
    print("Loading FAQ Documents...")
    docs = load_faq_documents(str(settings.FAQ_CSV_PATH))
    print(f"{len(docs)} FAQ entries loaded.")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{settings.COLLECTION_FAQ}'")
    vectorStore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=settings.COLLECTION_FAQ,
        persist_directory=str(settings.CHROMA_DIR),
    )

    print(f"Done. {vectorStore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()

