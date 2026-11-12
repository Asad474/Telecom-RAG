"""
Ingests data/telecom_guide.pdf into 'telecom_pdf' chroma collection.
Run once (or whenever guide changes): python -m telecom_rag.ingest.ingest_pdf
"""

import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from telecom_rag.config import settings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def main():
    print("Loading PDF...")
    loader = PyPDFLoader(str(settings.PDF_PATH))
    pages = loader.load()
    print(f"{len(pages)} pages loaded.")

    print(f"Chunking (size={settings.CHUNK_SIZE}), overlap={settings.CHUNK_OVERLAP}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    # Tag each chunk so we know it came from the guide
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{settings.COLLECTION_GUIDES}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.COLLECTION_GUIDES,
        persist_directory=str(settings.CHROMA_DIR),
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()

