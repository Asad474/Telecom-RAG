"""
Telecom RAG package.
"""

import sys
from telecom_rag.config import settings
from telecom_rag.retriever import build_retriever
from telecom_rag.rag_chain import build_chain


def main() -> None:
    # If user provided a query via command line arguments:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}\n")
        chain = build_chain()
        response = chain.invoke(query)
        print(f"Response:\n{response}")
        return

    print("=" * 50)
    print("Telecom RAG System")
    print("=" * 50)
    print(f"Project Root    : {settings.PROJECT_ROOT}")
    print(f"Embedding Model : {settings.EMBED_MODEL}")
    print(f"LLM Model       : {settings.LLM_MODEL}")
    print(f"Chroma DB Path  : {settings.CHROMA_DIR}")
    print("=" * 50)
    print("\nUsage:")
    print("  telecom-rag \"<your question here>\"")
    print("  uv run --frozen telecom-rag \"How do I activate roaming?\"")


__all__ = ["settings", "build_retriever", "build_chain", "main"]
