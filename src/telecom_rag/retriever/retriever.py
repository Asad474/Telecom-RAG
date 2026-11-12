"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from telecom_rag.config import settings


def build_retriever(
    k_faq: int = settings.K_FAQ,
    k_tickets: int = settings.K_TICKETS,
    k_guides: int = settings.K_GUIDES,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    faq_store = Chroma(
        collection_name=settings.COLLECTION_FAQ,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
    )

    ticket_store = Chroma(
        collection_name=settings.COLLECTION_TICKETS,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
    )

    guide_store = Chroma(
        collection_name=settings.COLLECTION_GUIDES,
        embedding_function=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
    )

    faq_retriever = faq_store.as_retriever(search_kwargs={"k": k_faq})
    ticket_retriever = ticket_store.as_retriever(search_kwargs={"k": k_tickets})
    guide_retriever = guide_store.as_retriever(search_kwargs={"k": k_guides})

    def retriever(query: str) -> list[Document]:
        return (
            faq_retriever.invoke(query)
            + ticket_retriever.invoke(query)
            + guide_retriever.invoke(query)
        )

    return RunnableLambda(retriever)

