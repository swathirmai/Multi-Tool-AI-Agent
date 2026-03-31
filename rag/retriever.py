"""Retriever factory and LCEL RAG chain helpers."""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda


def get_retriever(store: FAISS, k: int = 4) -> BaseRetriever:
    """
    Return a similarity-search retriever from a FAISS store.

    Args:
        store: A built FAISS vector store.
        k: Number of top documents to retrieve per query.

    Returns:
        A BaseRetriever using cosine similarity search.
    """
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents into a readable string with source citations.

    Args:
        docs: List of retrieved Document objects.

    Returns:
        Formatted string with source metadata and content for each chunk.
    """
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_rag_chain(retriever: BaseRetriever):
    """
    Build a standalone LCEL RAG retrieval chain.

    Returns a Runnable[str, str] using the pipe operator:
        query → retriever → format_docs → formatted context string

    Usage:
        chain = build_rag_chain(retriever)
        context = chain.invoke("What is machine learning?")

    Args:
        retriever: A configured BaseRetriever.

    Returns:
        A LangChain Runnable pipeline.
    """
    return retriever | RunnableLambda(format_docs)
