"""Wrap a FAISS retriever as a LangChain tool for agent use."""
from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, create_retriever_tool


def get_document_retrieval_tool(retriever: BaseRetriever) -> BaseTool:
    """
    Wrap a retriever as a named tool the agent can call.

    The tool takes a query string and returns formatted document chunks
    with source citations from the FAISS vector store.

    Args:
        retriever: A configured BaseRetriever (e.g., FAISS similarity search).

    Returns:
        A BaseTool named 'document_search'.
    """
    return create_retriever_tool(
        retriever,
        name="document_search",
        description=(
            "Search the internal knowledge base for information about AI, "
            "machine learning, climate change, Python programming, and related topics. "
            "Use this FIRST before web_search for questions likely covered by "
            "our internal documents. Returns relevant excerpts with source citations."
        ),
    )
