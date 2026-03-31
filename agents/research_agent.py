"""Specialised research sub-agent: document search + web search."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import settings
from tools.document_retrieval import get_document_retrieval_tool
from tools.web_search import get_web_search_tool

RESEARCH_SYSTEM_PROMPT = """You are a specialised research agent with access to:
1. document_search — searches the internal knowledge base (AI, climate change, Python docs)
2. web_search — searches the internet for current or general information

Research strategy:
- ALWAYS try document_search first for questions about AI, machine learning, climate, or Python.
- Fall back to web_search if the internal knowledge base doesn't have sufficient information.
- Cite your sources: include document names or URLs in your response.
- Provide a structured, accurate summary — not raw search results.
- Do NOT perform math calculations or weather lookups; those belong to other tools."""


def build_research_agent(retriever: BaseRetriever):
    """
    Build the research sub-agent with document search and web search tools.

    Args:
        retriever: A configured FAISS retriever for document search.

    Returns:
        A compiled LangGraph ReAct agent (CompiledStateGraph).
    """
    llm = ChatAnthropic(
        model=settings.model_name,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
    )

    tools = [
        get_document_retrieval_tool(retriever),
        get_web_search_tool(max_results=5),
    ]

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return agent
