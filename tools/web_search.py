"""DuckDuckGo web search tool — no API key required."""
from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool


def get_web_search_tool(max_results: int = 5) -> BaseTool:
    """
    Return a DuckDuckGoSearchRun tool configured for agent use.

    DuckDuckGo requires no API key. Results are plain text snippets.

    Args:
        max_results: Maximum number of search results to return.

    Returns:
        A LangChain BaseTool instance named 'web_search'.
    """
    return DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the internet for current events, facts, news, or any topic "
            "not covered by the local document knowledge base. Use this when you "
            "need real-time or recent information not available in internal documents."
        ),
    )
