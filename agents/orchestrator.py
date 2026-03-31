"""Main orchestrator agent: routes tasks across all specialized tools and sub-agents."""
from __future__ import annotations

import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import settings
from tools.calculator import calculator
from tools.weather import get_weather
from agents.research_agent import build_research_agent

ORCHESTRATOR_SYSTEM_PROMPT = """You are a powerful multi-tool AI assistant with access to:

1. calculator       — evaluate any math expression safely
2. get_weather      — get current weather for any city
3. research_agent   — delegate research tasks (document search + web search)

Decision rules:
- For ANY arithmetic or calculation → use calculator (never compute in your head)
- For weather queries → use get_weather
- For questions requiring facts, research, document lookup, or web search → use research_agent
- For simple conversational exchanges → respond directly without tools

Always synthesize tool results into a clear, helpful final answer.
If a task requires multiple tools, call them sequentially and combine the results."""


def _wrap_research_agent_as_tool(agent) -> StructuredTool:
    """
    Wrap the research agent graph as a StructuredTool callable by the orchestrator.

    Each call gets a fresh thread_id to keep research sessions isolated.

    Args:
        agent: A compiled LangGraph ReAct agent (the research agent).

    Returns:
        A StructuredTool named 'research_agent'.
    """
    def _run(query: str) -> str:
        thread_id = f"research-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
        messages = result.get("messages", [])
        # Return the last AI message that isn't a tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content
        return "Research agent returned no response."

    return StructuredTool.from_function(
        func=_run,
        name="research_agent",
        description=(
            "Delegate a research or information-retrieval task to the specialised "
            "research agent. It searches both internal documents and the web. "
            "Use this for any question requiring factual lookup, research synthesis, "
            "or information about AI, climate change, Python, or current events. "
            "Input: a clear, detailed research query string."
        ),
    )


def build_orchestrator(retriever: BaseRetriever):
    """
    Build the main orchestrator agent.

    Constructs the research sub-agent, wraps it as a tool, then creates the
    orchestrator with: calculator + get_weather + research_agent.

    Args:
        retriever: A configured FAISS retriever passed to the research agent.

    Returns:
        tuple: (orchestrator_graph, memory_saver)
            - orchestrator_graph: compiled LangGraph ReAct agent
            - memory_saver: MemorySaver for the orchestrator's conversation memory
    """
    llm = ChatAnthropic(
        model=settings.model_name,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
    )

    research_agent = build_research_agent(retriever)
    research_tool = _wrap_research_agent_as_tool(research_agent)

    tools = [
        calculator,
        get_weather,
        research_tool,
    ]

    memory = MemorySaver()

    orchestrator = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return orchestrator, memory
