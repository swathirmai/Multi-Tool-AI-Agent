"""
Multi-Tool AI Agent — Interactive CLI

Usage:
    python main.py                  # Start interactive chat (streaming)
    python main.py --no-stream      # Start interactive chat (non-streaming)
    python main.py --build-index    # Pre-build the FAISS vector index
    python main.py --query "..."    # Single query mode (non-interactive)

Setup:
    1. cp .env.example .env && edit .env  (add ANTHROPIC_API_KEY)
    2. pip install -r requirements.txt
    3. python main.py
"""
from __future__ import annotations

import argparse
import sys
import uuid

from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from rag.vectorstore import get_or_build_store
from rag.retriever import get_retriever
from agents.orchestrator import build_orchestrator


# ─────────────────────────────────────────
# Startup helpers
# ─────────────────────────────────────────

def init_pipeline():
    """Load embeddings, build/load FAISS store, and return the orchestrator."""
    print("Loading embeddings model (first run downloads ~90 MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = get_or_build_store(
        embeddings=embeddings,
        docs_dir=settings.docs_dir,
        index_path=settings.faiss_index_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    retriever = get_retriever(store, k=settings.retriever_k)
    orchestrator, _ = build_orchestrator(retriever)
    return orchestrator


def build_index_only():
    """Pre-build and persist the FAISS index, then exit."""
    from rag.loader import load_documents, split_documents
    from rag.vectorstore import build_faiss_store, save_faiss_store

    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
    )

    print(f"Loading documents from '{settings.docs_dir}'...")
    docs = load_documents(settings.docs_dir)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"  {len(docs)} documents → {len(chunks)} chunks")

    print("Building FAISS index...")
    store = build_faiss_store(chunks, embeddings)
    save_faiss_store(store, settings.faiss_index_path)
    print(f"Index saved to '{settings.faiss_index_path}'. Done.")


# ─────────────────────────────────────────
# Agent invocation
# ─────────────────────────────────────────

def run_query(orchestrator, query: str, thread_id: str, stream: bool = True) -> str:
    """
    Send a single query to the orchestrator and return the final response.

    Args:
        orchestrator: Compiled LangGraph agent.
        query: User's natural language input.
        thread_id: Conversation thread ID for memory persistence.
        stream: If True, stream tokens to stdout as they arrive.

    Returns:
        The agent's final response as a string.
    """
    inputs = {"messages": [{"role": "user", "content": query}]}
    config = {"configurable": {"thread_id": thread_id}}

    if stream:
        response_parts: list[str] = []
        print("Agent: ", end="", flush=True)

        for chunk in orchestrator.stream(inputs, config=config, stream_mode="updates"):
            for _node_name, node_output in chunk.items():
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        # Handle both string content and list-of-blocks content
                        if isinstance(msg.content, str):
                            print(msg.content, end="", flush=True)
                            response_parts.append(msg.content)
                        elif isinstance(msg.content, list):
                            for block in msg.content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block["text"]
                                    print(text, end="", flush=True)
                                    response_parts.append(text)
        print()  # newline after streamed response
        return "".join(response_parts)

    else:
        result = orchestrator.invoke(inputs, config=config)
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        b["text"] for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                return str(content)
        return "(No response received)"


# ─────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────

def interactive_chat(orchestrator, stream: bool = True):
    """Run an interactive REPL session with the orchestrator agent."""
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 60)
    print("  Multi-Tool AI Agent")
    print("=" * 60)
    print("Capabilities: document search (RAG), web search, calculator, weather")
    print(f"Session ID: {session_id}")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", "bye"):
            print("Goodbye!")
            break

        try:
            run_query(orchestrator, user_input, thread_id=session_id, stream=stream)
            print()
        except Exception as exc:
            print(f"[Error] {exc}\n")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Tool AI Agent powered by LangChain + Anthropic Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Pre-build the FAISS vector index and exit.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming (print full response at once).",
    )
    args = parser.parse_args()

    if args.build_index:
        build_index_only()
        sys.exit(0)

    orchestrator = init_pipeline()

    if args.query:
        thread_id = f"cli-{uuid.uuid4().hex[:8]}"
        response = run_query(
            orchestrator,
            args.query,
            thread_id=thread_id,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(f"Agent: {response}")
    else:
        interactive_chat(orchestrator, stream=not args.no_stream)


if __name__ == "__main__":
    main()
