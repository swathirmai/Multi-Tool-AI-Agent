# Multi-Tool AI Agent

A multi-agent LLM system with tool use and retrieval-augmented reasoning (RAG) for complex task execution, built with **LangChain** + **Anthropic Claude** + **LangGraph**.

## What It Does

The agent can:
- **Search documents (RAG)** — FAISS vector store with HuggingFace embeddings
- **Search the web** — DuckDuckGo (no API key required)
- **Calculate** — Safe AST-based math evaluator
- **Check weather** — OpenWeatherMap API (optional)
- **Multi-agent routing** — Orchestrator delegates to a specialised research sub-agent

## Architecture

```
User (CLI)
   │
   ▼
Orchestrator Agent  (Claude + LangGraph ReAct)
   ├── calculator          ← safe AST eval, no code execution
   ├── get_weather         ← OpenWeatherMap (graceful fallback if no key)
   └── research_agent  ─── Research Sub-Agent  (Claude + LangGraph ReAct)
                               ├── document_search  ← FAISS + HuggingFace embeddings
                               └── web_search       ← DuckDuckGo

Vector Store (FAISS)
  └── HuggingFaceEmbeddings: all-MiniLM-L6-v2 (local, CPU, no API key)
  └── Documents: data/sample_docs/*.txt
```

## Tech Stack

| Component | Library |
|-----------|---------|
| Agent framework | LangChain + LangGraph (`create_react_agent`) |
| LLM | Anthropic Claude via `langchain-anthropic` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector store | FAISS (`langchain-community`) |
| Web search | DuckDuckGo (`langchain-community`) |
| Weather API | OpenWeatherMap (`requests`) |
| Calculator | Python `ast` module (safe, no `eval`) |
| Memory | LangGraph `MemorySaver` (per-session thread) |

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Optionally add an OpenWeatherMap key for weather queries:
```
OPENWEATHER_API_KEY=your_key_here
```

### 3. Run

```bash
# Interactive chat (with streaming)
python main.py

# Single query
python main.py --query "What is RAG in AI?"

# Pre-build vector index (optional — auto-builds on first run)
python main.py --build-index

# Disable streaming
python main.py --no-stream
```

## Example Queries

| Query | Tools Used |
|-------|-----------|
| `What is machine learning?` | `document_search` |
| `What are the latest AI trends?` | `web_search` |
| `What is 17% of 348?` | `calculator` |
| `What's the weather in Tokyo?` | `get_weather` |
| `Explain climate change and calculate the CO2 ppm increase since 1850` | `research_agent` + `calculator` |

## Project Structure

```
Multi-Tool AI Agent/
├── main.py                     # CLI entry point
├── config.py                   # Settings loaded from .env
├── requirements.txt
├── .env.example                # Environment variable template
│
├── tools/
│   ├── calculator.py           # AST-based safe math evaluator
│   ├── weather.py              # OpenWeatherMap API tool
│   ├── web_search.py           # DuckDuckGo search tool
│   └── document_retrieval.py  # FAISS retriever tool wrapper
│
├── rag/
│   ├── loader.py               # Document loading + chunking
│   ├── vectorstore.py          # FAISS build/save/load
│   └── retriever.py            # Retriever factory + LCEL RAG chain
│
├── agents/
│   ├── research_agent.py       # Specialised research sub-agent
│   └── orchestrator.py         # Main orchestrator agent
│
└── data/
    └── sample_docs/            # Knowledge base documents
        ├── ai_overview.txt
        ├── climate_change.txt
        └── python_guide.txt
```

## Adding Your Own Documents

Drop `.txt` files into `data/sample_docs/` then rebuild the index:

```bash
python main.py --build-index
```

The agent will automatically use your new documents on the next run.

## Key Design Decisions

- **No raw `eval()`** — calculator uses Python's `ast` module with an explicit whitelist of safe operations
- **Graceful degradation** — weather tool works without an API key (returns a helpful message)
- **Research isolation** — each orchestrator call to the research agent uses a fresh thread ID to prevent memory bleed-over
- **LCEL pipe chains** — RAG retrieval uses `retriever | RunnableLambda(format_docs)` for clean composition
- **Persistent memory** — `MemorySaver` + `thread_id` gives each session its own conversation history
