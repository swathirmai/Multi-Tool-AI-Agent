"""Central configuration loaded from .env"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    anthropic_api_key: str
    model_name: str
    embedding_model: str
    openweather_api_key: str
    faiss_index_path: str
    docs_dir: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int


def _load_settings() -> Settings:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return Settings(
        anthropic_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "claude-sonnet-4-6"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/vectorstore/faiss_index"),
        docs_dir=os.getenv("DOCS_DIR", "data/sample_docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
    )


settings: Settings = _load_settings()
