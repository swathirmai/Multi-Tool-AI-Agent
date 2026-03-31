"""FAISS vector store: build, persist, and load."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss_store(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """
    Build a new FAISS index from document chunks.

    Args:
        documents: Chunked Document objects to embed.
        embeddings: Embeddings instance to use for vectorization.

    Returns:
        An in-memory FAISS vector store.
    """
    return FAISS.from_documents(documents, embeddings)


def save_faiss_store(store: FAISS, index_path: str) -> None:
    """
    Persist a FAISS index to disk.

    Creates parent directories if they don't exist.

    Args:
        store: The FAISS store to save.
        index_path: File path prefix (FAISS saves two files: .faiss and .pkl).
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    store.save_local(index_path)


def load_faiss_store(index_path: str, embeddings: Embeddings) -> FAISS:
    """
    Load a persisted FAISS index from disk.

    Args:
        index_path: Path prefix used when saving the store.
        embeddings: Must be the same embeddings model used during build.

    Returns:
        The loaded FAISS store.
    """
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_build_store(
    embeddings: Embeddings,
    docs_dir: str,
    index_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    """
    Load the FAISS index from disk if it exists; otherwise build and save it.

    This is the primary entry point used at agent startup.

    Args:
        embeddings: Embeddings instance.
        docs_dir: Directory containing source .txt documents.
        index_path: Disk path for the persisted FAISS index.
        chunk_size: Chunk size for splitting (only used when building).
        chunk_overlap: Chunk overlap (only used when building).

    Returns:
        A ready-to-use FAISS store.
    """
    # Check if a saved index already exists (FAISS saves index.faiss + index.pkl)
    faiss_file = Path(index_path).parent / (Path(index_path).name + ".faiss")
    if faiss_file.exists():
        return load_faiss_store(index_path, embeddings)

    # Build from scratch
    from rag.loader import load_documents, split_documents

    print(f"Building FAISS index from documents in '{docs_dir}'...")
    documents = load_documents(docs_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"  Loaded {len(documents)} documents → {len(chunks)} chunks")

    store = build_faiss_store(chunks, embeddings)
    save_faiss_store(store, index_path)
    print(f"  Index saved to '{index_path}'")
    return store
