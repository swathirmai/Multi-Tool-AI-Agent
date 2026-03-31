"""Load and chunk documents from a directory."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir: str) -> list[Document]:
    """
    Recursively load all .txt files from docs_dir.

    Each document's metadata includes the 'source' file path.

    Args:
        docs_dir: Path to the directory containing source documents.

    Returns:
        List of Document objects.

    Raises:
        FileNotFoundError: If docs_dir does not exist.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No .txt documents found in: {docs_dir}")

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which splits on paragraphs,
    sentences, and words in order, preserving semantic coherence.

    Args:
        documents: Source Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
