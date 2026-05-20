"""Text chunking utilities for retrieval."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.pdf_loader import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for document in documents:
        split_texts = splitter.split_text(document.text)
        for index, text in enumerate(split_texts):
            metadata = {**document.metadata, "chunk": index + 1}
            chunks.append(Document(text=text, metadata=metadata))
    return chunks
