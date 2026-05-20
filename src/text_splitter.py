"""Text chunking utilities for retrieval."""

from __future__ import annotations

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.pdf_loader import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[Document] = []
    for document in documents:
        split_texts = _split_text(document.text, chunk_size, chunk_overlap)
        for index, text in enumerate(split_texts):
            metadata = {**document.metadata, "chunk": index + 1}
            chunks.append(Document(text=text, metadata=metadata))
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks without loading heavy ML packages."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        if end < len(cleaned_text):
            split_at = _best_split_position(cleaned_text, start, end)
            if split_at > start:
                end = split_at

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def _best_split_position(text: str, start: int, end: int) -> int:
    """Find a natural split point near the end of a chunk."""
    separators = ["\n\n", "\n", ". ", " "]
    search_start = start + max((end - start) // 2, 1)
    for separator in separators:
        position = text.rfind(separator, search_start, end)
        if position != -1:
            return position + len(separator)
    return end
