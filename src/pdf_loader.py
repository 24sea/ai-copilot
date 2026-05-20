"""PDF loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterable

from pypdf import PdfReader

from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class Document:
    """Text extracted from one page of a source document."""

    text: str
    metadata: dict[str, str | int]


def extract_text_from_pdf(file: BinaryIO, filename: str | None = None) -> list[Document]:
    """Extract page-level text from a PDF file-like object."""
    source_name = filename or getattr(file, "name", "uploaded_document.pdf")
    documents: list[Document] = []

    try:
        reader = PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned_text = " ".join(text.split())
            if cleaned_text:
                documents.append(
                    Document(
                        text=cleaned_text,
                        metadata={"source": source_name, "page": page_number},
                    ),
                )
        logger.info("Extracted %s pages from %s", len(documents), source_name)
        return documents
    except Exception:
        logger.exception("Failed to extract text from PDF: %s", source_name)
        raise


def load_pdfs(files: Iterable[BinaryIO]) -> list[Document]:
    """Extract text from multiple PDF file-like objects."""
    documents: list[Document] = []
    for file in files:
        filename = getattr(file, "name", None)
        documents.extend(extract_text_from_pdf(file, filename))
    return documents
