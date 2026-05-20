"""Unit tests for PDF loading behavior."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from src.pdf_loader import extract_text_from_pdf, load_pdfs


def test_extract_text_from_blank_pdf_returns_no_documents() -> None:
    """A blank PDF should not produce page documents."""
    pdf_stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(pdf_stream)
    pdf_stream.seek(0)
    pdf_stream.name = "blank.pdf"

    documents = extract_text_from_pdf(pdf_stream)

    assert documents == []


def test_load_pdfs_delegates_multiple_files() -> None:
    """Multiple readable PDFs should be handled without error."""
    streams = []
    for index in range(2):
        pdf_stream = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf_stream)
        pdf_stream.seek(0)
        pdf_stream.name = f"blank-{index}.pdf"
        streams.append(pdf_stream)

    documents = load_pdfs(streams)

    assert documents == []


def test_extract_text_from_invalid_pdf_raises() -> None:
    """Invalid PDF bytes should surface an exception to callers."""
    invalid_pdf = BytesIO(b"not a real pdf")
    invalid_pdf.name = "invalid.pdf"

    with pytest.raises(Exception):
        extract_text_from_pdf(invalid_pdf)
