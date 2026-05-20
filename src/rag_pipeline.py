"""RAG ingestion and question-answering pipeline."""

from __future__ import annotations

from typing import BinaryIO, TypedDict

from openai import OpenAI, OpenAIError

from src.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RETRIEVAL_TOP_K,
)
from src.logger import get_logger
from src.pdf_loader import load_pdfs
from src.prompts import QA_PROMPT_TEMPLATE
from src.text_splitter import split_documents
from src.vector_store import add_document_chunks, retrieve_relevant_chunks


logger = get_logger(__name__)


class SourceChunk(TypedDict):
    """Retrieved source chunk returned with the answer."""

    text: str
    metadata: dict
    score: float


class AnswerResult(TypedDict):
    """Answer payload returned by the RAG pipeline."""

    answer: str
    sources: list[SourceChunk]


def ingest_documents(files: list[BinaryIO]) -> int:
    """Load, split, embed, and store uploaded PDF files."""
    logger.info("Starting ingestion for %s uploaded files", len(files))
    documents = load_pdfs(files)
    if not documents:
        logger.warning("No extractable text found in uploaded PDFs")
        return 0

    chunks = split_documents(documents)
    inserted_count = add_document_chunks(chunks)
    logger.info("Completed ingestion with %s chunks", inserted_count)
    return inserted_count


def answer_question(question: str, top_k: int = RETRIEVAL_TOP_K) -> AnswerResult:
    """Answer a user question using retrieved document context."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    retrieved = retrieve_relevant_chunks(question, top_k=top_k)
    if not retrieved:
        return {
            "answer": "No relevant context was found in the uploaded documents.",
            "sources": [],
        }

    sources: list[SourceChunk] = [
        {"text": text, "metadata": metadata, "score": float(score)}
        for text, metadata, score in retrieved
    ]
    context = _format_context(sources)
    prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not configured. Using demo answer mode.")
        return {
            "answer": _build_demo_answer(question, sources),
            "sources": sources,
        }

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions grounded in user-provided PDFs.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
    except OpenAIError as exc:
        logger.warning("LLM request failed. Using demo answer mode. Reason: %s", exc)
        return {
            "answer": _build_demo_answer(question, sources),
            "sources": sources,
        }

    logger.info("Generated answer for question with %s sources", len(sources))
    return {"answer": answer.strip(), "sources": sources}


def _format_context(sources: list[SourceChunk]) -> str:
    """Format retrieved chunks for prompt injection."""
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        metadata = source["metadata"]
        citation = (
            f"Source {index}: {metadata.get('source', 'unknown')}, "
            f"page {metadata.get('page', 'unknown')}, "
            f"chunk {metadata.get('chunk', 'unknown')}"
        )
        context_blocks.append(f"{citation}\n{source['text']}")
    return "\n\n".join(context_blocks)


def _build_demo_answer(question: str, sources: list[SourceChunk]) -> str:
    """Build a simple local answer when an LLM is unavailable."""
    best_source = sources[0]
    metadata = best_source["metadata"]
    source_name = metadata.get("source", "the uploaded document")
    page = metadata.get("page", "unknown")

    return (
        "Demo mode answer: I could not call the configured LLM, so I am showing "
        "the most relevant retrieved document context instead.\n\n"
        f"Question: {question}\n\n"
        f"Most relevant source: {source_name}, page {page}\n\n"
        f"{best_source['text']}"
    )
