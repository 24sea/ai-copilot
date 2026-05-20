"""ChromaDB vector store operations."""

from __future__ import annotations

from langchain_community.vectorstores import Chroma

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.embeddings import get_embedding_model
from src.logger import get_logger
from src.pdf_loader import Document


logger = get_logger(__name__)


def get_vector_store() -> Chroma:
    """Create or load the persistent Chroma vector store."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_document_chunks(chunks: list[Document]) -> int:
    """Add chunks to ChromaDB and return the number of inserted chunks."""
    if not chunks:
        logger.warning("No chunks were provided for ingestion")
        return 0

    vector_store = get_vector_store()
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    logger.info("Added %s chunks to ChromaDB", len(chunks))
    return len(chunks)


def retrieve_relevant_chunks(question: str, top_k: int) -> list[tuple[str, dict, float]]:
    """Retrieve relevant chunks and similarity scores for a question."""
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=top_k)
    return [(doc.page_content, doc.metadata, score) for doc, score in results]
