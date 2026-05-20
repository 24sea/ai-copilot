"""Vector store operations with ChromaDB and a local fallback."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.embeddings import get_embedding_model
from src.logger import get_logger
from src.pdf_loader import Document


logger = get_logger(__name__)
FALLBACK_STORE_PATH = Path(CHROMA_PERSIST_DIR) / "fallback_vector_store.json"


def get_vector_store():
    """Create or load the persistent Chroma vector store."""
    try:
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        logger.warning(
            "ChromaDB could not be loaded. Falling back to a local JSON vector "
            "store. Reason: %s",
            exc,
        )
        return LocalJsonVectorStore(FALLBACK_STORE_PATH)


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

    if isinstance(vector_store, LocalJsonVectorStore):
        return results
    return [(doc.page_content, doc.metadata, score) for doc, score in results]


class LocalJsonVectorStore:
    """Small persistent vector store for local development."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = get_embedding_model()

    def add_texts(self, texts: list[str], metadatas: list[dict]) -> None:
        """Embed and persist texts with metadata."""
        records = self._load_records()
        embeddings = self.embedding_model.embed_documents(texts)

        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            records.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding,
                },
            )

        self._save_records(records)

    def similarity_search_with_score(
        self,
        question: str,
        k: int,
    ) -> list[tuple[str, dict, float]]:
        """Return the top-k closest records by cosine similarity."""
        records = self._load_records()
        if not records:
            return []

        query_embedding = self.embedding_model.embed_query(question)
        scored_records = [
            (
                record["text"],
                record["metadata"],
                _cosine_distance(query_embedding, record["embedding"]),
            )
            for record in records
        ]
        scored_records.sort(key=lambda item: item[2])
        return scored_records[:k]

    def _load_records(self) -> list[dict]:
        if not self.store_path.exists():
            return []

        with self.store_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_records(self, records: list[dict]) -> None:
        with self.store_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)


def _cosine_distance(first: list[float], second: list[float]) -> float:
    """Calculate cosine distance where lower means more similar."""
    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))

    if first_norm == 0 or second_norm == 0:
        return 1.0
    return 1.0 - (dot_product / (first_norm * second_norm))
