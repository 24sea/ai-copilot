"""Embedding model adapter backed by sentence-transformers."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embedding wrapper for SentenceTransformer."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of documents."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    """Return a cached embedding model instance."""
    return SentenceTransformerEmbeddings()
