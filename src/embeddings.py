"""Embedding model adapter backed by sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from src.config import MODEL_NAME
from src.logger import get_logger


logger = get_logger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embedding wrapper with a local fallback."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = None
        self.dimension = 384
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            logger.info("Loaded SentenceTransformer embedding model: %s", model_name)
        except Exception as exc:
            logger.warning(
                "SentenceTransformer could not be loaded. "
                "Falling back to deterministic hash embeddings. Reason: %s",
                exc,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of documents."""
        if self.model is None:
            return [_hash_embedding(text, self.dimension) for text in texts]

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""
        if self.model is None:
            return _hash_embedding(text, self.dimension)

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    """Return a cached embedding model instance."""
    return SentenceTransformerEmbeddings()


def _hash_embedding(text: str, dimension: int) -> list[float]:
    """Create a deterministic bag-of-words embedding for local development."""
    vector = [0.0] * dimension
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
