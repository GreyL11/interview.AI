from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Turns text into L2-normalised vectors.

    Normalisation is part of the contract, not an implementation detail: the
    vector store uses inner product, so unit-length vectors make the score
    cosine similarity and RAG_MIN_SIMILARITY a meaningful [-1, 1] threshold.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dimension) float32 array of unit vectors."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Return a single (dimension,) float32 unit vector."""
        return self.embed([query])[0]


class EmbeddingError(Exception):
    """Raised when the embedding backend cannot produce vectors."""
