from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Manages vectors and their integer ids. Knows nothing about documents,
    parsing, or chunking — ids are the only shared vocabulary with the rest of
    the system."""

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return (vector_id, score) pairs, best first."""
        ...

    @abstractmethod
    def remove(self, ids: list[int]) -> int:
        """Remove vectors by id. Returns the number actually removed."""
        ...

    @abstractmethod
    def persist(self) -> None:
        ...


class VectorStoreError(Exception):
    ...
