from abc import ABC, abstractmethod

from app.documents.schemas import Chunk, NormalizedDocument


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: NormalizedDocument, **metadata) -> list[Chunk]:
        """Split a document into ordered, overlapping chunks. Deterministic:
        the same input must always produce the same chunks."""
        ...
