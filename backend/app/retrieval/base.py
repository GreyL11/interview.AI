from abc import ABC, abstractmethod

from app.documents.schemas import KnowledgeType, RetrievedChunk


class Retriever(ABC):
    """Interface for retrieving personal/knowledge-base context relevant to a
    question. MockRetriever is kept for tests that need a guaranteed-empty
    knowledge base; LocalRetriever is the real FAISS + SQLite implementation."""

    @abstractmethod
    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_types: list[KnowledgeType] | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return relevant chunks, most relevant first. An empty list means no
        grounded context was found — callers must treat that as 'the user has
        not told us they did this', never as 'no filter applied'."""
        ...
