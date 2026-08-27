from abc import ABC, abstractmethod


class Retriever(ABC):
    """Interface for retrieving personal/knowledge-base context relevant to a question.
    Phase 1 uses MockRetriever; a future phase swaps in FAISS + SQLite without
    touching callers."""

    @abstractmethod
    async def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        """Return up to top_k relevant context snippets, most relevant first."""
        ...
