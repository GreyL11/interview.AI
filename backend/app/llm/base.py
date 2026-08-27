from abc import ABC, abstractmethod

from app.schemas.answer import Answer


class LLMClient(ABC):
    """Interface for the reasoning backend. Business logic depends only on
    this, so Gemini can be swapped for another provider without touching
    the orchestrator."""

    @abstractmethod
    async def generate_answer(self, prompt: str) -> Answer:
        ...


class LLMError(Exception):
    """Raised on provider failure (timeout, API error, or a response that
    doesn't parse into a valid Answer)."""
