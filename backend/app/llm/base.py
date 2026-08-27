from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas.answer import Answer


class LLMClient(ABC):
    """Interface for the reasoning backend. Business logic depends only on
    this, so Gemini can be swapped for another provider without touching
    the orchestrator."""

    @abstractmethod
    async def generate_answer(self, prompt: str) -> Answer:
        ...

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        """Yield raw response text as it arrives.

        Default implementation falls back to the non-streaming call and emits
        one chunk, so every client satisfies the streaming contract even if the
        provider cannot stream.
        """
        answer = await self.generate_answer(prompt)
        yield answer.model_dump_json()


class LLMError(Exception):
    """Raised on provider failure (timeout, API error, or a response that
    doesn't parse into a valid Answer)."""
