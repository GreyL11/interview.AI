import hashlib

import numpy as np

from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMClient
from app.schemas.answer import Answer

DIMENSION = 32


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embeddings with no model and no network.

    Vectors are seeded from the set of words in the text, so texts sharing
    vocabulary land near each other and unrelated texts do not. That is enough
    to test retrieval ordering and thresholds without the real model's 90MB
    download or its nondeterministic-looking float noise.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        words = {w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")}
        for word in words:
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).reshape(1, -1)


class FakeLLM(LLMClient):
    def __init__(self, answer: Answer | None = None) -> None:
        self.answer = answer or Answer(
            summary="Make the pipeline idempotent.",
            key_points=["Identify the duplicate key", "Deduplicate on ingest"],
            detailed_answer="Full explanation.",
        )
        self.prompts: list[str] = []

    async def generate_answer(self, prompt: str) -> Answer:
        self.prompts.append(prompt)
        return self.answer


class SlowStreamingLLM(LLMClient):
    """Streams an answer in small chunks with a controllable delay.

    The delay is what makes cancellation testable: it holds a turn open long
    enough for a second question to supersede it.
    """

    def __init__(self, answer: Answer | None = None, chunk_delay: float = 0.02) -> None:
        self.answer = answer or Answer(
            summary="Stream the answer progressively.",
            key_points=["one", "two"],
            detailed_answer="detail",
        )
        self.chunk_delay = chunk_delay
        self.prompts: list[str] = []
        self.started = 0
        self.cancelled = 0

    async def generate_answer(self, prompt: str) -> Answer:
        self.prompts.append(prompt)
        return self.answer

    async def stream_answer(self, prompt: str):
        import asyncio

        self.prompts.append(prompt)
        self.started += 1
        payload = self.answer.model_dump_json()
        try:
            for i in range(0, len(payload), 12):
                await asyncio.sleep(self.chunk_delay)
                yield payload[i : i + 12]
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class BrokenLLM(LLMClient):
    def __init__(self, message: str = "provider exploded") -> None:
        self.message = message

    async def generate_answer(self, prompt: str) -> Answer:
        from app.llm.base import LLMError

        raise LLMError(self.message)

    async def stream_answer(self, prompt: str):
        from app.llm.base import LLMError

        raise LLMError(self.message)
        yield ""  # pragma: no cover - unreachable, keeps this an async generator
