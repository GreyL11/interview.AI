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
