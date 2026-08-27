import hashlib
import threading
from collections.abc import Iterator

import numpy as np

from app.audio.base import FRAME_SAMPLES, AudioChannel, AudioSource, DeviceInfo
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMClient
from app.schemas.answer import Answer
from app.stt.base import SttEngine, Transcript
from app.stt.vad import SpeechDetector

DIMENSION = 32


class FakeAudioSource(AudioSource):
    """Replays a fixed array of frames instead of opening a device.

    Everything downstream of capture is testable this way — no microphone, no
    PortAudio, and identical output on every run.
    """

    def __init__(self, frames: list[np.ndarray], channel: AudioChannel = AudioChannel.LOOPBACK) -> None:
        self._frames = frames
        self._channel = channel
        self._running = False
        self.exhausted = threading.Event()

    @property
    def channel(self) -> AudioChannel:
        return self._channel

    def describe(self) -> DeviceInfo:
        return DeviceInfo(index=0, name="fake", channel=self._channel)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def frames(self) -> Iterator[np.ndarray]:
        for frame in self._frames:
            if not self._running:
                break
            yield frame
        self.exhausted.set()


class ScriptedSpeechDetector(SpeechDetector):
    """Returns a predetermined probability per frame, so segmentation can be
    tested against an exact speech/silence pattern."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._index = 0
        self.resets = 0

    def probability(self, frame: np.ndarray) -> float:
        value = self._probabilities[min(self._index, len(self._probabilities) - 1)]
        self._index += 1
        return value

    def reset(self) -> None:
        self.resets += 1


class FakeSttEngine(SttEngine):
    """Returns scripted text, tracking how it was called."""

    def __init__(self, final_text: str = "How would you handle duplicate records?") -> None:
        self.final_text = final_text
        self.calls: list[tuple[int, bool]] = []

    def transcribe(self, audio: np.ndarray, is_final: bool) -> Transcript:
        self.calls.append((len(audio), is_final))
        text = self.final_text if is_final else self.final_text[: len(self.final_text) // 2]
        return Transcript(text=text, is_final=is_final, duration_ms=len(audio) // 16)


def speech_frames(count: int, amplitude: float = 0.3) -> list[np.ndarray]:
    rng = np.random.default_rng(1234)  # seeded: identical audio every run
    return [
        (rng.standard_normal(FRAME_SAMPLES) * amplitude).astype(np.float32)
        for _ in range(count)
    ]


def silence_frames(count: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(count)]


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
