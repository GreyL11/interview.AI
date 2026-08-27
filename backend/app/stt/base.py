from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class Transcript(BaseModel):
    text: str
    is_final: bool
    language: str = "en"
    duration_ms: int = 0


class SttError(Exception):
    """Raised when speech-to-text cannot run."""


class SttEngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, is_final: bool) -> Transcript:
        """Transcribe 16kHz mono float32 audio.

        `is_final` lets an implementation trade accuracy for latency on interim
        passes and spend more effort on the last one.
        """
        ...

    def warmup(self) -> None:
        """Optionally load the model ahead of first use, so the first question
        of a session isn't the one that pays for it."""
        return
