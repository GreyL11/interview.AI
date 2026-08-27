from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

#: Whisper and Silero VAD both expect 16kHz mono float32.
SAMPLE_RATE = 16_000
#: 32ms at 16kHz. Silero VAD wants 512-sample frames; 512 is one 32ms hop.
FRAME_SAMPLES = 512


class AudioChannel(StrEnum):
    MIC = "MIC"           # the candidate
    LOOPBACK = "LOOPBACK"  # system audio: the interviewer over Zoom/Meet


class DeviceInfo(BaseModel):
    index: int
    name: str
    channel: AudioChannel
    sample_rate: int = SAMPLE_RATE
    is_default: bool = False


class AudioError(Exception):
    """Raised when a capture device cannot be opened or read."""


class AudioSource(ABC):
    """A single capture stream, already resampled to 16kHz mono float32.

    One instance per channel. Keeping mic and loopback as separate sources is
    what makes speaker attribution reliable: it is a fact about which device the
    audio came from, not an acoustic guess about who was speaking.
    """

    @property
    @abstractmethod
    def channel(self) -> AudioChannel:
        ...

    @abstractmethod
    def describe(self) -> DeviceInfo:
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield FRAME_SAMPLES-length float32 frames until stop() is called."""
        ...
