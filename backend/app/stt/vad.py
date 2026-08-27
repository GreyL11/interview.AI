from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from app.audio.base import FRAME_SAMPLES, SAMPLE_RATE
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32ms


class SegmentEvent(StrEnum):
    NONE = "NONE"
    SPEECH_START = "SPEECH_START"
    SPEECH_CONTINUE = "SPEECH_CONTINUE"
    SPEECH_END = "SPEECH_END"


class SpeechDetector(ABC):
    """Per-frame speech probability. Separated from segmentation so the state
    machine can be tested without a model."""

    @abstractmethod
    def probability(self, frame: np.ndarray) -> float:
        ...

    def reset(self) -> None:
        return


@dataclass
class Segmenter:
    """Turns per-frame speech probabilities into utterance boundaries.

    Hysteresis in both directions: several speech frames are needed to open an
    utterance (so a cough or keyboard click doesn't), and a sustained silence to
    close it (so a mid-sentence pause doesn't cut the question in half). The
    max-duration cap exists because someone who never pauses would otherwise
    never produce a final transcript.
    """

    threshold: float = field(default_factory=lambda: settings.vad_threshold)
    start_frames: int = field(default_factory=lambda: settings.vad_start_frames)
    silence_ms: int = field(default_factory=lambda: settings.vad_silence_ms)
    max_utterance_ms: int = field(default_factory=lambda: settings.vad_max_utterance_ms)
    min_utterance_ms: int = field(default_factory=lambda: settings.vad_min_utterance_ms)

    in_speech: bool = False
    _speech_run: int = 0
    _silence_run: int = 0
    _duration_ms: int = 0

    def reset(self) -> None:
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._duration_ms = 0

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    def push(self, probability: float) -> SegmentEvent:
        is_speech = probability >= self.threshold

        if not self.in_speech:
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self.start_frames:
                self.in_speech = True
                self._silence_run = 0
                # Count the frames that opened the utterance, not just what follows.
                self._duration_ms = self._speech_run * FRAME_MS
                self._speech_run = 0
                return SegmentEvent.SPEECH_START
            return SegmentEvent.NONE

        self._duration_ms += FRAME_MS
        self._silence_run = 0 if is_speech else self._silence_run + 1

        if self._silence_run * FRAME_MS >= self.silence_ms:
            too_short = self._duration_ms < self.min_utterance_ms
            self.reset()
            # A blip that cleared the start threshold but carries no real speech
            # is discarded rather than sent to the model.
            return SegmentEvent.NONE if too_short else SegmentEvent.SPEECH_END

        if self._duration_ms >= self.max_utterance_ms:
            logger.info("vad_max_duration_reached ms=%d", self._duration_ms)
            self.reset()
            return SegmentEvent.SPEECH_END

        return SegmentEvent.SPEECH_CONTINUE


class SileroVad(SpeechDetector):
    """Silero VAD, loaded from the ONNX asset bundled inside faster-whisper.

    No separate download and no torch: the model ships with the package, so VAD
    works offline on first run.
    """

    def __init__(self) -> None:
        self._session = None
        self._state = None

    def _ensure_loaded(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
            from faster_whisper.vad import get_assets_path
        except ImportError as exc:
            raise RuntimeError(f"Silero VAD is unavailable: {exc}") from exc

        import os

        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.reset()
        return self._session

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        session = self._ensure_loaded()
        audio = frame.reshape(1, -1).astype(np.float32)
        out, self._state = session.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        return float(np.asarray(out).ravel()[0])


class EnergyVad(SpeechDetector):
    """RMS-energy fallback.

    ponytail: crude but dependency-free, and it keeps the pipeline runnable when
    the Silero asset cannot be loaded. Noticeably worse in noisy rooms — Silero
    is the real implementation, this is the safety net.
    """

    def __init__(self, floor: float = 0.01) -> None:
        self._floor = floor

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return min(1.0, rms / self._floor) if self._floor > 0 else 0.0


def build_speech_detector() -> SpeechDetector:
    try:
        detector = SileroVad()
        detector._ensure_loaded()
        return detector
    except Exception as exc:
        logger.warning("silero_vad_unavailable falling_back_to_energy error=%s", exc)
        return EnergyVad()
