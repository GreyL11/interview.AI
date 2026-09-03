import threading
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

    @property
    def name(self) -> str:
        """Which detector is actually running. Surfaced in the worker's
        end-of-session metrics, because the difference between Silero and the
        energy fallback is large and the fallback is otherwise invisible."""
        return type(self).__name__


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

    @property
    def silence_run_ms(self) -> int:
        """How long the current trailing-silence run is, inside an utterance.

        0 while speech is active. Once it reaches `silence_ms` the utterance
        closes, so `silence_ms - silence_run_ms` is an upper bound on how much
        longer the final can possibly be away -- which is what lets the
        scheduler tell a doomed partial from a useful one.
        """
        return self._silence_run * FRAME_MS if self.in_speech else 0

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


def _load_silero() -> tuple[object, str, int]:
    """Build the shared ONNX session. Returns (session, mode, input_samples)."""
    import os

    import onnxruntime as ort
    from faster_whisper.vad import get_assets_path

    path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])

    inputs = session.get_inputs()
    input_names = {item.name for item in inputs}
    if {"input", "state", "sr"}.issubset(input_names):
        mode = "state"
    elif {"input", "h", "c"}.issubset(input_names):
        mode = "lstm"
    else:
        raise RuntimeError(f"Unsupported Silero VAD model inputs: {sorted(input_names)}")

    # ONNX reports a dynamic axis as a symbolic name or None, not an int, and
    # this model's sample axis is routinely dynamic. int() on that raises --
    # which used to escape all the way out to build_speech_detector() and get
    # swallowed as "Silero unavailable", silently demoting the whole session
    # to the energy fallback over a shape annotation. FRAME_SAMPLES is what
    # the capture layer produces and what Silero v5/v6 expect anyway.
    declared = next(item.shape[-1] for item in inputs if item.name == "input")
    try:
        input_samples = int(declared)
    except (TypeError, ValueError):
        input_samples = FRAME_SAMPLES
        logger.debug("silero_vad_dynamic_input_axis declared=%r; using %d",
                     declared, input_samples)

    logger.info(
        "silero_vad_loaded mode=%s input_samples=%d inputs=%s",
        mode, input_samples, sorted(input_names),
    )
    return session, mode, input_samples


#: One ONNX session for the whole process. `InferenceSession.run()` is
#: thread-safe and the recurrent state is passed in explicitly on every call
#: (as `state`, or `h`/`c`), so the session holds nothing per-channel --
#: while a fresh one per instance would reload the model for every channel
#: and again on every audio stop/start.
_silero_session: tuple[object, str, int] | None = None
_silero_lock = threading.Lock()


class SileroVad(SpeechDetector):
    """Silero VAD loaded from the ONNX asset bundled with faster-whisper."""

    def __init__(self) -> None:
        self._session = None
        self._mode = None
        self._state = None
        self._h = None
        self._c = None
        self._input_samples = None

    def _ensure_loaded(self):
        if self._session is not None:
            return self._session

        global _silero_session
        with _silero_lock:
            if _silero_session is None:
                _silero_session = _load_silero()
            self._session, self._mode, self._input_samples = _silero_session

        self.reset()
        return self._session

    def reset(self) -> None:
        if self._mode == "lstm":
            self._h = np.zeros((1, 1, 128), dtype=np.float32)
            self._c = np.zeros((1, 1, 128), dtype=np.float32)
            self._state = None
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._h = None
            self._c = None

    def probability(self, frame: np.ndarray) -> float:
        session = self._ensure_loaded()

        audio = np.asarray(frame, dtype=np.float32).reshape(-1)

        if self._input_samples is None:
            raise RuntimeError("Silero VAD input size was not initialized")

        if audio.size < self._input_samples:
            audio = np.pad(
                audio,
                (0, self._input_samples - audio.size),
                mode="constant",
            )
        elif audio.size > self._input_samples:
            audio = audio[:self._input_samples]

        audio = audio.reshape(1, -1)

        if self._mode == "lstm":
            outputs = session.run(
                None,
                {
                    "input": audio,
                    "h": self._h,
                    "c": self._c,
                },
            )
            probability, self._h, self._c = outputs

        else:
            outputs = session.run(
                None,
                {
                    "input": audio,
                    "state": self._state,
                    "sr": np.array(SAMPLE_RATE, dtype=np.int64),
                },
            )
            probability, self._state = outputs

        return float(np.asarray(probability).ravel()[0])

class EnergyVad(SpeechDetector):
    """RMS-energy fallback.

    ponytail: crude but dependency-free, and it keeps the pipeline runnable when
    the Silero asset cannot be loaded. Noticeably worse in noisy rooms — Silero
    is the real implementation, this is the safety net.

    Calibration, checked against what the capture layer actually produces:
    both channels deliver float32 in [-1, 1] (the mic via sounddevice's
    `dtype="float32"`, loopback via int16 / 32768.0), so `rms` here is a true
    full-scale fraction. With `floor=0.01` and the default
    `vad_threshold=0.5`, a frame counts as speech at rms >= 0.005, i.e. about
    -46 dBFS. Ordinary speech sits near 0.05-0.2, and a quiet room floor near
    0.001-0.003, so the default has headroom -- but a laptop fan, a noisy
    line, or AGC on the far end can cross it, and then utterances never close
    on silence. BENCHMARK REQUIRED: the right floor is a property of the
    machine's noise level, not something that can be derived here.
    """

    def __init__(self, floor: float = 0.01) -> None:
        self._floor = floor

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return min(1.0, rms / self._floor) if self._floor > 0 else 0.0


#: Said once per process, not once per channel per session. Without this the
#: same fallback line repeats on every audio start and reads like noise.
_warned_fallback = False


def build_speech_detector() -> SpeechDetector:
    """Silero if it loads, energy RMS if it does not.

    The fallback is a real downgrade, not a cosmetic one: EnergyVad cannot
    tell speech from a fan or a keyboard, so in a noisy room utterances stop
    closing on silence and only end at `vad_max_utterance_ms`. That is worth
    an ERROR, and worth naming in the worker's metrics line, because
    everything downstream looks merely "slow" rather than misconfigured.
    """
    global _warned_fallback
    try:
        detector = SileroVad()
        detector._ensure_loaded()
        return detector
    except Exception as exc:
        if not _warned_fallback:
            _warned_fallback = True
            logger.error(
                "silero_vad_unavailable falling_back_to_energy error=%s: %s. "
                "Speech detection is now RMS-only -- utterance boundaries will "
                "be unreliable in a noisy room. Check that faster-whisper is "
                "installed and its bundled VAD asset is present.",
                type(exc).__name__, exc,
            )
        return EnergyVad()
