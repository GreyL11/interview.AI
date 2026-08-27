import asyncio
import threading

import numpy as np

from app.audio.base import SAMPLE_RATE, AudioChannel, AudioSource
from app.core.config import settings
from app.core.logging import get_logger
from app.sessions.schemas import TranscriptSource
from app.stt.base import SttEngine, SttError
from app.stt.vad import FRAME_MS, SegmentEvent, Segmenter, SpeechDetector

logger = get_logger(__name__)

_CHANNEL_TO_SOURCE = {
    AudioChannel.MIC: TranscriptSource.MIC,
    AudioChannel.LOOPBACK: TranscriptSource.LOOPBACK,
}


class TranscriptionWorker:
    """Runs one channel's audio -> VAD -> STT loop on a worker thread.

    Lives on a thread rather than the event loop because CTranslate2 inference
    is CPU-bound; it publishes results back with call_soon_threadsafe so the
    session's async code never sees a thread.
    """

    def __init__(
        self,
        source: AudioSource,
        detector: SpeechDetector,
        engine: SttEngine,
        loop: asyncio.AbstractEventLoop,
        on_transcript,
        partial_interval_ms: int | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._engine = engine
        self._loop = loop
        self._on_transcript = on_transcript
        self._partial_interval_ms = (
            partial_interval_ms if partial_interval_ms is not None
            else settings.stt_partial_interval_ms
        )
        self._segmenter = Segmenter()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.errors = 0

    @property
    def transcript_source(self) -> TranscriptSource:
        return _CHANNEL_TO_SOURCE[self._source.channel]

    def start(self) -> None:
        self._stop.clear()
        self._source.start()
        self._thread = threading.Thread(
            target=self._run, name=f"stt-{self._source.channel.value}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _publish(self, text: str, is_final: bool) -> None:
        if not text.strip():
            return
        asyncio.run_coroutine_threadsafe(
            self._on_transcript(text, self.transcript_source, is_final), self._loop
        )

    def _run(self) -> None:
        buffer: list[np.ndarray] = []
        # Cadence is measured in buffered audio, not wall clock. With live
        # capture the two are the same, but this keeps the behaviour identical
        # when frames arrive faster than real time (tests, replayed recordings).
        last_partial_ms = 0

        try:
            for frame in self._source.frames():
                if self._stop.is_set():
                    break

                event = self._segmenter.push(self._detector.probability(frame))

                if event == SegmentEvent.SPEECH_START:
                    buffer = [frame]
                    last_partial_ms = 0
                    continue

                if event == SegmentEvent.NONE:
                    continue

                buffer.append(frame)

                if event == SegmentEvent.SPEECH_CONTINUE:
                    buffered_ms = len(buffer) * FRAME_MS
                    if buffered_ms - last_partial_ms >= self._partial_interval_ms:
                        last_partial_ms = buffered_ms
                        self._transcribe(buffer, is_final=False)
                    continue

                if event == SegmentEvent.SPEECH_END:
                    self._transcribe(buffer, is_final=True)
                    buffer = []
                    self._detector.reset()

        except Exception:
            logger.exception("transcription_worker_crashed channel=%s", self._source.channel.value)
        finally:
            logger.info("transcription_worker_stopped channel=%s", self._source.channel.value)

    def _transcribe(self, buffer: list[np.ndarray], is_final: bool) -> None:
        if not buffer:
            return
        audio = np.concatenate(buffer)
        # ponytail: interim passes re-transcribe the whole utterance rather than
        # streaming incrementally. Utterances are bounded by vad_max_utterance_ms,
        # so the repeated work is small; revisit if that cap ever grows.
        try:
            transcript = self._engine.transcribe(audio, is_final=is_final)
        except SttError as exc:
            self.errors += 1
            logger.warning("transcription_failed final=%s error=%s", is_final, exc)
            return
        duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        logger.debug(
            "transcribed final=%s ms=%d chars=%d", is_final, duration_ms, len(transcript.text)
        )
        self._publish(transcript.text, is_final)


class AudioPipeline:
    """All capture channels for one session."""

    def __init__(self, workers: list[TranscriptionWorker]) -> None:
        self._workers = workers

    @property
    def channels(self) -> list[TranscriptSource]:
        return [w.transcript_source for w in self._workers]

    def start(self) -> None:
        started: list[TranscriptionWorker] = []
        for worker in self._workers:
            try:
                worker.start()
                started.append(worker)
            except Exception:
                logger.exception("worker_start_failed; stopping the ones already running")
                for running in started:
                    running.stop()
                raise
        self._workers = started

    def stop(self) -> None:
        for worker in self._workers:
            worker.stop()
