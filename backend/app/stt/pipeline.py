import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from app.audio.base import SAMPLE_RATE, AudioChannel, AudioSource
from app.core.config import settings
from app.core.logging import get_logger
from app.sessions.schemas import TranscriptSource
from app.stt.base import SttEngine, SttError
from app.stt.vad import FRAME_MS, SegmentEvent, Segmenter, SpeechDetector

logger = get_logger(__name__)

_LOOPBACK_DIAGNOSTIC_INTERVAL = 500

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
        self._schedule_lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._partial_future: Future | None = None
        self._pending_partial: tuple[int, np.ndarray] | None = None
        self._utterance_id = 0
        self._final_utterance_id = 0
        self._published_final_utterance_id = 0
        self._stopping = False
        self._loopback_frame_count = 0
        self.frames_consumed = 0
        self.partials_scheduled = 0
        self.partials_coalesced = 0
        self.errors = 0

    @property
    def transcript_source(self) -> TranscriptSource:
        return _CHANNEL_TO_SOURCE[self._source.channel]

    def start(self) -> None:
        self._stop.clear()
        self._stopping = False
        self._utterance_id = 0
        self._final_utterance_id = 0
        self._published_final_utterance_id = 0
        self._pending_partial = None
        self._partial_future = None
        self._source.start()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"stt-infer-{self._source.channel.value.lower()}",
        )
        self._thread = threading.Thread(
            target=self._run, name=f"stt-{self._source.channel.value}", daemon=True
        )
        self._thread.start()
        if self._source.channel == AudioChannel.LOOPBACK:
            logger.debug("loopback_worker_thread_started name=%s", self._thread.name)

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._schedule_lock:
            self._stopping = True
            self._pending_partial = None
        self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        with self._schedule_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            # Do not cancel a final already queued behind a partial. Shutdown
            # waits for that ordered work and avoids leaking a worker thread.
            executor.shutdown(wait=True)
        logger.info(
            "transcription_worker_metrics channel=%s frames=%d partials_scheduled=%d "
            "partials_coalesced=%d errors=%d",
            self._source.channel.value,
            self.frames_consumed,
            self.partials_scheduled,
            self.partials_coalesced,
            self.errors,
        )

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
            if self._source.channel == AudioChannel.LOOPBACK:
                logger.debug("loopback_worker_run_entered")
            for frame in self._source.frames():
                if self._stop.is_set():
                    break

                self.frames_consumed += 1

                is_loopback = self._source.channel == AudioChannel.LOOPBACK
                self._loopback_frame_count += int(is_loopback)
                should_log_frame = (
                    is_loopback
                    and self._loopback_frame_count % _LOOPBACK_DIAGNOSTIC_INTERVAL == 0
                )
                if should_log_frame:
                    logger.debug(
                        "loopback_worker_frame_received count=%d shape=%s dtype=%s rms=%.6f",
                        self._loopback_frame_count,
                        frame.shape,
                        frame.dtype,
                        float(np.sqrt(np.mean(np.square(frame)))),
                    )

                try:
                    if should_log_frame:
                        logger.debug(
                            "loopback_vad_probability_begin count=%d", self._loopback_frame_count
                        )
                    probability = self._detector.probability(frame)
                except Exception:
                    if is_loopback:
                        logger.exception(
                            "loopback_vad_probability_failed count=%d",
                            self._loopback_frame_count,
                        )
                    raise

                if should_log_frame:
                    logger.debug(
                        "loopback_vad_probability_end count=%d probability=%.6f",
                        self._loopback_frame_count,
                        probability,
                    )
                event = self._segmenter.push(probability)
                if is_loopback and (
                    event in (SegmentEvent.SPEECH_START, SegmentEvent.SPEECH_END)
                    or should_log_frame
                ):
                    logger.debug(
                        "loopback_segment_event count=%d probability=%.6f event=%s",
                        self._loopback_frame_count,
                        probability,
                        event,
                    )

                if event == SegmentEvent.SPEECH_START:
                    self._utterance_id += 1
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
                        self._schedule_partial(self._utterance_id, buffer)
                    continue

                if event == SegmentEvent.SPEECH_END:
                    self._schedule_final(self._utterance_id, buffer)
                    buffer = []
                    self._detector.reset()

        except Exception:
            logger.exception("transcription_worker_crashed channel=%s", self._source.channel.value)
        finally:
            logger.info("transcription_worker_stopped channel=%s", self._source.channel.value)

    def _schedule_partial(self, utterance_id: int, buffer: list[np.ndarray]) -> None:
        audio = np.concatenate(buffer)
        with self._schedule_lock:
            if self._stopping or utterance_id <= self._final_utterance_id:
                return
            if self._partial_future is not None and not self._partial_future.done():
                self._pending_partial = (utterance_id, audio)
                self.partials_coalesced += 1
                return
            self._submit_partial_locked(utterance_id, audio)

    def _submit_partial_locked(self, utterance_id: int, audio: np.ndarray) -> None:
        if self._executor is None or self._stopping:
            return
        future = self._executor.submit(self._transcribe, audio, False, utterance_id)
        self._partial_future = future
        self.partials_scheduled += 1
        future.add_done_callback(self._partial_finished)
        logger.debug(
            "partial_transcription_scheduled channel=%s utterance=%d samples=%d queue_backlog=%s",
            self._source.channel.value,
            utterance_id,
            len(audio),
            self._source_queue_backlog(),
        )

    def _partial_finished(self, future: Future) -> None:
        with self._schedule_lock:
            if self._partial_future is future:
                self._partial_future = None
            pending = self._pending_partial
            self._pending_partial = None
            if (
                pending is not None
                and pending[0] > self._final_utterance_id
                and not self._stopping
            ):
                self._submit_partial_locked(*pending)

    def _schedule_final(self, utterance_id: int, buffer: list[np.ndarray]) -> None:
        if not buffer:
            return
        audio = np.concatenate(buffer)
        with self._schedule_lock:
            if self._executor is None:
                return
            self._final_utterance_id = max(self._final_utterance_id, utterance_id)
            if self._pending_partial is not None and self._pending_partial[0] == utterance_id:
                self._pending_partial = None
            self._executor.submit(self._transcribe, audio, True, utterance_id)
        logger.debug(
            "final_transcription_scheduled channel=%s utterance=%d samples=%d queue_backlog=%s",
            self._source.channel.value,
            utterance_id,
            len(audio),
            self._source_queue_backlog(),
        )

    def _source_queue_backlog(self) -> int | None:
        queue = getattr(self._source, "_queue", None)
        qsize = getattr(queue, "qsize", None)
        return qsize() if callable(qsize) else None

    def _transcribe(self, audio: np.ndarray, is_final: bool, utterance_id: int) -> None:
        started = time.monotonic()
        # Interim passes re-transcribe the whole utterance snapshot. The audio
        # thread coalesces snapshots while one is in flight, bounding work to a
        # single active and a single newest pending partial per channel.
        try:
            transcript = self._engine.transcribe(audio, is_final=is_final)
        except SttError as exc:
            self.errors += 1
            logger.warning("transcription_failed final=%s error=%s", is_final, exc)
            return
        duration_seconds = time.monotonic() - started
        duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        logger.info(
            "transcription_completed channel=%s final=%s utterance=%d audio_ms=%d "
            "inference_seconds=%.3f chars=%d",
            self._source.channel.value,
            is_final,
            utterance_id,
            duration_ms,
            duration_seconds,
            len(transcript.text),
        )
        if not is_final:
            with self._schedule_lock:
                if self._stopping or utterance_id <= self._published_final_utterance_id:
                    logger.debug(
                        "partial_transcription_suppressed channel=%s utterance=%d",
                        self._source.channel.value,
                        utterance_id,
                    )
                    return
        else:
            with self._schedule_lock:
                self._published_final_utterance_id = max(
                    self._published_final_utterance_id, utterance_id
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
