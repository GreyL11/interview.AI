import asyncio
import threading
import time

import numpy as np

from app.audio.base import SAMPLE_RATE, AudioChannel, AudioSource
from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.sessions.schemas import TranscriptSource
from app.stt.base import SttEngine, SttError
from app.stt.scheduler import (
    WARMUP_PRIORITY,
    InferenceJob,
    InferenceScheduler,
    shared_scheduler,
)
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

    Inference itself is not run here. Every channel submits into one shared
    priority scheduler, because they all share one Whisper model and would
    otherwise queue against each other in arrival order — see stt/scheduler.py.
    """

    def __init__(
        self,
        source: AudioSource,
        detector: SpeechDetector,
        engine: SttEngine,
        loop: asyncio.AbstractEventLoop,
        on_transcript,
        partial_interval_ms: int | None = None,
        scheduler: InferenceScheduler | None = None,
        session_id: str | None = None,
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
        self._scheduler = scheduler or shared_scheduler()
        self._session_id = session_id
        self._segmenter = Segmenter()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._schedule_lock = threading.RLock()
        self._acquired = False
        self._partial_job: InferenceJob | None = None
        self._pending_partial: tuple[int, np.ndarray] | None = None
        self._outstanding: set[InferenceJob] = set()
        self._utterance_id = 0
        self._final_utterance_id = 0
        self._published_final_utterance_id = 0
        self._partials_this_utterance = 0
        #: Self-tuning back-off: never re-queue a partial more often than the
        #: last one took to run, so a slow machine simply produces fewer of them.
        self._last_partial_inference_ms = 0.0
        self._speech_end_at: dict[int, float] = {}
        self._stopping = False
        self._loopback_frame_count = 0
        self.frames_consumed = 0
        self.partials_scheduled = 0
        self.partials_coalesced = 0
        self.partials_skipped = 0
        self.partials_cancelled = 0
        self.errors = 0

    @property
    def transcript_source(self) -> TranscriptSource:
        return _CHANNEL_TO_SOURCE[self._source.channel]

    @property
    def channel(self) -> AudioChannel:
        return self._source.channel

    def start(self) -> None:
        self._stop.clear()
        self._stopping = False
        self._utterance_id = 0
        self._final_utterance_id = 0
        self._published_final_utterance_id = 0
        self._pending_partial = None
        self._partial_job = None
        self._outstanding.clear()
        self._speech_end_at.clear()
        self._source.start()
        self._scheduler.acquire()
        self._acquired = True
        try:
            self._thread = threading.Thread(
                target=self._run, name=f"stt-{self._source.channel.value}", daemon=True
            )
            self._thread.start()
        except BaseException:
            # Never leave a reference behind on a worker that never ran, or the
            # scheduler's threads outlive every session that used them.
            self._acquired = False
            self._scheduler.release()
            raise
        if self._source.channel == AudioChannel.LOOPBACK:
            logger.debug("loopback_worker_thread_started name=%s", self._thread.name)

    def warmup(self) -> None:
        """Load the model ahead of the first utterance.

        Submitted at the top of the queue rather than run inline: it has to
        happen before any real inference anyway, and doing it on the scheduler
        keeps it inside the same shutdown path as everything else.
        """
        self._scheduler.submit(
            self._engine.warmup,
            channel=self._source.channel,
            priority=WARMUP_PRIORITY,
        )

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._schedule_lock:
            self._stopping = True
            self._pending_partial = None
            outstanding = list(self._outstanding)
        self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        # A queued partial is worthless now; a queued final is the transcript of
        # something that was actually said, so it is waited for rather than cut.
        finals = []
        for job in outstanding:
            if job.is_final:
                finals.append(job)
            elif job.cancel():
                self.partials_cancelled += 1
        for job in finals:
            job.wait(timeout=timeout)

        if self._acquired:
            self._acquired = False
            self._scheduler.release(timeout=timeout)

        logger.info(
            "transcription_worker_metrics channel=%s frames=%d partials_scheduled=%d "
            "partials_coalesced=%d partials_skipped=%d partials_cancelled=%d errors=%d",
            self._source.channel.value,
            self.frames_consumed,
            self.partials_scheduled,
            self.partials_coalesced,
            self.partials_skipped,
            self.partials_cancelled,
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
                    self._partials_this_utterance = 0
                    log_metric(
                        "speech_start_detected",
                        session_id=self._session_id,
                        channel=self._source.channel.value,
                        utterance_id=self._utterance_id,
                    )
                    continue

                if event == SegmentEvent.NONE:
                    continue

                buffer.append(frame)

                if event == SegmentEvent.SPEECH_CONTINUE:
                    buffered_ms = len(buffer) * FRAME_MS
                    if buffered_ms - last_partial_ms >= self._next_partial_after_ms():
                        last_partial_ms = buffered_ms
                        self._schedule_partial(self._utterance_id, buffer, buffered_ms)
                    continue

                if event == SegmentEvent.SPEECH_END:
                    log_metric(
                        "speech_end_detected",
                        session_id=self._session_id,
                        channel=self._source.channel.value,
                        utterance_id=self._utterance_id,
                        audio_duration_ms=len(buffer) * FRAME_MS,
                    )
                    self._schedule_final(self._utterance_id, buffer)
                    buffer = []
                    self._detector.reset()

        except Exception:
            logger.exception("transcription_worker_crashed channel=%s", self._source.channel.value)
        finally:
            logger.info("transcription_worker_stopped channel=%s", self._source.channel.value)

    # ------------------------------------------------------------- scheduling

    def _next_partial_after_ms(self) -> float:
        """Adaptive cadence: the configured interval, or the time the last
        partial actually took, whichever is longer."""
        return max(self._partial_interval_ms, self._last_partial_inference_ms)

    def _schedule_partial(
        self, utterance_id: int, buffer: list[np.ndarray], buffered_ms: int
    ) -> None:
        if not settings.stt_enable_partials:
            return
        if buffered_ms < settings.stt_partial_min_audio_ms:
            self.partials_skipped += 1
            return
        with self._schedule_lock:
            if self._stopping or utterance_id <= self._final_utterance_id:
                return
            if self._partials_this_utterance >= settings.stt_max_partials_per_utterance:
                self.partials_skipped += 1
                return
            audio = np.concatenate(buffer)
            if self._partial_job is not None and not self._partial_job.finished:
                self._pending_partial = (utterance_id, audio)
                self.partials_coalesced += 1
                return
            self._submit_partial_locked(utterance_id, audio)

    def _submit_partial_locked(self, utterance_id: int, audio: np.ndarray) -> None:
        if self._stopping:
            return
        job = self._scheduler.submit(
            lambda: self._transcribe(audio, False, utterance_id),
            channel=self._source.channel,
            utterance_id=utterance_id,
            is_final=False,
        )
        if job is None:
            return
        self._partial_job = job
        self._outstanding.add(job)
        self._partials_this_utterance += 1
        self.partials_scheduled += 1
        logger.debug(
            "partial_transcription_scheduled channel=%s utterance=%d samples=%d "
            "queue_depth=%d",
            self._source.channel.value,
            utterance_id,
            len(audio),
            self._scheduler.depth,
        )

    def _schedule_final(self, utterance_id: int, buffer: list[np.ndarray]) -> None:
        if not buffer:
            return
        audio = np.concatenate(buffer)
        with self._schedule_lock:
            self._final_utterance_id = max(self._final_utterance_id, utterance_id)
            self._speech_end_at[utterance_id] = time.monotonic()
            if self._pending_partial is not None and self._pending_partial[0] <= utterance_id:
                self._pending_partial = None
            # Anything still queued for this utterance is now dead weight in
            # front of the only job that matters.
            self._cancel_stale_partials_locked(utterance_id)
            job = self._scheduler.submit(
                lambda: self._transcribe(audio, True, utterance_id),
                channel=self._source.channel,
                utterance_id=utterance_id,
                is_final=True,
            )
            if job is None:
                return
            self._outstanding.add(job)
        logger.debug(
            "final_transcription_scheduled channel=%s utterance=%d samples=%d queue_depth=%d",
            self._source.channel.value,
            utterance_id,
            len(audio),
            self._scheduler.depth,
        )

    def _cancel_stale_partials_locked(self, utterance_id: int) -> None:
        for job in list(self._outstanding):
            if job.is_final or job.utterance_id > utterance_id:
                continue
            if job.cancel():
                self.partials_cancelled += 1
                self._outstanding.discard(job)
                if self._partial_job is job:
                    self._partial_job = None

    # -------------------------------------------------------------- inference

    def _transcribe(self, audio: np.ndarray, is_final: bool, utterance_id: int) -> None:
        started = time.monotonic()
        audio_ms = int(len(audio) / SAMPLE_RATE * 1000)
        kind = "final" if is_final else "partial"
        speech_end_at = self._speech_end_at.get(utterance_id) if is_final else None

        log_metric(
            f"{kind}_transcription_started",
            session_id=self._session_id,
            channel=self._source.channel.value,
            utterance_id=utterance_id,
            audio_duration_ms=audio_ms,
            queue_wait_ms=(
                elapsed_ms(speech_end_at, started) if speech_end_at is not None else None
            ),
        )

        # Interim passes re-transcribe the whole utterance snapshot. Snapshots
        # are coalesced, capped per utterance, and cancelled once the final for
        # that utterance is queued, so the wasted work is bounded.
        try:
            transcript = self._engine.transcribe(audio, is_final=is_final)
        except SttError as exc:
            self.errors += 1
            logger.warning("transcription_failed final=%s error=%s", is_final, exc)
            return
        finally:
            if not is_final:
                self._last_partial_inference_ms = (time.monotonic() - started) * 1000
                self._finish_partial(utterance_id)

        completed = time.monotonic()
        log_metric(
            f"{kind}_transcription_completed",
            session_id=self._session_id,
            channel=self._source.channel.value,
            utterance_id=utterance_id,
            audio_duration_ms=audio_ms,
            duration_ms=elapsed_ms(started, completed),
            speech_end_to_transcript_ms=(
                elapsed_ms(speech_end_at, completed) if speech_end_at is not None else None
            ),
            chars=len(transcript.text),
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
                self._speech_end_at.pop(utterance_id, None)
        self._publish(transcript.text, is_final)

    def _finish_partial(self, utterance_id: int) -> None:
        """Release the in-flight slot and promote the newest pending snapshot."""
        with self._schedule_lock:
            for job in list(self._outstanding):
                if not job.is_final and job.utterance_id == utterance_id:
                    self._outstanding.discard(job)
            self._partial_job = None
            pending = self._pending_partial
            self._pending_partial = None
            if (
                pending is not None
                and pending[0] > self._final_utterance_id
                and not self._stopping
            ):
                self._submit_partial_locked(*pending)


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
        # One warmup for the whole pipeline: every channel shares one engine.
        if started:
            started[0].warmup()

    def stop(self) -> None:
        for worker in self._workers:
            worker.stop()
