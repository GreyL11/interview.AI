"""Process-wide priority scheduler for Whisper inference.

Why this exists rather than one ThreadPoolExecutor per channel: every channel
shares a single CTranslate2 model (`get_stt_engine` is lru_cached), and
CTranslate2 serialises concurrent calls internally across `num_workers` slots.
Two per-channel executors therefore do not run in parallel — they queue inside
the C++ layer in arrival order, where no Python-side priority can reach them.
Funnelling every job through one queue makes that ordering ours to choose.

Ordering is (priority, submission sequence): strict priority, FIFO within a
band. The default bands put the interviewer's final transcript first, because
that is the only job on the critical path to an answer.
"""

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.audio.base import AudioChannel
from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import log_metric

logger = get_logger(__name__)

#: Beats every real job. Loading the model is a prerequisite for all of them.
WARMUP_PRIORITY = -1
#: Loses to every real job, so shutdown drains queued work first.
_SENTINEL_PRIORITY = 1 << 30

#: Above this depth the queue is no longer keeping up, and we say so once.
_BACKLOG_WARN_DEPTH = 6


def priority_for(channel: AudioChannel, is_final: bool) -> int:
    """Where a job sits in the queue.

    Configurable because the right answer depends on how the machine is used:
    the default assumes the interviewer is on loopback and the candidate on the
    microphone, which is the product's normal shape but not a law.
    """
    if channel == AudioChannel.LOOPBACK:
        return (
            settings.stt_priority_loopback_final
            if is_final
            else settings.stt_priority_loopback_partial
        )
    return (
        settings.stt_priority_mic_final if is_final else settings.stt_priority_mic_partial
    )


class InferenceJob:
    """A single queued transcription.

    `cancel()` only succeeds before a worker picks the job up; CTranslate2
    inference is not interruptible, so a running job always runs to completion.
    """

    __slots__ = (
        "channel",
        "utterance_id",
        "is_final",
        "priority",
        "enqueued_at",
        "_fn",
        "_lock",
        "_state",
        "_done",
    )

    def __init__(
        self,
        fn: Callable[[], Any],
        channel: AudioChannel,
        utterance_id: int,
        is_final: bool,
        priority: int,
    ) -> None:
        self._fn = fn
        self.channel = channel
        self.utterance_id = utterance_id
        self.is_final = is_final
        self.priority = priority
        #: When this job joined the queue. Used to decide, at pickup time,
        #: whether transcribing it is still worth the CPU -- see `expired`.
        self.enqueued_at = time.monotonic()
        self._lock = threading.Lock()
        self._state = "queued"  # queued -> running -> finished, or queued -> cancelled
        self._done = threading.Event()

    def age_seconds(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.enqueued_at

    def expired(self, now: float | None = None) -> bool:
        """Has this job been waiting so long that its output is worthless?

        A transcript is a real-time product. Once it is this far behind, the
        conversation has moved on: the words are no longer on screen when they
        would matter, and for the interviewer channel the question they carry
        has already been superseded. Running it anyway costs the CPU that the
        *next* utterance needs, which is how a queue that is merely late turns
        into one that never recovers.

        Deadlines are per-role because the roles differ. The microphone is
        transcribed for review only and nothing downstream waits on it, so it
        yields first. Loopback drives question detection and is given
        substantially longer before it is given up on.
        """
        budget = (
            settings.stt_max_job_age_seconds
            if self.channel == AudioChannel.LOOPBACK
            else settings.stt_max_mic_job_age_seconds
        )
        if budget <= 0:
            return False  # shedding disabled
        return self.age_seconds(now) > budget

    def drop(self) -> bool:
        """Discard a job that is still queued. True if it will now never run."""
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "dropped"
        self._done.set()
        return True

    @property
    def cancelled(self) -> bool:
        return self._state == "cancelled"

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> bool:
        """True if the job was still queued and will now never run."""
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "cancelled"
        self._done.set()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def claim(self) -> bool:
        """Move queued -> running. False means it was cancelled while waiting."""
        with self._lock:
            if self._state != "queued":
                return False
            self._state = "running"
        return True

    def run(self) -> None:
        try:
            self._fn()
        except Exception:
            logger.exception(
                "inference_job_crashed channel=%s utterance=%d final=%s",
                self.channel.value, self.utterance_id, self.is_final,
            )
        finally:
            self._state = "finished"
            self._done.set()


@dataclass(order=True)
class _Queued:
    priority: int
    sequence: int
    job: InferenceJob | None = field(compare=False, default=None)


class InferenceScheduler:
    """Priority-ordered thread pool shared by every transcription worker.

    Reference counted: the first worker to start it brings the threads up, the
    last to release it takes them down. Thread lifetime is tied to actual
    capture rather than to process lifetime, which keeps leaks visible to
    `threading.enumerate()` in tests.
    """

    def __init__(self, workers: int | None = None) -> None:
        self._worker_count = max(
            1, workers if workers is not None else settings.stt_inference_concurrency
        )
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._refcount = 0
        self._running = False
        self._warned_backlog = False
        #: Jobs shed because they aged out. Surfaced in the worker's
        #: end-of-session metrics line so a session that silently lost
        #: transcript is still auditable afterwards.
        self.dropped_jobs = 0
        self._warned_dropping = False

    # ------------------------------------------------------------- lifecycle

    def acquire(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._running:
                return
            self._running = True
            self._warned_backlog = False
            self._threads = [
                threading.Thread(target=self._work, name=f"stt-infer-{i}", daemon=True)
                for i in range(self._worker_count)
            ]
            threads = list(self._threads)
        for thread in threads:
            thread.start()
        logger.info("inference_scheduler_started workers=%d", self._worker_count)

    def release(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._refcount == 0:
                return
            self._refcount -= 1
            if self._refcount > 0 or not self._running:
                return
            self._running = False
            threads, self._threads = self._threads, []

        # Sentinels sort last, so queued work — a final in particular — still runs.
        for _ in threads:
            self._put(_SENTINEL_PRIORITY, None)
        for thread in threads:
            thread.join(timeout=timeout)
        logger.info("inference_scheduler_stopped")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    # ---------------------------------------------------------------- submit

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        channel: AudioChannel,
        utterance_id: int = 0,
        is_final: bool = False,
        priority: int | None = None,
    ) -> InferenceJob | None:
        """Queue a job. Returns None if the scheduler is not running."""
        if not self._running:
            return None
        job_priority = priority if priority is not None else priority_for(channel, is_final)
        job = InferenceJob(fn, channel, utterance_id, is_final, job_priority)
        self._put(job_priority, job)
        log_metric(
            "stt_job_enqueued",
            channel=channel.value,
            utterance_id=utterance_id,
            is_final=is_final,
            priority=job_priority,
            queue_depth=self.depth,
        )
        log_metric(
            "stt_job_priority",
            channel=channel.value,
            utterance_id=utterance_id,
            is_final=is_final,
            priority=job_priority,
        )
        log_metric(
            "stt_queue_depth",
            channel=channel.value,
            utterance_id=utterance_id,
            depth=self.depth,
        )
        return job

    def _put(self, priority: int, job: InferenceJob | None) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._queue.put(_Queued(priority, sequence, job))
        self._check_backlog()

    def _note_drop(self, job: InferenceJob) -> None:
        """Say it once, loudly, rather than once per dropped job.

        Shedding is the system working as designed under overload, but it means
        the user is losing transcript -- so it must never be silent. A line per
        drop would bury the log at exactly the moment it is being read.
        """
        self.dropped_jobs += 1
        if job.channel == AudioChannel.LOOPBACK and not self._warned_dropping:
            self._warned_dropping = True
            logger.warning(
                "stt_shedding_load: transcription is too far behind, so "
                "interviewer audio is being dropped rather than answered late. "
                "Turn off microphone capture, choose a smaller STT_MODEL, or "
                "set STT_CPU_THREADS."
            )

    def _check_backlog(self) -> None:
        # The queue is bounded in practice — each channel holds at most one
        # in-flight and one pending partial — so depth here means finals are
        # arriving faster than the model can clear them.
        depth = self._queue.qsize()
        if depth > _BACKLOG_WARN_DEPTH and not self._warned_backlog:
            self._warned_backlog = True
            logger.warning(
                "inference_backlog depth=%d; STT cannot keep up with speech. "
                "Consider a smaller STT_MODEL or STT_ENABLE_PARTIALS=false.",
                depth,
            )
        elif depth <= 1:
            self._warned_backlog = False

    # ----------------------------------------------------------------- drain

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item.job is None:
                    return
                # Checked at pickup, not at submit: a job is only worthless
                # once it has actually sat in the queue, and that is exactly
                # the moment this loop finds out.
                if item.job.expired() and item.job.drop():
                    log_metric(
                        "stt_job_dropped",
                        channel=item.job.channel.value,
                        utterance_id=item.job.utterance_id,
                        is_final=item.job.is_final,
                        reason="too_stale",
                        age_ms=int(item.job.age_seconds() * 1000),
                        queue_depth=self.depth,
                    )
                    self._note_drop(item.job)
                    continue
                if not item.job.claim():
                    continue  # cancelled while it waited
                log_metric(
                    "stt_job_started",
                    channel=item.job.channel.value,
                    utterance_id=item.job.utterance_id,
                    is_final=item.job.is_final,
                    priority=item.job.priority,
                )
                item.job.run()
                log_metric(
                    "stt_job_completed",
                    channel=item.job.channel.value,
                    utterance_id=item.job.utterance_id,
                    is_final=item.job.is_final,
                    priority=item.job.priority,
                )
            finally:
                self._queue.task_done()


_shared: InferenceScheduler | None = None
_shared_lock = threading.Lock()


def shared_scheduler() -> InferenceScheduler:
    """The one scheduler every channel funnels through.

    A module-level singleton because the constraint it models — one Whisper
    model, one machine's worth of CPU — is itself process-wide. Tests construct
    their own instance and inject it instead.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = InferenceScheduler()
        return _shared


def reset_shared_scheduler() -> None:
    """Drop the singleton so a concurrency settings change can take effect."""
    global _shared
    with _shared_lock:
        _shared = None
