"""Scheduling and partial-transcript policy.

Every test here is gate-driven rather than sleep-driven: a job blocks on a
threading.Event until the test releases it, so ordering assertions are facts
about the queue rather than races against the clock.
"""

import asyncio
import threading

import numpy as np
import pytest

from app.audio.base import AudioChannel
from app.core.config import settings
from app.stt.pipeline import AudioPipeline, TranscriptionWorker
from app.stt.scheduler import InferenceScheduler, priority_for
from app.stt.vad import FRAME_MS
from tests.fakes import (
    FakeAudioSource,
    FakeSttEngine,
    ScriptedSpeechDetector,
    silence_frames,
    speech_frames,
)

SILENCE_TO_END = 700 // FRAME_MS + 2


@pytest.fixture
def scheduler():
    sched = InferenceScheduler(workers=1)
    sched.acquire()
    yield sched
    sched.release()


class Gate:
    """A job that parks the single scheduler thread until the test says go."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self) -> None:
        self.entered.set()
        self.release.wait(5.0)


class GatedSttEngine(FakeSttEngine):
    """Records calls and optionally blocks inside transcribe()."""

    def __init__(self, final_text: str = "How would you handle duplicate records?") -> None:
        super().__init__(final_text)
        self.gate: Gate | None = None

    def transcribe(self, audio, is_final):
        if self.gate is not None and not is_final:
            self.gate()
        return super().transcribe(audio, is_final)


# ------------------------------------------------------------------ scheduler


def test_priority_bands_follow_the_configured_order():
    # Every final must outrank every partial, regardless of channel -- a MIC
    # final must never queue behind a LOOPBACK partial -- with loopback still
    # winning within each band.
    assert priority_for(AudioChannel.LOOPBACK, True) < priority_for(AudioChannel.MIC, True)
    assert priority_for(AudioChannel.MIC, True) < priority_for(AudioChannel.LOOPBACK, False)
    assert priority_for(AudioChannel.LOOPBACK, False) < priority_for(AudioChannel.MIC, False)


def test_a_final_never_queues_behind_a_partial_of_any_channel(scheduler):
    """Regression: bands used to be per-channel (LB final, LB partial, MIC
    final, MIC partial), so a queued LOOPBACK partial could outrank a MIC
    final and delay it. Every final must now beat every partial outright."""
    gate = Gate()
    scheduler.submit(gate, channel=AudioChannel.LOOPBACK, is_final=True)
    assert gate.entered.wait(5.0)

    order: list[str] = []
    scheduler.submit(
        lambda: order.append("loopback_partial"), channel=AudioChannel.LOOPBACK, is_final=False
    )
    scheduler.submit(lambda: order.append("mic_final"), channel=AudioChannel.MIC, is_final=True)

    gate.release.set()
    scheduler._queue.join()

    assert order == ["mic_final", "loopback_partial"]


def test_loopback_final_runs_before_everything_queued_ahead_of_it(scheduler):
    gate = Gate()
    scheduler.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    order: list[str] = []
    lock = threading.Lock()

    def record(label):
        def run():
            with lock:
                order.append(label)
        return run

    # Submitted worst-first, so FIFO alone would give the reverse of the answer.
    scheduler.submit(record("mic_partial"), channel=AudioChannel.MIC, is_final=False)
    scheduler.submit(record("mic_final"), channel=AudioChannel.MIC, is_final=True)
    scheduler.submit(record("loopback_partial"), channel=AudioChannel.LOOPBACK, is_final=False)
    scheduler.submit(record("loopback_final"), channel=AudioChannel.LOOPBACK, is_final=True)

    gate.release.set()
    scheduler._queue.join()

    assert order == ["loopback_final", "mic_final", "loopback_partial", "mic_partial"]


def test_mic_backlog_does_not_delay_a_loopback_final(scheduler):
    gate = Gate()
    scheduler.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    order: list[str] = []
    for i in range(20):
        scheduler.submit(
            lambda i=i: order.append(f"mic{i}"), channel=AudioChannel.MIC, is_final=False
        )
    scheduler.submit(
        lambda: order.append("loopback_final"),
        channel=AudioChannel.LOOPBACK,
        is_final=True,
    )

    gate.release.set()
    scheduler._queue.join()

    assert order[0] == "loopback_final"


def test_a_queued_job_can_be_cancelled_before_it_starts(scheduler):
    gate = Gate()
    scheduler.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    ran = []
    job = scheduler.submit(
        lambda: ran.append(1), channel=AudioChannel.MIC, is_final=False
    )
    assert job.cancel() is True

    gate.release.set()
    scheduler._queue.join()

    assert ran == []
    assert job.cancelled


def test_a_running_job_cannot_be_cancelled(scheduler):
    gate = Gate()
    job = scheduler.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    assert job.cancel() is False
    gate.release.set()
    assert job.wait(5.0)


def test_release_drains_queued_work_before_stopping():
    sched = InferenceScheduler(workers=1)
    sched.acquire()
    gate = Gate()
    sched.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    ran = []
    sched.submit(lambda: ran.append("final"), channel=AudioChannel.LOOPBACK, is_final=True)
    gate.release.set()

    sched.release()

    assert ran == ["final"]  # a queued final is never dropped on shutdown
    assert not sched.running


def test_refcount_keeps_threads_up_until_the_last_worker_releases():
    sched = InferenceScheduler(workers=1)
    sched.acquire()
    sched.acquire()
    sched.release()
    assert sched.running
    sched.release()
    assert not sched.running


# --------------------------------------------------------------------- worker


class Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, text, source, is_final):
        self.calls.append((text, source, is_final))


def build_worker(scheduler, engine, frames, probabilities, channel, **kwargs):
    return TranscriptionWorker(
        source=FakeAudioSource(frames, channel=channel),
        detector=ScriptedSpeechDetector(probabilities),
        engine=engine,
        loop=asyncio.get_running_loop(),
        on_transcript=Recorder(),
        scheduler=scheduler,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_speech_end_cancels_a_partial_still_waiting_in_the_queue(scheduler):
    """The partial is queued behind another channel, so it is still cancellable
    when speech ends -- which is exactly when it stops being worth running."""
    gate = Gate()
    scheduler.submit(gate, channel=AudioChannel.MIC, is_final=True)
    assert gate.entered.wait(5.0)

    engine = GatedSttEngine()
    frames = speech_frames(60) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 60 + [0.0] * SILENCE_TO_END
    worker = build_worker(
        scheduler, engine, frames, probabilities, AudioChannel.LOOPBACK,
        partial_interval_ms=100,
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(
        None, worker._source.exhausted.wait, 5.0
    )
    # The audio thread has run to SPEECH_END while inference was blocked.
    assert worker.partials_scheduled >= 1
    assert worker.partials_cancelled >= 1

    gate.release.set()
    worker.stop()
    await asyncio.sleep(0.05)

    assert [is_final for _, is_final in engine.calls] == [True]


@pytest.mark.asyncio
async def test_a_partial_never_publishes_after_its_final(scheduler):
    engine = FakeSttEngine()
    worker = build_worker(
        scheduler, engine, silence_frames(1), [0.0], AudioChannel.LOOPBACK
    )
    recorder = worker._on_transcript
    worker._published_final_utterance_id = 3

    worker._transcribe(np.zeros(16_000, dtype=np.float32), False, 3)
    worker._transcribe(np.zeros(16_000, dtype=np.float32), False, 2)

    assert recorder.calls == []


@pytest.mark.asyncio
async def test_partial_work_per_utterance_is_bounded(scheduler):
    """Speech far longer than the cadence must not queue unbounded snapshots."""
    engine = FakeSttEngine()
    frames = speech_frames(300) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 300 + [0.0] * SILENCE_TO_END
    worker = build_worker(
        scheduler, engine, frames, probabilities, AudioChannel.LOOPBACK,
        partial_interval_ms=32,
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(
        None, worker._source.exhausted.wait, 5.0
    )
    worker.stop()
    await asyncio.sleep(0.05)

    assert worker.partials_scheduled <= settings.stt_max_partials_per_utterance
    partial_calls = [c for c in engine.calls if not c[1]]
    assert len(partial_calls) <= settings.stt_max_partials_per_utterance


@pytest.mark.asyncio
async def test_partials_are_skipped_below_the_minimum_audio_floor(scheduler):
    """Under the audio floor a partial says nothing the final will not say.

    The VAD keeps buffering through its trailing-silence hangover before it
    confirms SPEECH_END, so a short burst of speech still ends up with a long
    buffer eventually -- the floor's job is to skip the *earliest* snapshots,
    not to suppress partials for the whole utterance.
    """
    frames = speech_frames(5) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 5 + [0.0] * SILENCE_TO_END
    engine = FakeSttEngine()
    worker = build_worker(
        scheduler, engine, frames, probabilities, AudioChannel.LOOPBACK,
        partial_interval_ms=32,
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(
        None, worker._source.exhausted.wait, 5.0
    )
    worker.stop()
    await asyncio.sleep(0.05)

    assert worker.partials_skipped > 0
    for samples, is_final in engine.calls:
        if not is_final:
            assert samples / 16_000 * 1000 >= settings.stt_partial_min_audio_ms


@pytest.mark.asyncio
async def test_partials_can_be_disabled_entirely(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "stt_enable_partials", False)
    engine = FakeSttEngine()
    frames = speech_frames(120) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 120 + [0.0] * SILENCE_TO_END
    worker = build_worker(
        scheduler, engine, frames, probabilities, AudioChannel.LOOPBACK,
        partial_interval_ms=32,
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(
        None, worker._source.exhausted.wait, 5.0
    )
    worker.stop()
    await asyncio.sleep(0.05)

    assert worker.partials_scheduled == 0
    assert [is_final for _, is_final in engine.calls] == [True]


@pytest.mark.asyncio
async def test_pipeline_shutdown_leaves_no_inference_threads():
    before = {t.name for t in threading.enumerate()}
    workers = [
        TranscriptionWorker(
            FakeAudioSource(
                speech_frames(20) + silence_frames(SILENCE_TO_END), channel=channel
            ),
            ScriptedSpeechDetector([0.9] * 20 + [0.0] * SILENCE_TO_END),
            FakeSttEngine(),
            asyncio.get_running_loop(),
            Recorder(),
        )
        for channel in (AudioChannel.LOOPBACK, AudioChannel.MIC)
    ]
    pipeline = AudioPipeline(workers)
    pipeline.start()
    await asyncio.sleep(0.2)
    pipeline.stop()
    await asyncio.sleep(0.05)

    leaked = {
        t.name for t in threading.enumerate()
        if t.is_alive() and (t.name.startswith("stt-infer") or t.name.startswith("stt-"))
    } - before
    assert leaked == set()


@pytest.mark.asyncio
async def test_pipeline_start_warms_the_engine_once():
    class CountingEngine(FakeSttEngine):
        def __init__(self):
            super().__init__()
            self.warmups = 0

        def warmup(self):
            self.warmups += 1

    engine = CountingEngine()
    workers = [
        TranscriptionWorker(
            FakeAudioSource(silence_frames(5), channel=channel),
            ScriptedSpeechDetector([0.0] * 5),
            engine,
            asyncio.get_running_loop(),
            Recorder(),
        )
        for channel in (AudioChannel.LOOPBACK, AudioChannel.MIC)
    ]
    pipeline = AudioPipeline(workers)
    pipeline.start()
    await asyncio.sleep(0.2)
    pipeline.stop()

    # One shared model, so one warmup -- not one per channel.
    assert engine.warmups == 1
