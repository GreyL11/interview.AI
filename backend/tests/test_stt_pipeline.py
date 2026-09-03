import asyncio
import logging
import threading
import time

import numpy as np
import pytest

from app.audio.base import AudioChannel, AudioError
from app.sessions.schemas import TranscriptSource
from app.stt.base import SttError, Transcript
from app.stt.pipeline import AudioPipeline, TranscriptionWorker
from app.stt.scheduler import InferenceScheduler
from app.stt.vad import FRAME_MS
from tests.fakes import (
    FakeAudioSource,
    FakeSttEngine,
    ScriptedSpeechDetector,
    silence_frames,
    speech_frames,
)

pytestmark = pytest.mark.asyncio

SILENCE_TO_END = 700 // FRAME_MS + 2


class Recorder:
    def __init__(self):
        self.calls = []
        #: (text, source, is_final, trace, now, received_at) for every call --
        #: kept separately from `calls` so the many existing 3-tuple
        #: assertions don't all need updating for the fields they don't care
        #: about. `received_at` is wall-clock time.monotonic() at delivery,
        #: for comparing against the speech-timeline `now` the pipeline sent.
        self.raw_calls = []

    async def __call__(self, text, source, is_final, trace=None, now=None):
        self.calls.append((text, source, is_final))
        self.raw_calls.append((text, source, is_final, trace, now, time.monotonic()))

    def finals(self):
        return [c for c in self.calls if c[2]]

    def partials(self):
        return [c for c in self.calls if not c[2]]


async def run_worker(
    probabilities,
    frames,
    engine=None,
    partial_interval_ms=1000,
    channel=AudioChannel.LOOPBACK,
    on_transcript=None,
):
    """Drive a worker to completion over a fixed frame script.

    `on_transcript` overrides the default Recorder, for tests that need the
    publish side to misbehave.
    """
    recorder = Recorder()
    source = FakeAudioSource(frames, channel=channel)
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector(probabilities),
        engine=engine or FakeSttEngine(),
        loop=asyncio.get_running_loop(),
        on_transcript=on_transcript or recorder,
        partial_interval_ms=partial_interval_ms,
    )
    worker.start()

    await asyncio.get_running_loop().run_in_executor(None, source.exhausted.wait, 5.0)
    await asyncio.sleep(0.2)  # let queued coroutines land on the loop
    worker.stop()
    await asyncio.sleep(0.05)  # let executor-published coroutines land on the loop
    return recorder, worker


async def test_utterance_produces_a_final_transcript():
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    recorder, _ = await run_worker(probabilities, frames)

    finals = recorder.finals()
    assert len(finals) == 1
    assert finals[0][0] == "How would you handle duplicate records?"
    assert finals[0][1] == TranscriptSource.LOOPBACK


async def test_partials_are_emitted_during_a_long_utterance():
    # 80 frames is ~2.5s of audio, so a 500ms cadence should fire several times.
    frames = speech_frames(80) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 80 + [0.0] * SILENCE_TO_END

    recorder, _ = await run_worker(probabilities, frames, partial_interval_ms=500)

    assert recorder.partials()
    assert len(recorder.finals()) == 1
    # The final must be the last thing the session sees for the utterance.
    assert recorder.calls[-1][2] is True


async def test_silence_produces_nothing():
    frames = silence_frames(40)
    recorder, _ = await run_worker([0.0] * 40, frames)
    assert recorder.calls == []


async def test_mic_channel_is_tagged_as_mic():
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    recorder, _ = await run_worker(probabilities, frames, channel=AudioChannel.MIC)

    assert recorder.finals()[0][1] == TranscriptSource.MIC


async def test_two_utterances_produce_two_finals():
    burst = speech_frames(15)
    gap = silence_frames(SILENCE_TO_END)
    frames = burst + gap + burst + gap
    probabilities = [0.9] * 15 + [0.0] * SILENCE_TO_END + [0.9] * 15 + [0.0] * SILENCE_TO_END

    recorder, _ = await run_worker(probabilities, frames)
    assert len(recorder.finals()) == 2


async def test_empty_transcript_is_not_published():
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    recorder, _ = await run_worker(probabilities, frames, engine=FakeSttEngine(final_text="   "))
    assert recorder.calls == []


async def test_stt_failure_is_survivable():
    class BrokenEngine(FakeSttEngine):
        def transcribe(self, audio, is_final):
            raise SttError("model exploded")

    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    recorder, worker = await run_worker(probabilities, frames, engine=BrokenEngine())

    assert recorder.calls == []
    assert worker.errors > 0  # counted, not crashed


async def test_final_pass_gets_the_whole_utterance():
    engine = FakeSttEngine()
    frames = speech_frames(30) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 30 + [0.0] * SILENCE_TO_END

    await run_worker(probabilities, frames, engine=engine, partial_interval_ms=200)

    final_calls = [c for c in engine.calls if c[1]]
    partial_calls = [c for c in engine.calls if not c[1]]
    assert final_calls
    # If an interim pass completed before the final was scheduled, the final
    # must contain at least as much audio.
    if partial_calls:
        assert final_calls[-1][0] >= max(c[0] for c in partial_calls)


async def test_pipeline_starts_and_stops_all_workers():
    loop = asyncio.get_running_loop()
    recorder = Recorder()
    workers = [
        TranscriptionWorker(
            FakeAudioSource(silence_frames(5), channel=channel),
            ScriptedSpeechDetector([0.0] * 5),
            FakeSttEngine(),
            loop,
            recorder,
        )
        for channel in (AudioChannel.MIC, AudioChannel.LOOPBACK)
    ]
    pipeline = AudioPipeline(workers)

    assert pipeline.channels == [TranscriptSource.MIC, TranscriptSource.LOOPBACK]
    pipeline.start()
    await asyncio.sleep(0.1)
    pipeline.stop()


async def test_pipeline_rolls_back_when_a_worker_fails_to_start():
    loop = asyncio.get_running_loop()

    good = TranscriptionWorker(
        FakeAudioSource(silence_frames(5)), ScriptedSpeechDetector([0.0]),
        FakeSttEngine(), loop, Recorder(),
    )

    class ExplodingSource(FakeAudioSource):
        def start(self):
            raise RuntimeError("device busy")

    bad = TranscriptionWorker(
        ExplodingSource(silence_frames(5)), ScriptedSpeechDetector([0.0]),
        FakeSttEngine(), loop, Recorder(),
    )

    with pytest.raises(RuntimeError):
        AudioPipeline([good, bad]).start()

    # The already-running worker must not be left orphaned holding a device.
    assert good._thread is None or not good._thread.is_alive()


class SlowSttEngine(FakeSttEngine):
    def __init__(self, delay: float = 0.05):
        super().__init__()
        self.delay = delay

    def transcribe(self, audio, is_final):
        time.sleep(self.delay)
        return super().transcribe(audio, is_final)


async def test_slow_stt_does_not_block_audio_frame_consumption():
    frames = speech_frames(100) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 100 + [0.0] * SILENCE_TO_END

    _, worker = await run_worker(
        probabilities,
        frames,
        engine=SlowSttEngine(delay=0.1),
        partial_interval_ms=200,
    )

    assert worker.frames_consumed == len(frames)


class ExplodingRecorder:
    """An on_transcript that fails on the event loop, the way a SQLite write
    or a dead subscriber would."""

    def __init__(self, exc=None):
        self.calls = 0
        self.exc = exc or RuntimeError("on_transcript exploded")

    async def __call__(self, text, source, is_final, trace=None, now=None):
        self.calls += 1
        raise self.exc


async def test_a_failure_inside_on_transcript_leaves_evidence(caplog):
    """`run_coroutine_threadsafe` returns a concurrent.futures.Future, which
    never reports an unretrieved exception the way asyncio.Task does. Without
    observing it, a question could be lost with no log, no event and no
    counter -- which is exactly how it used to behave."""
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END
    exploding = ExplodingRecorder()

    with caplog.at_level(logging.ERROR, logger="app.stt.pipeline"):
        _, worker = await run_worker(probabilities, frames, on_transcript=exploding)

    # Both the interim and the final publish are offered and both fail here;
    # asserting the relationship rather than a fixed count keeps this test
    # independent of partial cadence.
    assert exploding.calls >= 1, "the transcript should still have been offered"
    assert worker.publish_failures == exploding.calls, (
        f"{exploding.calls} publishes failed but "
        f"{worker.publish_failures} were counted"
    )
    assert "transcript_publish_failed" in caplog.text, "no evidence logged"
    assert "on_transcript exploded" in caplog.text, "traceback not preserved"
    # The inference-side counter must not absorb a delivery-side failure.
    assert worker.errors == 0


async def test_a_successful_publish_records_no_failure():
    """Guard against the observer itself becoming a false-positive source."""
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    recorder, worker = await run_worker(probabilities, frames)

    assert recorder.finals()
    assert worker.publish_failures == 0
    assert worker.errors == 0


async def test_publishing_to_a_closed_loop_is_counted_not_swallowed():
    """A final that lands after the session tore down. The transcript is
    genuinely lost, so it must not look like a successful publish -- and it
    must not raise into the inference worker either."""
    dead_loop = asyncio.new_event_loop()
    dead_loop.close()

    worker = TranscriptionWorker(
        source=FakeAudioSource([], channel=AudioChannel.LOOPBACK),
        detector=ScriptedSpeechDetector([]),
        engine=FakeSttEngine(),
        loop=dead_loop,
        on_transcript=Recorder(),
    )

    worker._publish("a lost question", is_final=True)  # must not raise

    assert worker.publish_failures == 1


async def test_final_publishes_with_speech_end_time_not_delivery_time():
    """Turn/question assembly has to key off when the interviewer stopped
    talking, not when a backlogged Whisper happened to get around to it --
    otherwise queue delay masquerades as a real pause between utterances (or
    swallows one). `now` on the final call must be speech-end time, which
    predates delivery by roughly the injected STT delay, not the delivery
    moment itself."""
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END
    delay = 0.2

    recorder, _ = await run_worker(
        probabilities, frames, engine=SlowSttEngine(delay=delay), partial_interval_ms=1000,
    )

    finals = [c for c in recorder.raw_calls if c[2]]
    assert len(finals) == 1
    _, _, _, trace, now, received_at = finals[0]
    assert trace is not None
    assert now == trace.speech_end_at
    # Loose bound (half the injected delay) so this isn't flaky on a slow CI
    # box, while still failing hard if `now` regressed to time-of-delivery
    # (in which case the gap would be ~0, not ~`delay`).
    assert received_at - now >= delay / 2


async def test_slow_partials_are_coalesced_and_final_is_executed():
    frames = speech_frames(100) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 100 + [0.0] * SILENCE_TO_END
    engine = SlowSttEngine(delay=0.1)

    recorder, worker = await run_worker(
        probabilities,
        frames,
        engine=engine,
        partial_interval_ms=100,
    )

    assert worker.partials_scheduled >= 1
    assert worker.partials_coalesced > 0
    assert any(is_final for _, is_final in engine.calls)
    assert recorder.calls[-1][2] is True


async def test_stt_queue_metrics_are_emitted(monkeypatch):
    events: list[tuple[str, dict]] = []

    def record(event, **fields):
        events.append((event, fields))

    monkeypatch.setattr("app.stt.scheduler.log_metric", record)
    monkeypatch.setattr("app.stt.pipeline.log_metric", record)

    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    await run_worker(probabilities, frames, partial_interval_ms=200)

    event_names = [name for name, _ in events]
    assert "stt_job_enqueued" in event_names
    assert "stt_job_started" in event_names
    assert "stt_job_completed" in event_names
    assert "stt_job_priority" in event_names
    assert "stt_queue_depth" in event_names


async def test_stop_releases_the_inference_scheduler():
    frames = speech_frames(20) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 20 + [0.0] * SILENCE_TO_END

    _, worker = await run_worker(probabilities, frames, engine=SlowSttEngine())

    assert not worker._scheduler.running
    assert not any(t.name.startswith("stt-infer") for t in threading.enumerate())


class BlockingWarmupEngine(FakeSttEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def warmup(self):
        self.entered.set()
        self.release.wait(5.0)


class ExplodingWarmupEngine(FakeSttEngine):
    def warmup(self):
        raise SttError("warmup failed")


class RecordingAudioSource(FakeAudioSource):
    def __init__(self, frames, channel=AudioChannel.LOOPBACK):
        super().__init__(frames, channel=channel)
        self.started = threading.Event()

    def start(self):
        self.started.set()
        super().start()


async def test_pipeline_waits_for_warmup_before_starting_audio():
    loop = asyncio.get_running_loop()
    engine = BlockingWarmupEngine()
    source = RecordingAudioSource(silence_frames(5))
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector([0.0] * 5),
        engine=engine,
        loop=loop,
        on_transcript=Recorder(),
    )
    pipeline = AudioPipeline([worker])

    start_task = asyncio.create_task(asyncio.to_thread(pipeline.start))
    await asyncio.get_running_loop().run_in_executor(None, engine.entered.wait, 5.0)

    assert not source.started.is_set()

    engine.release.set()
    await start_task

    assert source.started.is_set()
    pipeline.stop()


async def test_pipeline_cleans_up_when_warmup_fails():
    loop = asyncio.get_running_loop()
    source = RecordingAudioSource(silence_frames(5))
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector([0.0] * 5),
        engine=ExplodingWarmupEngine(),
        loop=loop,
        on_transcript=Recorder(),
    )
    pipeline = AudioPipeline([worker])

    with pytest.raises(SttError, match="warmup failed"):
        await asyncio.to_thread(pipeline.start)

    assert not source.started.is_set()
    assert not worker._scheduler.running


# ---------------------------------------------------- partial/final divergence


class DivergingSttEngine(FakeSttEngine):
    """A partial that looks nothing like the eventual final -- simulates an
    STT engine guessing wrong on interim, unstable audio."""

    def transcribe(self, audio, is_final):
        self.calls.append((len(audio), is_final))
        text = self.final_text if is_final else "completely unrelated garbled words"
        return Transcript(text=text, is_final=is_final, duration_ms=len(audio) // 16)


async def test_final_diverging_from_its_partial_is_logged(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.stt.pipeline.log_metric",
        lambda event, **f: events.append((event, f)),
    )

    frames = speech_frames(30) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 30 + [0.0] * SILENCE_TO_END

    await run_worker(probabilities, frames, engine=DivergingSttEngine(), partial_interval_ms=200)

    diverged = [f for e, f in events if e == "stt_final_diverges_from_partial"]
    assert diverged
    assert diverged[0]["overlap"] < 0.4


async def test_a_final_matching_its_partial_is_not_flagged(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.stt.pipeline.log_metric",
        lambda event, **f: events.append((event, f)),
    )

    frames = speech_frames(30) + silence_frames(SILENCE_TO_END)
    probabilities = [0.9] * 30 + [0.0] * SILENCE_TO_END

    # FakeSttEngine's partial is a literal prefix of its final -- high overlap.
    await run_worker(probabilities, frames, engine=FakeSttEngine(), partial_interval_ms=200)

    assert not [f for e, f in events if e == "stt_final_diverges_from_partial"]


# --------------------------------------------- partial capture-device failure


class UnopenableAudioSource(RecordingAudioSource):
    """Enumerates fine, fails when the stream is actually opened -- a device
    that is disabled in the OS or held by another process. Reproduced on real
    hardware: a laptop whose mic raised PaErrorCode -9996 at open time."""

    def start(self):
        raise AudioError("Error opening InputStream: Invalid device")


async def test_a_channel_that_fails_to_open_does_not_take_down_the_others():
    """Loopback is the channel that hears the interviewer and drives question
    detection. A broken mic must cost the mic only -- audio_binding already
    applies this policy at describe() time, and open() must match it."""
    loop = asyncio.get_running_loop()
    loopback = RecordingAudioSource(silence_frames(5), channel=AudioChannel.LOOPBACK)
    mic = UnopenableAudioSource(silence_frames(5), channel=AudioChannel.MIC)

    def worker_for(source):
        return TranscriptionWorker(
            source=source,
            detector=ScriptedSpeechDetector([0.0] * 5),
            engine=FakeSttEngine(),
            loop=loop,
            on_transcript=Recorder(),
        )

    pipeline = AudioPipeline([worker_for(loopback), worker_for(mic)])
    await asyncio.to_thread(pipeline.start)

    assert loopback.started.is_set()
    assert pipeline.channels == [TranscriptSource.LOOPBACK]
    pipeline.stop()


async def test_every_channel_failing_to_open_is_still_an_error():
    loop = asyncio.get_running_loop()
    mic = UnopenableAudioSource(silence_frames(5), channel=AudioChannel.MIC)
    worker = TranscriptionWorker(
        source=mic,
        detector=ScriptedSpeechDetector([0.0] * 5),
        engine=FakeSttEngine(),
        loop=loop,
        on_transcript=Recorder(),
    )
    pipeline = AudioPipeline([worker])

    with pytest.raises(AudioError, match="No capture device could be opened"):
        await asyncio.to_thread(pipeline.start)


# ------------------------------------ stale-partial handling (measured issue)
#
# The scheduler already cancels *queued* partials when a final is scheduled
# (_cancel_stale_partials_locked) and suppresses stale partial *results* at
# emission. What it cannot do is preempt a partial that is already running --
# CTranslate2 inference is not interruptible. The measured 1608ms of final
# queue wait came from exactly that case, so these tests pin the one lever that
# is actually available: not starting a doomed partial in the first place.


def _worker(source, engine, detector, loop):
    """A worker with its OWN scheduler.

    Not the shared singleton: acquire/release on it is refcounted and its
    shutdown sentinels live in a process-wide queue, so a test that leaves it
    even slightly unbalanced silently starves a *later* test of inference
    workers. That was reproducible here as an intermittent failure in
    test_partials_are_emitted_during_a_long_utterance. The scheduler's own
    docstring calls for tests to inject an instance; this does.
    """
    return TranscriptionWorker(
        source=source,
        detector=detector,
        engine=engine,
        loop=loop,
        on_transcript=Recorder(),
        scheduler=InferenceScheduler(workers=1),
    )


async def test_final_cancels_a_queued_partial_without_running_inference():
    """FINAL arrives while a PARTIAL is still queued: the partial must be
    skipped before Whisper is ever called, and the final must still run."""
    loop = asyncio.get_running_loop()
    engine = FakeSttEngine()
    worker = _worker(RecordingAudioSource(silence_frames(1)), engine,
                     ScriptedSpeechDetector([0.0]), loop)
    scheduler = worker._scheduler
    scheduler.acquire()
    blocker = threading.Event()
    try:
        # Occupy the single inference worker so the partial provably stays
        # QUEUED -- otherwise this races the pool and tests nothing.
        scheduler.submit(lambda: blocker.wait(5), channel=AudioChannel.LOOPBACK)
        audio = np.concatenate(speech_frames(20))
        worker._submit_partial_locked(1, audio)
        queued = worker._partial_job
        assert queued is not None
        worker._cancel_stale_partials_locked(1)

        assert queued.cancelled
        assert worker.partials_cancelled == 1
        # The decisive assertion: no inference was performed for the partial.
        assert engine.calls == []
    finally:
        blocker.set()
        scheduler.release()


async def test_a_partial_is_not_started_when_the_final_is_provably_closer():
    """The one real lever for the *running* case: don't start the partial.

    Reproduces the measured trace -- a partial scheduled part-way into the
    trailing silence run, whose measured cost exceeds the remaining silence
    budget, so it could only ever delay the final."""
    loop = asyncio.get_running_loop()
    engine = FakeSttEngine()
    worker = _worker(RecordingAudioSource(silence_frames(1)), engine,
                     ScriptedSpeechDetector([0.0]), loop)

    # Speech is open and 224ms into its trailing silence run (7 frames).
    worker._segmenter.in_speech = True
    worker._segmenter._silence_run = 7
    # The last partial measurably took longer than the remaining silence
    # budget (700 - 224 = 476ms), so this one cannot finish first.
    worker._last_partial_inference_ms = 2092.0

    assert worker._final_beats_partial() is True
    worker._schedule_partial(1, speech_frames(40), buffered_ms=1280)

    assert worker.partials_skipped_near_final == 1
    assert worker.partials_scheduled == 0
    assert engine.calls == []


async def test_a_partial_still_runs_while_speech_is_active():
    """The guard must be inert during real speech, or live transcript
    responsiveness is lost for no latency gain."""
    loop = asyncio.get_running_loop()
    engine = FakeSttEngine()
    worker = _worker(RecordingAudioSource(silence_frames(1)), engine,
                     ScriptedSpeechDetector([0.0]), loop)
    scheduler = worker._scheduler
    scheduler.acquire()
    try:
        worker._segmenter.in_speech = True
        worker._segmenter._silence_run = 0  # still talking
        worker._last_partial_inference_ms = 2092.0

        assert worker._final_beats_partial() is False
        worker._schedule_partial(1, speech_frames(40), buffered_ms=1280)

        assert worker.partials_skipped_near_final == 0
        assert worker.partials_scheduled == 1
    finally:
        scheduler.release()


@pytest.mark.parametrize("measured", [0.0, 1.0, FRAME_MS - 1])
async def test_a_partial_cheaper_than_one_frame_is_never_skipped(measured):
    """Two cases in one predicate. Nothing measured yet: a cold worker must
    still produce its first partial, or it never learns a cost. Measured but
    trivial (a fast model, or a fake engine in tests): it cannot meaningfully
    delay the final, so suppressing it would cost live-transcript text for no
    latency gain."""
    loop = asyncio.get_running_loop()
    worker = _worker(RecordingAudioSource(silence_frames(1)), FakeSttEngine(),
                     ScriptedSpeechDetector([0.0]), loop)
    worker._segmenter.in_speech = True
    worker._segmenter._silence_run = 21  # deep into the silence run
    worker._last_partial_inference_ms = measured

    assert worker._final_beats_partial() is False


async def test_a_cheap_partial_still_runs_late_in_the_silence_run():
    """The guard is a comparison against measured cost, not a blanket ban on
    partials during silence: a fast model should still get its partial."""
    loop = asyncio.get_running_loop()
    worker = _worker(RecordingAudioSource(silence_frames(1)), FakeSttEngine(),
                     ScriptedSpeechDetector([0.0]), loop)
    worker._segmenter.in_speech = True
    worker._segmenter._silence_run = 7          # 224ms in, 476ms of budget left
    worker._last_partial_inference_ms = 120.0   # comfortably finishes first

    assert worker._final_beats_partial() is False


async def test_a_final_is_never_skipped_by_partial_invalidation():
    """Invalidation must only ever touch partials."""
    loop = asyncio.get_running_loop()
    engine = FakeSttEngine()
    worker = _worker(RecordingAudioSource(silence_frames(1)), engine,
                     ScriptedSpeechDetector([0.0]), loop)
    scheduler = worker._scheduler
    scheduler.acquire()
    try:
        worker._schedule_final(1, speech_frames(20))
        final = next(j for j in worker._outstanding if j.is_final)
        # A later final for the same utterance must not cancel the first.
        worker._cancel_stale_partials_locked(1)
        assert not final.cancelled
        final.wait(timeout=5)
        assert any(is_final for _, is_final in engine.calls)
    finally:
        scheduler.release()


async def test_a_partial_for_a_newer_utterance_is_not_cancelled():
    """Utterance ids are the generation token; a final for utterance 1 must
    leave utterance 2's partial alone."""
    loop = asyncio.get_running_loop()
    worker = _worker(RecordingAudioSource(silence_frames(1)), FakeSttEngine(),
                     ScriptedSpeechDetector([0.0]), loop)
    scheduler = worker._scheduler
    scheduler.acquire()
    try:
        worker._submit_partial_locked(2, np.concatenate(speech_frames(20)))
        newer = worker._partial_job
        worker._cancel_stale_partials_locked(1)
        assert newer is not None and not newer.cancelled
        newer.cancel()
    finally:
        scheduler.release()


async def test_a_partial_result_arriving_after_its_final_is_suppressed():
    """PARTIAL running when FINAL lands: no preemption is claimed, but the
    stale text must never reach the UI."""
    loop = asyncio.get_running_loop()
    recorder = Recorder()
    worker = TranscriptionWorker(
        source=RecordingAudioSource(silence_frames(1)),
        detector=ScriptedSpeechDetector([0.0]),
        engine=FakeSttEngine(),
        loop=loop,
        on_transcript=recorder,
    )
    worker._published_final_utterance_id = 1  # final already published

    await asyncio.to_thread(worker._transcribe, np.concatenate(speech_frames(20)), False, 1)
    await asyncio.sleep(0.05)

    assert recorder.calls == []
