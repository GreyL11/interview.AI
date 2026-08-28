import asyncio
import threading
import time

import numpy as np
import pytest

from app.audio.base import AudioChannel
from app.sessions.schemas import TranscriptSource
from app.stt.base import SttError
from app.stt.pipeline import AudioPipeline, TranscriptionWorker
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

    async def __call__(self, text, source, is_final):
        self.calls.append((text, source, is_final))

    def finals(self):
        return [c for c in self.calls if c[2]]

    def partials(self):
        return [c for c in self.calls if not c[2]]


async def run_worker(probabilities, frames, engine=None, partial_interval_ms=1000, channel=AudioChannel.LOOPBACK):
    """Drive a worker to completion over a fixed frame script."""
    recorder = Recorder()
    source = FakeAudioSource(frames, channel=channel)
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector(probabilities),
        engine=engine or FakeSttEngine(),
        loop=asyncio.get_running_loop(),
        on_transcript=recorder,
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
