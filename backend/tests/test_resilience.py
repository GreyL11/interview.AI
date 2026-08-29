"""Failure-recovery and resilience scenarios.

Complements existing coverage rather than repeating it: provider retry/classification
and message sanitization live in test_groq_client.py; WebSocket malformed
messages, reconnect replay, and disconnect survival live in test_ws.py; basic
STT failure survival lives in test_stt_pipeline.py. This file covers the gaps
those leave -- mid-stream failure, recovery *after* a failure, capture failing
partway through, and shutdown while work is genuinely in flight.
"""

import asyncio

import pytest

from app.audio.base import AudioChannel, AudioError
from app.llm.base import LLMClient, LLMError
from app.realtime.events import EventType
from app.schemas.answer import Answer
from app.sessions.schemas import TranscriptSource, TurnStatus
from app.stt.base import SttError, Transcript
from app.stt.pipeline import TranscriptionWorker
from app.stt.vad import FRAME_MS
from tests.fakes import (
    FakeAudioSource,
    FakeSttEngine,
    ScriptedSpeechDetector,
    SlowStreamingLLM,
    silence_frames,
    speech_frames,
)
from tests.replay_harness import ReplayEvent, ReplayHarness

pytestmark = pytest.mark.asyncio

SILENCE_TO_END = 700 // FRAME_MS + 2


# ------------------------------------------------------------------- LLM


class FailsMidStreamLLM(LLMClient):
    """Yields part of an answer, then fails -- the case a
    fail-before-first-token test cannot cover."""

    def __init__(self, fail_after: int = 2) -> None:
        self.fail_after = fail_after
        self.prompts: list[str] = []
        self.calls = 0

    async def generate_answer(self, prompt: str) -> Answer:
        raise NotImplementedError

    async def stream_answer(self, prompt: str):
        self.prompts.append(prompt)
        self.calls += 1
        payload = Answer(summary="Partial answer text here.", key_points=["a"]).model_dump_json()
        for i in range(0, len(payload), 12):
            if i // 12 >= self.fail_after:
                raise LLMError("connection reset mid-stream")
            yield payload[i : i + 12]


class RecoveringLLM(LLMClient):
    """Fails the first question, succeeds on every later one."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_answer(self, prompt: str) -> Answer:
        raise NotImplementedError

    async def stream_answer(self, prompt: str):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            raise LLMError("service unavailable")
        yield Answer(summary="Recovered answer.", key_points=["ok"]).model_dump_json()


async def test_llm_failure_mid_stream_reports_an_error_and_does_not_complete(monkeypatch):
    h = ReplayHarness(llm=FailsMidStreamLLM(), monkeypatch=monkeypatch)
    try:
        result = await h.play([ReplayEvent(at_ms=0, text="Explain caching.")])

        assert result.of(EventType.ANSWER_ERROR)
        assert result.of(EventType.ANSWER_COMPLETED) == []
        failed = [t for t in h.sessions.get_turns(h.session_id)
                  if t.status == TurnStatus.FAILED]
        assert len(failed) == 1
    finally:
        h.dispose()


async def test_the_next_question_still_works_after_an_llm_failure(monkeypatch):
    h = ReplayHarness(llm=RecoveringLLM(), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="Explain caching."),
            ReplayEvent(at_ms=5000, text="What is a database index?"),
        ])

        assert result.of(EventType.ANSWER_ERROR)
        completed = result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1
        assert completed[0].data["answer"]["summary"] == "Recovered answer."
    finally:
        h.dispose()


# ------------------------------------------------------------------- STT


class FailsOnceSttEngine(FakeSttEngine):
    """Raises on the first final, succeeds afterwards -- proves one failed
    item does not wedge the worker or the shared scheduler."""

    def __init__(self) -> None:
        super().__init__()
        self.finals_seen = 0

    def transcribe(self, audio, is_final):
        self.calls.append((len(audio), is_final))
        if is_final:
            self.finals_seen += 1
            if self.finals_seen == 1:
                raise SttError("model exploded")
        return Transcript(
            text=self.final_text if is_final else "partial",
            is_final=is_final, duration_ms=len(audio) // 16,
        )


class Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, text, source, is_final, trace=None):
        self.calls.append((text, source, is_final))


async def test_stt_recovers_after_a_failed_utterance():
    """Two utterances, the first of which fails inside the engine. The second
    must still be transcribed and published."""
    engine = FailsOnceSttEngine()
    recorder = Recorder()
    burst, gap = speech_frames(15), silence_frames(SILENCE_TO_END)
    frames = burst + gap + burst + gap
    probabilities = (
        [0.9] * 15 + [0.0] * SILENCE_TO_END + [0.9] * 15 + [0.0] * SILENCE_TO_END
    )
    source = FakeAudioSource(frames, channel=AudioChannel.LOOPBACK)
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector(probabilities),
        engine=engine,
        loop=asyncio.get_running_loop(),
        on_transcript=recorder,
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(None, source.exhausted.wait, 5.0)
    await asyncio.sleep(0.2)
    worker.stop()
    await asyncio.sleep(0.05)

    assert worker.errors == 1              # the failure was counted, not swallowed
    assert engine.finals_seen == 2         # the worker kept going
    finals = [c for c in recorder.calls if c[2]]
    assert len(finals) == 1                # only the surviving utterance published


# ----------------------------------------------------------------- audio


class FailsMidCaptureSource(FakeAudioSource):
    """Raises partway through the frame stream, modelling a device that
    disappears mid-session."""

    def __init__(self, frames, fail_after: int = 5):
        super().__init__(frames)
        self.fail_after = fail_after

    def frames(self):
        for i, frame in enumerate(self._frames):
            if not self._running:
                break
            if i >= self.fail_after:
                self.exhausted.set()
                raise AudioError("device disconnected")
            yield frame
        self.exhausted.set()


async def test_capture_failing_mid_session_does_not_crash_the_worker():
    source = FailsMidCaptureSource(silence_frames(50), fail_after=5)
    worker = TranscriptionWorker(
        source=source,
        detector=ScriptedSpeechDetector([0.0] * 50),
        engine=FakeSttEngine(),
        loop=asyncio.get_running_loop(),
        on_transcript=Recorder(),
    )

    worker.start()
    await asyncio.get_running_loop().run_in_executor(None, source.exhausted.wait, 5.0)
    await asyncio.sleep(0.1)
    worker.stop()  # must not raise

    assert worker._thread is None
    assert not worker._scheduler.running


async def test_manual_questions_still_work_when_audio_never_started(monkeypatch):
    """A machine with no usable capture device must still be usable."""
    h = ReplayHarness(monkeypatch=monkeypatch)
    try:
        assert not h.live.audio_active
        turn_id = await h.live.ask("What is a database index?")
        await h.settle()

        assert turn_id is not None
        assert h.result.of(EventType.ANSWER_COMPLETED)
    finally:
        h.dispose()


# -------------------------------------------------------------- shutdown


async def test_close_during_a_streaming_answer_emits_no_post_close_answer(monkeypatch):
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.05), monkeypatch=monkeypatch)
    try:
        await h.live.on_transcript(
            "Explain caching.", TranscriptSource.LOOPBACK, is_final=True, now=0.0
        )
        assert h.live._task is not None and not h.live._task.done()

        await h.live.close()
        seq_at_close = h.result.events[-1].seq
        await asyncio.sleep(0.15)  # well past when the stream would have finished

        # Nothing may be emitted after SESSION_ENDED.
        assert h.result.events[-1].seq == seq_at_close
        assert h.result.events[-1].type == EventType.SESSION_ENDED
        assert h.result.of(EventType.ANSWER_COMPLETED) == []
    finally:
        h.dispose()


async def test_close_cancels_a_pending_stabilization_timer(monkeypatch):
    h = ReplayHarness(stabilization_ms=5000, monkeypatch=monkeypatch)
    try:
        await h.live.on_transcript(
            "Can you explain what happens when", TranscriptSource.LOOPBACK,
            is_final=True, now=0.0,
        )
        pending = h.live._pending_ask
        assert pending is not None and not pending.done()

        await h.live.close()
        await asyncio.gather(pending, return_exceptions=True)

        assert pending.cancelled()
        assert h.result.of(EventType.QUESTION_DETECTED) == []
    finally:
        h.dispose()


async def test_close_leaves_no_running_tasks(monkeypatch):
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.05), monkeypatch=monkeypatch)
    try:
        await h.live.on_transcript(
            "Explain caching.", TranscriptSource.LOOPBACK, is_final=True, now=0.0
        )
        await h.live.close()
        await asyncio.sleep(0.05)

        assert h.live._task is None or h.live._task.done()
        assert h.live._pending_ask is None or h.live._pending_ask.done()
        assert all(t.done() for t in h.live._background)
    finally:
        h.dispose()
