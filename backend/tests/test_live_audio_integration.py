"""End-to-end: audio frames in, coaching answer out.

Real devices and real Whisper are replaced by fakes, so this exercises the
whole wiring -- VAD segmentation, channel routing, question detection,
cancellation, retrieval, streaming -- deterministically and with no hardware.
"""

import asyncio
import uuid

import pytest

from app.audio.base import AudioChannel
from app.documents.schemas import KnowledgeType, utcnow
from app.memory.session_memory import InMemorySessionMemory
from app.realtime.audio_binding import build_pipeline
from app.realtime.events import EventType
from app.realtime.session import LiveSession
from app.sessions.schemas import Session, TranscriptSource
from app.storage.session_repository import SessionRepository
from app.stt.vad import FRAME_MS, EnergyVad
from tests.fakes import (
    FakeAudioSource,
    FakeSttEngine,
    SlowStreamingLLM,
    silence_frames,
    speech_frames,
)

pytestmark = pytest.mark.asyncio

SILENCE_TO_END = 700 // FRAME_MS + 2


@pytest.fixture
def sessions(database) -> SessionRepository:
    return SessionRepository(database)


@pytest.fixture
def live(sessions, retriever):
    session_id = str(uuid.uuid4())
    sessions.create(Session(session_id=session_id, started_at=utcnow()))
    return LiveSession(
        session_id=session_id,
        sessions=sessions,
        retriever=retriever,
        llm=SlowStreamingLLM(chunk_delay=0),
        memory=InMemorySessionMemory(),
    )


class Collector:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]

    def of(self, t):
        return [e for e in self.events if e.type == t]


def utterance(frame_count=20):
    return speech_frames(frame_count) + silence_frames(SILENCE_TO_END)


async def run_audio(live, sources, engine=None, timeout=6.0):
    """Start capture, wait for sources to drain, then stop."""
    pipeline = build_pipeline(
        live,
        engine or FakeSttEngine(),
        sources=sources,
        loop=asyncio.get_running_loop(),
        detector_factory=EnergyVad,
    )
    await live.start_audio(pipeline)

    loop = asyncio.get_running_loop()
    for source in sources:
        await loop.run_in_executor(None, source.exhausted.wait, timeout)
    await asyncio.sleep(0.3)

    await live.stop_audio()
    if live._task is not None:
        await asyncio.gather(live._task, return_exceptions=True)


async def test_spoken_question_reaches_a_completed_answer(live):
    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(utterance(), channel=AudioChannel.LOOPBACK)

    await run_audio(live, [source])

    types = collector.types()
    assert EventType.TRANSCRIPT_FINAL in types
    assert EventType.QUESTION_DETECTED in types or EventType.ANSWER_STARTED in types
    assert EventType.ANSWER_COMPLETED in types
    assert collector.of(EventType.ANSWER_COMPLETED)[0].data["answer"]["summary"]


async def test_transcript_is_persisted(live, sessions):
    source = FakeAudioSource(utterance(), channel=AudioChannel.LOOPBACK)
    await run_audio(live, [source])

    stored = sessions.get_transcript(live.session_id)
    assert stored
    assert stored[0].source == TranscriptSource.LOOPBACK


async def test_mic_audio_is_transcribed_but_never_answered(live, sessions):
    """The candidate's own voice must not trigger the coach."""
    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(utterance(), channel=AudioChannel.MIC)

    await run_audio(live, [source])

    assert EventType.TRANSCRIPT_FINAL in collector.types()
    assert EventType.ANSWER_STARTED not in collector.types()
    assert sessions.get_transcript(live.session_id)[0].source == TranscriptSource.MIC
    assert sessions.get_turns(live.session_id) == []


async def test_both_channels_are_tagged_correctly(live, sessions):
    mic = FakeAudioSource(utterance(15), channel=AudioChannel.MIC)
    loopback = FakeAudioSource(utterance(15), channel=AudioChannel.LOOPBACK)

    await run_audio(live, [loopback, mic])

    sources = {t.source for t in sessions.get_transcript(live.session_id)}
    assert sources == {TranscriptSource.MIC, TranscriptSource.LOOPBACK}


async def test_silence_yields_nothing(live, sessions):
    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(silence_frames(40), channel=AudioChannel.LOOPBACK)

    await run_audio(live, [source])

    assert EventType.TRANSCRIPT_FINAL not in collector.types()
    assert sessions.get_turns(live.session_id) == []


async def test_filler_speech_is_rejected(live, sessions):
    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(utterance(), channel=AudioChannel.LOOPBACK)

    await run_audio(live, [source], engine=FakeSttEngine(final_text="yeah"))

    assert collector.of(EventType.QUESTION_REJECTED)
    assert sessions.get_turns(live.session_id) == []


async def test_grounded_answer_when_documents_exist(live, service):
    doc = await service.upload(
        "cv.txt", b"At Acme I built a Kafka streaming ingestion pipeline.", KnowledgeType.RESUME
    )
    await service.ingest(doc.document_id)

    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(utterance(), channel=AudioChannel.LOOPBACK)
    engine = FakeSttEngine(final_text="Tell me about a project you built with Kafka")

    await run_audio(live, [source], engine=engine)

    completed = collector.of(EventType.ANSWER_COMPLETED)[0]
    assert completed.data["context_found"] is True
    assert completed.data["retrieval_hits"]


async def test_audio_status_is_emitted_on_start_and_stop(live):
    collector = Collector()
    live.subscribe(collector)
    source = FakeAudioSource(silence_frames(5), channel=AudioChannel.LOOPBACK)

    await run_audio(live, [source])

    statuses = [e.data.get("audio") for e in collector.of(EventType.SESSION_STATUS)]
    assert "ok" in statuses
    assert "stopped" in statuses


async def test_starting_audio_twice_is_idempotent(live):
    source = FakeAudioSource(silence_frames(200), channel=AudioChannel.LOOPBACK)
    pipeline = build_pipeline(
        live, FakeSttEngine(), sources=[source], detector_factory=EnergyVad
    )

    await live.start_audio(pipeline)
    channels = await live.start_audio(pipeline)

    assert live.audio_active
    assert channels == [TranscriptSource.LOOPBACK.value]
    await live.stop_audio()


async def test_close_stops_audio(live):
    source = FakeAudioSource(silence_frames(500), channel=AudioChannel.LOOPBACK)
    pipeline = build_pipeline(
        live, FakeSttEngine(), sources=[source], detector_factory=EnergyVad
    )
    await live.start_audio(pipeline)

    await live.close()

    assert not live.audio_active


async def test_stopping_audio_that_never_started_is_harmless(live):
    await live.stop_audio()
    assert not live.audio_active


async def test_a_question_produces_one_correlated_latency_trace(live, monkeypatch):
    """The whole speech-end -> first-visible-token path must be reconstructable
    from one grep-able log line, keyed by question_id."""
    import app.core.metrics as metrics_module

    traces = []
    original = metrics_module.log_metric

    def record(event, **fields):
        if event == "question_latency_trace":
            traces.append(fields)
        return original(event, **fields)

    monkeypatch.setattr(metrics_module, "log_metric", record)

    source = FakeAudioSource(utterance(), channel=AudioChannel.LOOPBACK)
    await run_audio(live, [source])

    assert len(traces) == 1
    trace = traces[0]
    assert trace["question_id"] is not None
    # STT and LLM stage timings must all be present and non-negative.
    for field in (
        "speech_end_to_stt_final_ms",
        "stt_queue_wait_ms",
        "stt_inference_ms",
        "stt_final_to_question_detected_ms",
        "llm_task_to_request_ms",
        "llm_request_to_first_text_token_ms",
        "total_question_to_first_visible_token_ms",
    ):
        assert field in trace, f"missing {field}"
        assert trace[field] >= 0, f"{field} was negative: {trace[field]}"
    # Cancelling nothing (first question of the session) costs nothing.
    assert trace["previous_answer_cancel_wait_ms"] == 0
