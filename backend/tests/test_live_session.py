import asyncio
import uuid

import pytest

from app.documents.schemas import KnowledgeType, utcnow
from app.memory.session_memory import InMemorySessionMemory
from app.realtime.events import CancelReason, EventType
from app.realtime.session import LiveSession
from app.schemas.answer import Answer
from app.sessions.schemas import Session, TranscriptSource, TurnStatus
from app.storage.session_repository import SessionRepository
from tests.fakes import BrokenLLM, FakeLLM, SlowStreamingLLM

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sessions(database) -> SessionRepository:
    return SessionRepository(database)


@pytest.fixture
def session_id(sessions) -> str:
    sid = str(uuid.uuid4())
    sessions.create(Session(session_id=sid, started_at=utcnow()))
    return sid


def build(sessions, session_id, retriever, llm, memory=None) -> LiveSession:
    return LiveSession(
        session_id=session_id,
        sessions=sessions,
        retriever=retriever,
        llm=llm,
        memory=memory or InMemorySessionMemory(),
    )


class Collector:
    def __init__(self) -> None:
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    def types(self):
        return [e.type for e in self.events]

    def of(self, event_type):
        return [e for e in self.events if e.type == event_type]


async def drain(live):
    if live._task is not None:
        await asyncio.gather(live._task, return_exceptions=True)


# ------------------------------------------------------------------ happy path


async def test_manual_question_produces_a_completed_answer(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("How would you handle duplicate records in a pipeline?")
    await drain(live)

    assert EventType.ANSWER_STARTED in collector.types()
    assert EventType.ANSWER_COMPLETED in collector.types()
    completed = collector.of(EventType.ANSWER_COMPLETED)[0]
    assert completed.data["answer"]["summary"]
    assert completed.data["latency_ms"] >= 0


async def test_events_carry_monotonic_seq(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("What is a database index?")
    await drain(live)

    seqs = [e.seq for e in collector.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


async def test_answer_deltas_stream_the_summary(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("What is a database index?")
    await drain(live)

    deltas = collector.of(EventType.ANSWER_DELTA)
    assert deltas
    assert deltas[-1].data["summary"] in collector.of(EventType.ANSWER_COMPLETED)[0].data["answer"]["summary"]


async def test_turn_is_persisted_as_answered(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    await live.ask("What is a database index?")
    await drain(live)

    turns = sessions.get_turns(session_id)
    assert len(turns) == 1
    assert turns[0].status == TurnStatus.ANSWERED
    assert turns[0].answer is not None


# ---------------------------------------------------------------- cancellation


async def test_new_question_cancels_the_previous_answer(sessions, session_id, retriever):
    llm = SlowStreamingLLM(chunk_delay=0.05)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    first = await live.ask("How would you design a URL shortener?")
    await asyncio.sleep(0.06)  # let it start streaming
    second = await live.ask("What is a database index?")
    await drain(live)

    cancelled = collector.of(EventType.ANSWER_CANCELLED)
    assert cancelled
    assert cancelled[0].turn_id == first
    assert cancelled[0].data["reason"] == CancelReason.SUPERSEDED.value

    # The superseded turn must never report a completed answer.
    completed_turns = [e.turn_id for e in collector.of(EventType.ANSWER_COMPLETED)]
    assert first not in completed_turns
    assert second in completed_turns


async def test_cancelled_turn_is_marked_in_storage(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    first = await live.ask("How would you design a URL shortener?")
    await asyncio.sleep(0.06)
    await live.ask("What is a database index?")
    await drain(live)

    by_id = {t.turn_id: t for t in sessions.get_turns(session_id)}
    assert by_id[first].status == TurnStatus.CANCELLED


async def test_cancelled_turn_is_excluded_from_memory(sessions, session_id, retriever):
    from app.memory.sqlite_memory import SqliteSessionMemory

    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    await live.ask("How would you design a URL shortener?")
    await asyncio.sleep(0.06)
    await live.ask("What is a database index?")
    await drain(live)

    history = " ".join(SqliteSessionMemory(sessions).get_history(session_id))
    assert "URL shortener" not in history


async def test_explicit_cancel(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("How would you design a URL shortener?")
    await asyncio.sleep(0.06)
    await live.cancel(CancelReason.USER_STOP)

    cancelled = collector.of(EventType.ANSWER_CANCELLED)
    assert cancelled[0].data["reason"] == CancelReason.USER_STOP.value


async def test_cancelling_with_nothing_running_is_harmless(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, FakeLLM())
    await live.cancel()  # must not raise


async def test_rapid_fire_questions_leave_one_winner(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.02))
    collector = Collector()
    live.subscribe(collector)

    for i in range(5):
        await live.ask(f"How would you design system number {i}?")
        await asyncio.sleep(0.01)
    await drain(live)

    completed = collector.of(EventType.ANSWER_COMPLETED)
    assert len(completed) == 1
    answered = [t for t in sessions.get_turns(session_id) if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1


# ------------------------------------------------------------------ transcript


async def test_partial_transcript_never_triggers_an_answer(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, FakeLLM())
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript("How would you handle dup", TranscriptSource.LOOPBACK, is_final=False)

    assert collector.types() == [EventType.TRANSCRIPT_PARTIAL]
    assert sessions.get_turns(session_id) == []


async def test_final_transcript_triggers_detection(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "How would you handle duplicate records?", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    assert EventType.TRANSCRIPT_FINAL in collector.types()
    assert EventType.ANSWER_COMPLETED in collector.types()


async def test_mic_transcript_is_recorded_but_never_answered(sessions, session_id, retriever):
    """The candidate's own speech is for review, not for answering."""
    live = build(sessions, session_id, retriever, FakeLLM())
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "How would you handle duplicate records?", TranscriptSource.MIC, is_final=True
    )

    assert EventType.TRANSCRIPT_FINAL in collector.types()
    assert EventType.ANSWER_STARTED not in collector.types()
    assert sessions.get_transcript(session_id)[0].source == TranscriptSource.MIC


async def test_filler_utterance_is_rejected_visibly(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, FakeLLM())
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript("yeah", TranscriptSource.LOOPBACK, is_final=True)

    rejected = collector.of(EventType.QUESTION_REJECTED)
    assert rejected
    assert rejected[0].data["reason"]
    assert sessions.get_turns(session_id) == []


# ----------------------------------------------------------------- retrieval


async def test_personal_question_retrieves_and_reports_hits(
    sessions, session_id, service, retriever
):
    doc = await service.upload(
        "cv.txt", b"At Acme I built a Kafka streaming ingestion pipeline.", KnowledgeType.RESUME
    )
    await service.ingest(doc.document_id)

    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("Tell me about a project you built with Kafka")
    await drain(live)

    assert EventType.ANSWER_RETRIEVING in collector.types()
    completed = collector.of(EventType.ANSWER_COMPLETED)[0]
    assert completed.data["context_found"] is True
    assert completed.data["retrieval_hits"]


async def test_technical_question_skips_retrieval(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("What is a database index?")
    await drain(live)

    assert EventType.ANSWER_RETRIEVING not in collector.types()
    assert collector.of(EventType.ANSWER_COMPLETED)[0].data["context_found"] is False


async def test_ungrounded_personal_answer_is_flagged(sessions, session_id, retriever):
    llm = SlowStreamingLLM(
        Answer(summary="I built a Kafka pipeline at Acme."), chunk_delay=0
    )
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await live.ask("Tell me about a project you built")
    await drain(live)

    warnings = collector.of(EventType.ANSWER_COMPLETED)[0].data["answer"]["warnings"]
    assert warnings


# --------------------------------------------------------------------- errors


async def test_llm_failure_emits_answer_error(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, BrokenLLM())
    collector = Collector()
    live.subscribe(collector)

    await live.ask("What is a database index?")
    await drain(live)

    errors = collector.of(EventType.ANSWER_ERROR)
    assert errors
    assert "exploded" in errors[0].data["message"]
    assert sessions.get_turns(session_id)[0].status == TurnStatus.FAILED


async def test_session_survives_an_llm_failure(sessions, session_id, retriever):
    live = LiveSession(
        session_id=session_id, sessions=sessions, retriever=retriever,
        llm=BrokenLLM(), memory=InMemorySessionMemory(),
    )
    await live.ask("What is a database index?")
    await drain(live)

    live._llm = SlowStreamingLLM(chunk_delay=0)
    collector = Collector()
    live.subscribe(collector)
    await live.ask("What is sharding?")
    await drain(live)

    assert collector.of(EventType.ANSWER_COMPLETED)


# ------------------------------------------------------------------- replay


async def test_replay_returns_only_missed_events(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    await live.ask("What is a database index?")
    await drain(live)

    everything = live.replay_since(0)
    assert everything
    tail = live.replay_since(everything[0].seq)
    assert len(tail) == len(everything) - 1


async def test_dead_subscriber_is_dropped_not_fatal(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))

    async def broken(event):
        raise RuntimeError("socket closed")

    live.subscribe(broken)
    good = Collector()
    live.subscribe(good)

    await live.ask("What is a database index?")
    await drain(live)

    assert good.events  # the healthy subscriber still received everything


async def test_close_ends_the_session(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.close()

    assert EventType.SESSION_ENDED in collector.types()
    from app.sessions.schemas import SessionStatus

    assert sessions.get(session_id).status == SessionStatus.ENDED
