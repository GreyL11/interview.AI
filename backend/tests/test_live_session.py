import asyncio
import time
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


async def test_question_detected_precedes_the_answer(sessions, session_id, retriever):
    """The UI shows what was heard and how it was classified before any answer
    exists, so this must be emitted even if generation later fails."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("How would you handle duplicate records in a pipeline?")
    await drain(live)

    types = collector.types()
    assert EventType.QUESTION_DETECTED in types
    assert types.index(EventType.QUESTION_DETECTED) < types.index(EventType.ANSWER_STARTED)

    detected = collector.of(EventType.QUESTION_DETECTED)[0]
    assert detected.turn_id is not None
    assert detected.data["question"]
    assert detected.data["classification"]["category"]


async def test_question_detected_emitted_even_when_the_answer_fails(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, BrokenLLM())
    collector = Collector()
    live.subscribe(collector)

    await live.ask("What is a database index?")
    await drain(live)

    assert EventType.QUESTION_DETECTED in collector.types()
    assert EventType.ANSWER_ERROR in collector.types()


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


async def test_superseding_a_question_records_a_real_cancel_wait(sessions, session_id, retriever):
    """ask()'s cancel-and-await pattern must actually record time spent
    waiting for the superseded task to unwind, not just report zero."""
    from app.core.metrics import LatencyTrace

    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.02))
    trace = LatencyTrace(speech_end_at=time.monotonic())

    await live.ask("How would you design a URL shortener?")
    await asyncio.sleep(0.01)  # let the first task get into its stream loop
    await live.ask("What is a database index?", trace=trace)
    await drain(live)

    assert trace.cancel_wait_ms is not None
    assert trace.cancel_wait_ms >= 0


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


async def test_a_dangling_question_does_not_fire_immediately(sessions, session_id, retriever):
    """A mid-clause question must not spend a Gemini call before its brief
    stabilization window elapses."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Can you explain what happens when", TranscriptSource.LOOPBACK, is_final=True
    )

    assert EventType.QUESTION_DETECTED not in collector.types()
    assert live._pending_ask is not None

    await live._pending_ask  # let the stabilization window elapse
    await drain(live)

    assert EventType.QUESTION_DETECTED in collector.types()
    assert EventType.ANSWER_COMPLETED in collector.types()


async def test_a_quick_continuation_supersedes_the_pending_dangling_question(
    sessions, session_id, retriever
):
    """The continuation arrives inside the correction-coalesce window, so the
    detector merges it into one complete question -- only that one Gemini
    call should ever happen."""
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Can you explain what happens when", TranscriptSource.LOOPBACK, is_final=True
    )
    pending = live._pending_ask
    assert pending is not None

    await live.on_transcript(
        "a transaction fails?", TranscriptSource.LOOPBACK, is_final=True
    )
    await asyncio.gather(pending, return_exceptions=True)
    await drain(live)

    # The stale timer must not still be alive and must not have fired its own ask.
    assert pending.cancelled()
    assert live._pending_ask is None
    detected = collector.of(EventType.QUESTION_DETECTED)
    assert len(detected) == 1
    assert "transaction fails" in detected[0].data["question"]
    assert len(llm.prompts) == 1


async def test_pending_ask_is_cancelled_on_session_close(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))

    await live.on_transcript(
        "Can you explain what happens when", TranscriptSource.LOOPBACK, is_final=True
    )
    pending = live._pending_ask
    assert pending is not None

    await live.close()
    await asyncio.gather(pending, return_exceptions=True)

    assert pending.cancelled()


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


async def test_detector_diagnostics_are_off_by_default(sessions, session_id, retriever, monkeypatch):
    calls = []
    monkeypatch.setattr("app.realtime.session.log_metric", lambda event, **f: calls.append((event, f)))
    live = build(sessions, session_id, retriever, FakeLLM())

    await live.on_transcript("What is caching?", TranscriptSource.LOOPBACK, is_final=True)

    assert "question_detector_decision" not in [e for e, _ in calls]


async def test_detector_diagnostics_log_the_decision_when_enabled(
    sessions, session_id, retriever, monkeypatch
):
    from app.core.config import settings

    calls = []
    monkeypatch.setattr("app.realtime.session.log_metric", lambda event, **f: calls.append((event, f)))
    monkeypatch.setattr(settings, "question_detector_diagnostics", True)
    live = build(sessions, session_id, retriever, FakeLLM())

    await live.on_transcript("What is caching?", TranscriptSource.LOOPBACK, is_final=True)
    await live.on_transcript("yeah", TranscriptSource.LOOPBACK, is_final=True)

    decisions = [f for e, f in calls if e == "question_detector_decision"]
    assert len(decisions) == 2
    assert decisions[0]["detected"] is True
    assert decisions[0]["category"] == "TECHNICAL_KNOWLEDGE"
    assert decisions[1]["detected"] is False
    assert decisions[1]["reason"]


async def test_diagnostics_never_see_mic_content(sessions, session_id, retriever, monkeypatch):
    from app.core.config import settings

    calls = []
    monkeypatch.setattr("app.realtime.session.log_metric", lambda event, **f: calls.append((event, f)))
    monkeypatch.setattr(settings, "question_detector_diagnostics", True)
    live = build(sessions, session_id, retriever, FakeLLM())

    await live.on_transcript(
        "I think we should use a dictionary", TranscriptSource.MIC, is_final=True
    )

    assert not [f for e, f in calls if e == "question_detector_decision"]


async def test_interviewer_setup_context_reaches_the_llm_but_not_the_ui(
    sessions, session_id, retriever
):
    """Reported bug: 'By using this study, just write a character count
    program.' isn't a question on its own and was previously discarded, so
    the next utterance was answered as a bare, context-free fragment.

    The fix attaches that setup to what the LLM sees, not to what the panel
    displays -- the interviewer didn't ask a two-sentence question, so the UI
    shouldn't show one."""
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "By using this study, just write a character count program.",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await live.on_transcript(
        "How many times each character is repeated?",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert detected
    assert detected[0].data["question"] == "How many times each character is repeated?"

    assert llm.prompts
    assert "character count program" in llm.prompts[-1]
    assert "How many times each character is repeated" in llm.prompts[-1]


async def test_mic_speech_never_becomes_interviewer_context(sessions, session_id, retriever):
    """The candidate's own words, even spoken right before an interviewer
    question, must never be folded into what gets asked."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "I think the answer involves a character count program.",
        TranscriptSource.MIC, is_final=True,
    )
    await live.on_transcript(
        "How many times each character is repeated?",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert detected
    assert "character count program" not in detected[0].data["question"]


async def test_manual_question_is_not_augmented_with_interviewer_context(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "By using this study, just write a character count program.",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await live.on_transcript(
        "What is a database index?", TranscriptSource.MANUAL, is_final=True,
    )
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert detected
    assert detected[0].data["question"] == "What is a database index?"


async def test_short_followup_reaches_the_llm_with_conversation_history(
    sessions, session_id, retriever
):
    """'Why?' is below question_min_words and would normally be rejected --
    but right after a real, answered question it's a legitimate follow-up.
    SessionMemory already carries the prior Q&A into every prompt, so once
    the detector lets it through, no extra plumbing is needed."""
    from app.memory.sqlite_memory import SqliteSessionMemory

    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript("What is a hash map?", TranscriptSource.LOOPBACK, is_final=True)
    await drain(live)

    await live.on_transcript("Why?", TranscriptSource.LOOPBACK, is_final=True)
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert len(detected) == 2
    assert detected[1].data["question"] == "Why?"
    assert "hash map" in llm.prompts[-1]


async def test_self_interruption_never_leaves_an_incorrect_answer(sessions, session_id, retriever):
    """'Explain hash ma-' matches on its own (it contains 'explain'), but an
    interviewer correcting themselves mid-thought must never let that
    incomplete fragment surface as a completed, incorrect answer."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript("Explain hash ma-", TranscriptSource.LOOPBACK, is_final=True)
    await live.on_transcript(
        "No actually explain hash collisions.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    completed = collector.of(EventType.ANSWER_COMPLETED)
    assert len(completed) == 1

    turns = sessions.get_turns(session_id)
    answered = [t for t in turns if t.status == TurnStatus.ANSWERED]
    cancelled = [t for t in turns if t.status == TurnStatus.CANCELLED]
    assert len(answered) == 1
    assert len(cancelled) == 1
    assert "collisions" in answered[0].question.lower()


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
