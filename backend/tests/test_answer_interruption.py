"""Superseding an answer that had already streamed useful text.

The distinction under test: a turn cancelled *before* any visible content is
noise and stays CANCELLED; one cancelled *after* the user saw text is
INTERRUPTED, keeps its partial for history, and -- critically -- stays out of
the conversation memory fed back to the LLM.
"""

import asyncio
import uuid

import pytest

from app.documents.schemas import utcnow
from app.memory.session_memory import InMemorySessionMemory
from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.realtime.session import LiveSession
from app.sessions.schemas import Session, TurnStatus
from app.storage.session_repository import SessionRepository
from tests.fakes import SlowStreamingLLM

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
        session_id=session_id, sessions=sessions, retriever=retriever,
        llm=llm, memory=memory or InMemorySessionMemory(),
    )


class Collector:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    def of(self, t):
        return [e for e in self.events if e.type == t]


async def drain(live):
    if live._task is not None:
        await asyncio.gather(live._task, return_exceptions=True)


async def _supersede_mid_stream(live) -> None:
    """Start A, let it stream visible text, then supersede with B."""
    await live.ask("Explain caching?")
    for _ in range(40):
        await asyncio.sleep(0.01)
        if live._current_partial:
            break
    assert live._current_partial, "A never produced visible text; test is not exercising interruption"
    await live.ask("What is a database index?")
    await drain(live)


async def test_interrupted_turn_keeps_its_partial_text(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01))

    await _supersede_mid_stream(live)

    turns = sessions.get_turns(session_id)
    interrupted = [t for t in turns if t.status == TurnStatus.INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].question == "Explain caching?"
    assert interrupted[0].answer is not None
    assert interrupted[0].answer.summary, "the partial the user already saw must survive"


async def test_interrupted_partial_never_reaches_conversation_memory(
    sessions, session_id, retriever
):
    """A truncated answer is useful to a human reading history and misleading
    to the model. get_answered_turns() filters on ANSWERED, so INTERRUPTED is
    excluded structurally rather than by a second filter."""
    memory = SqliteSessionMemory(sessions)
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01), memory)

    await _supersede_mid_stream(live)

    history = " ".join(memory.get_history(session_id))
    assert "Explain caching?" not in history
    assert not [t for t in memory._sessions.get_answered_turns(session_id)
                if t.status == TurnStatus.INTERRUPTED]


async def test_cancel_before_any_content_stays_cancelled(sessions, session_id, retriever):
    """No visible text was produced, so there is nothing worth preserving --
    it must not become a meaningless INTERRUPTED entry with an empty answer."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))

    await live.ask("Explain caching?")
    await live.ask("What is a database index?")  # immediately, before any delta
    await drain(live)

    statuses = {t.question: t.status for t in sessions.get_turns(session_id)}
    assert statuses["Explain caching?"] == TurnStatus.CANCELLED
    assert not [t for t in sessions.get_turns(session_id)
                if t.status == TurnStatus.INTERRUPTED]


async def test_the_cancelled_event_reports_interruption_and_partial(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01))
    collector = Collector()
    live.subscribe(collector)

    await _supersede_mid_stream(live)

    cancelled = collector.of(EventType.ANSWER_CANCELLED)
    assert cancelled
    data = cancelled[0].data
    assert data["interrupted"] is True
    assert data["partial_summary"]
    assert data["reason"] == "superseded"  # existing field, unchanged


async def test_cancelled_before_content_event_is_flagged_not_interrupted(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    collector = Collector()
    live.subscribe(collector)

    await live.ask("Explain caching?")
    await live.ask("What is a database index?")
    await drain(live)

    cancelled = collector.of(EventType.ANSWER_CANCELLED)
    assert cancelled
    assert cancelled[0].data["interrupted"] is False
    assert cancelled[0].data["partial_summary"] is None


async def test_the_newest_question_still_wins(sessions, session_id, retriever):
    """Preserving the partial must not weaken supersession: exactly one turn
    completes, and it is the newest."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01))
    collector = Collector()
    live.subscribe(collector)

    await _supersede_mid_stream(live)

    completed = collector.of(EventType.ANSWER_COMPLETED)
    assert len(completed) == 1
    answered = [t for t in sessions.get_turns(session_id) if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1
    assert answered[0].question == "What is a database index?"


async def test_rapid_fire_preserves_partials_and_leaves_one_winner(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01))

    for question in ("What is OOP?", "What is polymorphism?", "What is inheritance?"):
        await live.ask(question)
        for _ in range(40):
            await asyncio.sleep(0.01)
            if live._current_partial:
                break
    await drain(live)

    turns = sessions.get_turns(session_id)
    answered = [t for t in turns if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1
    assert answered[0].question == "What is inheritance?"
    # The two superseded turns kept what the user had already seen.
    interrupted = [t for t in turns if t.status == TurnStatus.INTERRUPTED]
    assert len(interrupted) == 2
    assert all(t.answer and t.answer.summary for t in interrupted)


async def test_session_close_mid_stream_preserves_the_partial(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.01))

    await live.ask("Explain caching?")
    for _ in range(40):
        await asyncio.sleep(0.01)
        if live._current_partial:
            break
    await live.close()

    interrupted = [t for t in sessions.get_turns(session_id)
                   if t.status == TurnStatus.INTERRUPTED]
    assert len(interrupted) == 1
    assert interrupted[0].answer.summary
