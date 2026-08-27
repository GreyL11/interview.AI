import uuid

import pytest

from app.documents.schemas import utcnow
from app.memory.session_memory import InMemorySessionMemory
from app.memory.sqlite_memory import SqliteSessionMemory
from app.memory.summarizer import SessionSummarizer
from app.schemas.answer import Answer
from app.sessions.schemas import Session, Turn
from app.storage.session_repository import SessionRepository
from tests.fakes import FakeLLM


@pytest.fixture
def sessions(database) -> SessionRepository:
    return SessionRepository(database)


@pytest.fixture
def session_id(sessions) -> str:
    sid = str(uuid.uuid4())
    sessions.create(Session(session_id=sid, started_at=utcnow()))
    return sid


def add_answered_turn(sessions, session_id, seq, question, summary):
    turn = sessions.create_turn(Turn(session_id=session_id, seq=seq, question=question))
    sessions.complete_turn(turn.turn_id, Answer(summary=summary), False, 10)
    return turn


# ------------------------------------------------------------------ in-memory


def test_in_memory_roundtrip():
    memory = InMemorySessionMemory()
    memory.append_turn("s1", "What is a B-tree?", "A sorted index structure.")
    assert memory.get_history("s1") == ["Q: What is a B-tree?", "A: A sorted index structure."]


def test_in_memory_ignores_missing_session_id():
    memory = InMemorySessionMemory()
    memory.append_turn(None, "q", "a")
    assert memory.get_history(None) == []


def test_in_memory_is_bounded_by_token_budget():
    memory = InMemorySessionMemory(max_tokens=40)
    for i in range(40):
        memory.append_turn("s1", f"Question number {i} with padding text", f"Answer {i} padded out")

    history = memory.get_history("s1")
    assert len(history) < 80
    assert len(history) % 2 == 0  # never a question without its answer
    assert "Question number 39" in history[-2]


def test_in_memory_keeps_at_least_the_last_pair():
    memory = InMemorySessionMemory(max_tokens=1)
    memory.append_turn("s1", "x" * 500, "y" * 500)
    assert len(memory.get_history("s1")) == 2


def test_bounded_context_prefixes_summary():
    memory = InMemorySessionMemory()
    memory.append_turn("s1", "q", "a")
    memory.set_summary("s1", "Discussed Kafka.", ["kafka"])

    context = memory.bounded_context("s1")
    assert context[0].startswith("[Earlier in this session] Discussed Kafka.")
    assert "Q: q" in context


# ---------------------------------------------------------------------- sqlite


def test_sqlite_memory_reads_persisted_turns(sessions, session_id):
    add_answered_turn(sessions, session_id, 0, "What is sharding?", "Splitting data by key.")
    memory = SqliteSessionMemory(sessions)
    assert memory.get_history(session_id) == ["Q: What is sharding?", "A: Splitting data by key."]


def test_sqlite_memory_survives_a_new_instance(sessions, session_id):
    add_answered_turn(sessions, session_id, 0, "q", "a")
    # A fresh instance models a process restart: history is in SQLite, not RAM.
    assert SqliteSessionMemory(sessions).get_history(session_id) == ["Q: q", "A: a"]


def test_sqlite_memory_is_bounded(sessions, session_id):
    for i in range(60):
        add_answered_turn(sessions, session_id, i, f"Question {i} " + "pad " * 20, f"Answer {i}")

    history = SqliteSessionMemory(sessions, max_tokens=200).get_history(session_id)
    assert len(history) % 2 == 0
    assert len(history) < 120
    assert "Question 59" in history[-2]


def test_sqlite_memory_excludes_unanswered_turns(sessions, session_id):
    sessions.create_turn(Turn(session_id=session_id, seq=0, question="never answered"))
    assert SqliteSessionMemory(sessions).get_history(session_id) == []


def test_sqlite_memory_empty_for_unknown_session(sessions):
    assert SqliteSessionMemory(sessions).get_history("nope") == []
    assert SqliteSessionMemory(sessions).get_history(None) == []


# ------------------------------------------------------------------ summarizer


@pytest.mark.asyncio
async def test_summarizer_persists_summary_and_topics(sessions, session_id):
    for i in range(4):
        add_answered_turn(sessions, session_id, i, f"q{i}", f"a{i}")

    llm = FakeLLM(Answer(summary='{"summary": "Talked about Kafka.", "topics": ["kafka", "etl"]}'))
    result = await SessionSummarizer(sessions, llm).summarize(session_id, through_seq=3)

    assert result.summary == "Talked about Kafka."
    assert result.topics == ["kafka", "etl"]
    assert result.covered_through_seq == 3
    assert sessions.get_summary(session_id).summary == "Talked about Kafka."


@pytest.mark.asyncio
async def test_summarizer_survives_llm_failure(sessions, session_id):
    from app.llm.base import LLMError

    add_answered_turn(sessions, session_id, 0, "q", "a")

    class BrokenLLM(FakeLLM):
        async def generate_answer(self, prompt):
            raise LLMError("gemini down")

    # Degrades context quality; must never break the session.
    assert await SessionSummarizer(sessions, BrokenLLM()).summarize(session_id, 1) is None


@pytest.mark.asyncio
async def test_summarizer_noop_without_turns(sessions, session_id):
    assert await SessionSummarizer(sessions, FakeLLM()).summarize(session_id, 5) is None


def test_needs_summary_only_past_what_is_covered(sessions, session_id):
    summarizer = SessionSummarizer(sessions, FakeLLM())
    assert summarizer.needs_summary(session_id, 0) is False
    assert summarizer.needs_summary(session_id, 3) is True


@pytest.mark.asyncio
async def test_needs_summary_false_once_covered(sessions, session_id):
    add_answered_turn(sessions, session_id, 0, "q", "a")
    llm = FakeLLM(Answer(summary='{"summary": "s", "topics": []}'))
    summarizer = SessionSummarizer(sessions, llm)

    await summarizer.summarize(session_id, through_seq=3)
    assert summarizer.needs_summary(session_id, 3) is False
    assert summarizer.needs_summary(session_id, 4) is True
