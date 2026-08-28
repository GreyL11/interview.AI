"""End-to-end coding-interview scenarios through the real LiveSession/
QuestionDetector pipeline. Unit-level coverage for the underlying mechanisms
(merge windows, classifier regexes) lives in test_question_detector.py,
test_classifier.py, and test_prompt_detector.py -- this file is about the
observable, wired-together behavior a candidate would actually see.
"""

import uuid

import pytest

from app.documents.schemas import utcnow
from app.memory.session_memory import InMemorySessionMemory
from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.realtime.session import LiveSession
from app.sessions.schemas import Session, TranscriptSource, TurnStatus
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
    if live._pending_ask is not None:
        await live._pending_ask
    if live._task is not None:
        import asyncio
        await asyncio.gather(live._task, return_exceptions=True)


async def test_multi_utterance_coding_problem_reaches_gemini_complete(
    sessions, session_id, retriever
):
    """'Given an array, find two numbers...' [pause] '...whose sum equals a
    target value.' must reach Gemini as one complete problem, not two."""
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Given an array of integers, I want you to find two numbers",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await live.on_transcript(
        "whose sum equals a target value", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    assert llm.prompts
    final_prompt = llm.prompts[-1]
    assert "find two numbers" in final_prompt
    assert "sum equals a target value" in final_prompt


async def test_constraint_added_immediately_reaches_gemini(sessions, session_id, retriever):
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)

    await live.on_transcript(
        "Find duplicate elements in an array.", TranscriptSource.LOOPBACK, is_final=True
    )
    await live.on_transcript(
        "But you cannot use extra space.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    assert "duplicate elements" in llm.prompts[-1]
    assert "cannot use extra space" in llm.prompts[-1]


async def test_correction_to_a_different_coding_problem(sessions, session_id, retriever):
    """The old problem's answer must be cancelled; only the corrected
    problem's answer survives."""
    llm = SlowStreamingLLM(chunk_delay=0.02)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Write a function to reverse a string.", TranscriptSource.LOOPBACK, is_final=True
    )
    await live.on_transcript(
        "No, actually reverse a linked list.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    completed = collector.of(EventType.ANSWER_COMPLETED)
    assert len(completed) == 1
    turns = sessions.get_turns(session_id)
    answered = [t for t in turns if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1
    assert "linked list" in answered[0].question
    assert "reverse a string" not in answered[0].question


async def test_topic_change_mid_problem_does_not_contaminate_the_new_one(
    sessions, session_id, retriever
):
    llm = SlowStreamingLLM(chunk_delay=0.02)
    live = build(sessions, session_id, retriever, llm)

    await live.on_transcript(
        "Find the longest substring without repeating characters.",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await live.on_transcript(
        "Actually, let's do longest palindromic substring instead.",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await drain(live)

    answered = [t for t in sessions.get_turns(session_id) if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1
    assert "palindromic" in answered[0].question
    assert "repeating characters" not in answered[0].question


async def test_coding_followup_uses_previous_conversation(sessions, session_id, retriever):
    """'Can you do it in O(n)?' after 'How would you solve two sum?' must be
    answerable using the previous Q&A, which SessionMemory already threads
    into every prompt -- no coding-specific memory needed."""
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "How would you solve two sum?", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    await live.on_transcript(
        "Can you do it in O(n)?", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert len(detected) == 2
    assert "two sum" in llm.prompts[-1]  # prior question, via conversation history
    assert "O(n)" in detected[1].data["question"]


async def test_edge_case_followup_is_treated_as_a_question(sessions, session_id, retriever):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Find duplicate elements in an array.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)
    await live.on_transcript(
        "What happens if the array is empty?", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)

    assert len(collector.of(EventType.QUESTION_DETECTED)) == 2


async def test_debugging_followup_on_verbal_coding_description(sessions, session_id, retriever):
    """No literal code was provided -- the interviewer described the problem
    verbally. This must still route as a real question, not be rejected."""
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "I have a loop inside another loop and it's timing out. "
        "How would you optimize this?",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await drain(live)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert detected
    assert detected[0].data["classification"]["category"] != "UNKNOWN"


async def test_acknowledgement_after_coding_answer_is_rejected_not_answered(
    sessions, session_id, retriever
):
    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0))
    collector = Collector()
    live.subscribe(collector)

    await live.on_transcript(
        "Write a function to reverse a linked list.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)
    await live.on_transcript("Okay.", TranscriptSource.LOOPBACK, is_final=True)
    await drain(live)

    assert len(collector.of(EventType.QUESTION_DETECTED)) == 1
    assert collector.of(EventType.QUESTION_REJECTED)
