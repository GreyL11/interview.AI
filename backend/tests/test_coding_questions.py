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


# ---------------------------------------------------- coding follow-up chains
# The mechanism under test is the one that already exists: SqliteSessionMemory
# threads previous ANSWERED turns into every prompt, and a narrow follow-up
# classifies away from Category.CODING so it gets the generic schema rather
# than being told to regenerate approach + code + complexity + edge cases.


async def _ask(live, text):
    await live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True)
    await drain(live)


@pytest.mark.parametrize(
    "problem,followups",
    [
        (
            "Find two numbers in an array whose sum equals a target.",
            ["What is the time complexity?", "What about space complexity?",
             "Can we improve the complexity?"],
        ),
        ("Find duplicates in an array.", ["Can you optimize it?"]),
        (
            "Reverse a linked list.",
            ["What happens if the list is empty?", "What about a single node?",
             "What edge cases should we consider?"],
        ),
        (
            "Find the longest substring without repeating characters.",
            ["Is there another approach?", "What if we use a sliding window?"],
        ),
        (
            "Write a function for two sum.",
            ["Explain the code.", "Why are we using a hash map?"],
        ),
        (
            "Find two numbers whose sum equals target.",
            ["What if the array is sorted?", "What if duplicates are allowed?",
             "What if we cannot use extra memory?"],
        ),
    ],
)
async def test_coding_followups_keep_the_original_problem_in_context(
    sessions, session_id, retriever, problem, followups
):
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))
    collector = Collector()
    live.subscribe(collector)

    await _ask(live, problem)
    for followup in followups:
        await _ask(live, followup)

    detected = collector.of(EventType.QUESTION_DETECTED)
    assert len(detected) == 1 + len(followups)

    # Every follow-up prompt carries the original problem via conversation
    # history, and asks only its own narrow question.
    anchor = problem.rstrip(".").split()[-1]
    for followup, prompt in zip(followups, llm.prompts[1:]):
        assert anchor in prompt, followup
        # The detector normalises terminal punctuation, so compare on the stem.
        assert followup.rstrip(".?") in prompt, followup


async def test_a_coding_followup_does_not_request_the_full_coding_schema(
    sessions, session_id, retriever
):
    """"What is the time complexity?" must not be handed the CODING schema --
    that hint instructs the model to emit approach + code + complexity + edge
    cases, i.e. to restate the whole solution the candidate already has."""
    from app.llm.prompts import CODING_SCHEMA_HINT

    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))

    await _ask(live, "Find two numbers in an array whose sum equals a target.")
    assert CODING_SCHEMA_HINT in llm.prompts[0]

    await _ask(live, "What is the time complexity?")
    assert CODING_SCHEMA_HINT not in llm.prompts[-1]


async def test_rapid_followup_chain_keeps_one_answer_per_question(
    sessions, session_id, retriever
):
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))
    collector = Collector()
    live.subscribe(collector)

    for text in ("Find two sum.", "What's the complexity?", "Can we optimize it?",
                 "What if the array is sorted?"):
        await _ask(live, text)

    assert len(collector.of(EventType.ANSWER_COMPLETED)) == 4
    answered = [t for t in sessions.get_turns(session_id) if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 4
    # Last prompt sees the whole chain, not just the immediately previous turn.
    assert "two sum" in llm.prompts[-1]
    assert "sorted" in llm.prompts[-1]


@pytest.mark.parametrize(
    "switch",
    [
        "Actually, let's solve longest palindromic substring.",
        "Actually, let's do longest palindromic substring instead.",
        "Actually, let's switch to longest palindromic substring.",
    ],
)
async def test_topic_change_phrasings_do_not_carry_the_old_problem(
    sessions, session_id, retriever, switch
):
    """Measured gap: only the "let's do ... instead" phrasing dropped the
    previous problem; "let's solve"/"let's switch to" merged it in."""
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)

    await live.on_transcript("Find two sum.", TranscriptSource.LOOPBACK, is_final=True)
    await live.on_transcript(switch, TranscriptSource.LOOPBACK, is_final=True)
    await drain(live)

    answered = [t for t in sessions.get_turns(session_id) if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 1
    assert "palindromic" in answered[0].question
    assert "two sum" not in answered[0].question.lower()


async def test_followups_after_a_topic_change_reference_the_new_topic(
    sessions, session_id, retriever
):
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))

    await _ask(live, "Find two sum.")
    await _ask(live, "Actually, let's solve longest palindromic substring.")
    await _ask(live, "What is the time complexity?")

    assert "palindromic" in llm.prompts[-1]


@pytest.mark.parametrize("ack", ["Okay.", "Got it.", "That makes sense."])
async def test_acknowledgements_after_a_coding_answer_never_reach_the_llm(
    sessions, session_id, retriever, ack
):
    llm = SlowStreamingLLM(chunk_delay=0)
    live = build(sessions, session_id, retriever, llm)
    collector = Collector()
    live.subscribe(collector)

    await _ask(live, "Find two numbers whose sum equals a target.")
    prompts_before = len(llm.prompts)
    await _ask(live, ack)

    assert len(llm.prompts) == prompts_before
    assert len(collector.of(EventType.QUESTION_DETECTED)) == 1
    assert collector.of(EventType.QUESTION_REJECTED)


async def test_an_interrupted_turn_is_excluded_from_followup_context(
    sessions, session_id, retriever
):
    """A superseded answer is INTERRUPTED, and SqliteSessionMemory reads only
    ANSWERED turns -- so a half-finished answer must not leak into the prompt
    for the question that replaced it."""
    llm = SlowStreamingLLM(chunk_delay=0.05)
    live = build(sessions, session_id, retriever, llm, memory=SqliteSessionMemory(sessions))

    await live.on_transcript(
        "Find the longest substring without repeating characters.",
        TranscriptSource.LOOPBACK, is_final=True,
    )
    await live.on_transcript(
        "Actually, let's solve merge intervals.", TranscriptSource.LOOPBACK, is_final=True
    )
    await drain(live)
    await _ask(live, "What is the time complexity?")

    statuses = {t.status for t in sessions.get_turns(session_id)}
    assert TurnStatus.ANSWERED in statuses
    assert "repeating characters" not in llm.prompts[-1]
