"""The active thread versus the current-utterance premise.

These two are separate concepts and must stay separate:

* **Current-utterance premise** -- whatever the interviewer said in *this*
  breath before asking ("We have ten million rows. How would you speed this
  up?"). Scoped to one utterance by construction.
* **Active thread** -- the task the conversation is on. Lives in conversation
  history plus the thread anchor, selected by relationship via `ContextPlan`.

Conflating them is a real bug that happened: a merge prepends the previously
accepted question, and treating that prefix as a premise brought a *superseded*
question back as the premise of the question that replaced it. The sentence
scan dropping the merged prefix is the mechanism that prevents it, and the
premise field must not undo that.

So the pair of tests that matter here are the two halves:

    a related coding follow-up keeps the original problem  (via history)
    a superseded or unrelated question does not             (via nothing)

Both decided only by the relationship the classifier reported -- same
machinery, opposite outcomes, no keyword anywhere.
"""

import pytest

from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.schemas.answer import Answer
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayHarness
from tests.test_understanding_session import CountingCompleter, reply

asyncio_test = pytest.mark.asyncio

CODING_PROBLEM = "Find the longest substring without repeating characters."
CODE = "def longest(s):\n    seen = {}\n    return 0"


def harness_with(completer, monkeypatch, answer=None) -> ReplayHarness:
    h = ReplayHarness(
        llm=SlowStreamingLLM(answer=answer, chunk_delay=0),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    from app.realtime.question_understanding import QuestionUnderstander

    h.live._understander = QuestionUnderstander(completer)
    return h


def coding_answer() -> Answer:
    return Answer(
        summary="Slide a window and track last-seen indices.",
        approach=["expand", "shrink on repeat"],
        code=CODE,
        complexity={"time": "O(n)", "space": "O(k)"},
    )


async def say(h, text, now):
    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    await h.settle()


def prompts(h) -> list[str]:
    return h.llm.prompts


# ================================ half one: related follow-ups keep the task


@pytest.mark.parametrize(
    ("followup", "relationship", "flags"),
    [
        ("How would you implement that?", "new_implementation",
         {"needs_previous_context": True}),
        ("What is the time complexity?", "follow_up",
         {"needs_previous_code": True}),
        ("How would you optimize it?", "follow_up",
         {"needs_previous_context": True, "needs_previous_code": True}),
        ("What edge cases should we consider?", "follow_up",
         {"needs_previous_context": True}),
    ],
)
@asyncio_test
async def test_a_coding_followup_keeps_the_original_problem(
    monkeypatch, followup, relationship, flags
):
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="coding", relationship=relationship, **flags),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, CODING_PROBLEM, now=100.0)
        await say(h, followup, now=130.0)

        prompt = prompts(h)[1]
        assert "longest substring without repeating characters" in prompt, (
            f"{followup!r} lost the original problem"
        )
        assert followup.rstrip("?.") in prompt, "the follow-up's own words changed"
        # Through history, which is where the thread lives -- not spliced into
        # the question, which is what a premise would have done.
        assert "Previous Q&A" in prompt
        assert prompt.index("longest substring") < prompt.index(
            "CURRENT INTERVIEWER QUESTION"
        ), "the problem was injected into the question rather than its context"
    finally:
        h.dispose()


# ============================ half two: unrelated and superseded get nothing


@asyncio_test
async def test_an_unrelated_question_after_a_coding_problem_starts_clean(monkeypatch):
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="behavioral", relationship="new_question"),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, CODING_PROBLEM, now=100.0)
        await say(h, "Tell me about a time you missed a deadline.", now=400.0)

        prompt = prompts(h)[1]
        assert "longest substring" not in prompt, "the old task leaked forward"
        assert CODE not in prompt, "the old implementation leaked forward"
        assert "Previous Q&A" not in prompt
    finally:
        h.dispose()


def test_a_merged_prefix_never_becomes_a_later_premise():
    """The regression this pass introduced and fixed, guarded precisely.

    A merge prepends the previously accepted text, so `extract_interview_prompt`
    runs over both. Deriving the premise from that merged string made the
    earlier question the premise of a *later, unrelated* one. The invariant is
    exact and checkable at the detector: a question with no setup of its own
    has `effective_text == text`.
    """
    from app.realtime.question_detector import QuestionDetector

    detector = QuestionDetector()
    detector.inspect(CODING_PROBLEM, now=100.0)
    merged = detector.inspect(
        "Actually, let us solve merge intervals instead.", now=100.4
    )
    assert merged.accepted and merged.supersedes, "no merge happened to guard against"

    later = detector.inspect("How would you design a URL shortener?", now=200.0)
    assert later.accepted
    assert later.effective_text == later.text, (
        f"setup leaked into a later question: {later.effective_text!r}"
    )
    assert "longest substring" not in later.effective_text
    assert "merge intervals" not in later.effective_text


def test_a_premise_of_the_same_utterance_still_reaches_the_model():
    """The positive half of the same invariant, so the guard above cannot be
    satisfied by disabling premises altogether."""
    from app.realtime.question_detector import QuestionDetector

    detection = QuestionDetector().inspect(
        "We have ten million rows in the orders table. How would you speed up this join?",
        now=100.0,
    )
    assert detection.effective_text != detection.text
    assert "ten million rows" in detection.effective_text


@asyncio_test
async def test_correcting_the_problem_revises_it_without_rewriting_history(
    monkeypatch,
):
    """The revision is the question; the original stays reachable as context
    and neither transcript is altered."""
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="coding", relationship="correction",
              needs_previous_context=True),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, CODING_PROBLEM, now=100.0)
        await say(h, "Actually, make it the longest palindromic substring.",
                  now=130.0)

        prompt = prompts(h)[1]
        current = prompt[prompt.index("CURRENT INTERVIEWER QUESTION"):]
        assert "palindromic" in current, "the revision is not the question"
        assert "longest substring without repeating" in prompt, (
            "the original problem became unreachable"
        )
        assert "repeating characters" not in current, (
            "the superseded wording leaked into the question"
        )
    finally:
        h.dispose()


# ====================== retrieval is enrichment, not the answer itself


def _chunk(text: str, title: str = "CV") -> "RetrievedChunk":
    from app.documents.schemas import KnowledgeType, RetrievedChunk

    return RetrievedChunk(
        chunk_id="c1", document_id="d1", text=text, score=0.9,
        knowledge_type=KnowledgeType.RESUME, title=title,
    )


@asyncio_test
async def test_retrieval_success_reaches_the_prompt_as_background(monkeypatch):
    """The path the degradation below is a fallback *from*. Asserted first, so
    "retrieval fails safely" cannot be satisfied by retrieval never working:
    a retriever that always returned [] would pass every failure test here.
    """
    completer = CountingCompleter(reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.realtime.session.log_metric",
        lambda name, **fields: captured.append({"name": name, **fields}),
    )

    async def populated(question, **kwargs):
        return [_chunk("Built a Kafka ingestion pipeline at Acme.")]

    h.live._retriever.retrieve = populated
    try:
        # FOLLOW_UP-categorised, so retrieval is actually attempted.
        await say(h, "What about a single node?", now=100.0)

        prompt = prompts(h)[0]
        assert "Kafka ingestion pipeline at Acme" in prompt, "chunks never arrived"
        # Background, explicitly not part of the question -- retrieved
        # knowledge is the candidate's own material, weighed against what the
        # model knows, and must not read as interviewer instruction.
        assert "Relevant personal/knowledge-base context" in prompt
        assert prompt.index("Kafka ingestion") < prompt.index(
            "CURRENT INTERVIEWER QUESTION"
        )

        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1
        assert completed[0].data["context_found"] is True
        assert completed[0].data["retrieval_hits"], "hits not reported"
        assert not [m for m in captured if m["name"] == "retrieval_failed"]
        assert h.result.of(EventType.ANSWER_RETRIEVING), "no retrieving event"
    finally:
        h.dispose()


@asyncio_test
async def test_a_retrieval_failure_still_answers_the_question(monkeypatch):
    """A missing or unreadable index used to take the whole turn down. Because
    only RAG/FOLLOW_UP-routed questions retrieve, that meant every follow-up
    failing while every other question worked -- which is what made the
    original symptom so confusing to read."""
    completer = CountingCompleter(reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)

    async def exploding(question, **kwargs):
        raise RuntimeError("index is not readable")

    captured: list[dict] = []
    monkeypatch.setattr(
        "app.realtime.session.log_metric",
        lambda name, **fields: captured.append({"name": name, **fields}),
    )
    h.live._retriever.retrieve = exploding
    try:
        # FOLLOW_UP-categorised, so retrieval is actually attempted.
        await say(h, "What about a single node?", now=100.0)

        assert h.result.of(EventType.ANSWER_ERROR) == [], "a turn failed on retrieval"
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1
        assert completed[0].data["context_found"] is False

        # Observable, and by exception type only -- a store error can quote a
        # path or the query text.
        failures = [m for m in captured if m["name"] == "retrieval_failed"]
        assert len(failures) == 1, captured
        assert failures[0]["error"] == "RuntimeError"
        assert "index is not readable" not in str(failures[0])
    finally:
        h.dispose()


@asyncio_test
async def test_a_retrieval_failure_does_not_leak_store_detail(monkeypatch):
    completer = CountingCompleter(reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)

    async def exploding(question, **kwargs):
        raise RuntimeError("/home/user/.local/share/call-assistant/index.faiss")

    h.live._retriever.retrieve = exploding
    try:
        await say(h, "What about a single node?", now=100.0)
        for ev in h.result.events:
            assert "index.faiss" not in str(ev.data), "a store path reached the UI"
    finally:
        h.dispose()


@asyncio_test
async def test_a_cancelled_retrieval_still_cancels_the_turn(monkeypatch):
    """Degrading on failure must not swallow cancellation -- that is how a
    stale turn would go on to answer."""
    import asyncio

    completer = CountingCompleter(reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)
    entered = asyncio.Event()

    async def hanging(question, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        return []

    h.live._retriever.retrieve = hanging
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What about a single node?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await asyncio.wait_for(entered.wait(), timeout=2)

        await h.live.close()

        assert h.result.of(EventType.ANSWER_COMPLETED) == []
        assert h.live._task is None or h.live._task.done()
    finally:
        h.dispose()
