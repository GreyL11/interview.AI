"""Understanding wired into LiveSession: call counts and prompt assembly.

The router's own behaviour is covered in `test_question_understanding.py`.
This file asserts the integration invariants -- how many provider calls a turn
costs, what reaches the answer prompt, and that a superseded turn cannot
answer.

No real Groq: the answer LLM is the existing fake, and the classifier is a
scripted completer injected into `QuestionUnderstander`.
"""

import asyncio
import json

import pytest

from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.realtime.question_understanding import (
    QuestionUnderstander,
    Relationship,
    UnderstandingSource,
)
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayHarness

asyncio_test = pytest.mark.asyncio

SCHEMA = "customers(customer_id, name)\norders(order_id, customer_id, order_date)"
SQL = "SELECT * FROM orders WHERE order_date < now() - interval '90 days';"


class CountingCompleter:
    """Scripted classifier that counts calls and can be made slow or broken."""

    def __init__(self, *responses: str, delay: float = 0.0, error: Exception | None = None):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.delay = delay
        self.error = error

    async def complete_json(self, prompt: str, *, model: str, timeout_seconds: float) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if not self._responses:
            return reply()
        return self._responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def reply(**overrides) -> str:
    body = {
        "intent": "conceptual",
        "relationship": "new_question",
        "topic": "t",
        "domain": "d",
        "task": "explain",
        "constraints": [],
        "requested_output": ["explanation"],
        "entities": [],
        "needs_previous_context": False,
        "needs_previous_answer": False,
        "needs_previous_code": False,
        "needs_attachments": False,
        "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body)


def harness_with(completer, monkeypatch, chunk_delay: float = 0.0) -> ReplayHarness:
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=chunk_delay),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    h.live._understander = QuestionUnderstander(completer)
    return h


async def say(h, text, now, settle=True):
    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    if settle:
        await h.settle()


def prompts(h) -> list[str]:
    return h.llm.prompts


# ============================================ AL, AQ: one call per turn


@asyncio_test
async def test_a_complete_question_costs_one_understanding_and_one_answer(monkeypatch):
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert completer.calls == 1, "understanding must run exactly once"
        assert len(prompts(h)) == 1, "one answer request"
    finally:
        h.dispose()


@asyncio_test
async def test_a_partial_transcript_never_calls_understanding(monkeypatch):
    """AJ: partials are display-only and never reach `ask()`."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_transcript(
            "What is Azure Data", TranscriptSource.LOOPBACK, is_final=False, now=100.0
        )
        await h.settle()

        assert completer.calls == 0
        assert prompts(h) == []
    finally:
        h.dispose()


@asyncio_test
async def test_an_incomplete_fragment_never_calls_understanding(monkeypatch):
    """AK: the fragment is held by the deterministic layer, which is upstream
    of this module entirely."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "Can you explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )
        # Deliberately not settled: the turn is still accumulating.
        assert completer.calls == 0
        assert prompts(h) == []
    finally:
        h.live._abandon_accumulation()
        h.dispose()


@asyncio_test
async def test_a_multi_fragment_question_costs_one_understanding_call(monkeypatch):
    """N fragments assemble into one turn upstream, so this layer sees one."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        for i, fragment in enumerate([
            "Can you explain", "how dependency injection works", "in FastAPI?",
        ]):
            await h.live.on_speech_start(i + 1)
            await h.live.on_transcript(
                fragment, TranscriptSource.LOOPBACK, is_final=True, now=100.0 + i * 0.4
            )
        await h.settle()

        assert completer.calls == 1, "one turn, one understanding call"
        assert len(prompts(h)) == 1
    finally:
        h.dispose()


@asyncio_test
async def test_an_acknowledgement_calls_neither_model(monkeypatch):
    """U / AP: the deterministic layer rejects filler before `ask()`, so the
    understanding call is never even reached -- item 28's shortcut."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        for filler in ("Okay.", "Right.", "Got it.", "That makes sense."):
            await say(h, filler, now=100.0)

        assert completer.calls == 0
        assert prompts(h) == []
        assert h.result.of(EventType.QUESTION_REJECTED)
    finally:
        h.dispose()


@asyncio_test
async def test_acknowledgement_plus_question_is_a_real_turn(monkeypatch):
    """V: "Okay, now how would you handle failures?" is not filler."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Okay, now how would you handle failures?", now=100.0)

        assert completer.calls == 1
        assert len(prompts(h)) == 1
    finally:
        h.dispose()


# ============================================== AM, AN, AO: invariants


@asyncio_test
async def test_understanding_never_rewrites_the_answered_question(monkeypatch):
    """AM / item 20. The classifier is told to return a paraphrase and an
    entirely different exact_question; neither may reach the answer prompt."""
    hostile = json.loads(reply())
    hostile["exact_question"] = "Explain DI. (rewritten by the classifier)"
    hostile["task"] = "summarised task"
    completer = CountingCompleter(json.dumps(hostile))
    h = harness_with(completer, monkeypatch)
    question = "Can you explain how dependency injection works in FastAPI?"
    try:
        await say(h, question, now=100.0)

        prompt = prompts(h)[0]
        assert f"CURRENT INTERVIEWER QUESTION (TECHNICAL_KNOWLEDGE): {question}" in prompt
        assert "rewritten by the classifier" not in prompt
    finally:
        h.dispose()


@asyncio_test
async def test_exact_question_and_exact_attachment_both_reach_the_prompt(monkeypatch):
    """AN, and the success criterion from the brief."""
    completer = CountingCompleter(reply(intent="query", needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    question = "Can you write a SQL query to find customers who haven't placed an order in the last 90 days?"
    try:
        await h.live.on_context_attached(kind="table", content=SCHEMA, now=100.0)
        await h.settle()
        await say(h, question, now=100.5)

        assert len(prompts(h)) == 1
        prompt = prompts(h)[0]
        assert question in prompt, "exact question missing"
        assert SCHEMA in prompt, "exact schema missing"
        assert "MATERIAL THE INTERVIEWER PROVIDED" in prompt
        assert "HOW THIS QUESTION WAS UNDERSTOOD" in prompt
        # The understanding is subordinate to the evidence, and says so.
        assert "the words are correct" in prompt
    finally:
        h.dispose()


@asyncio_test
async def test_the_classifier_never_sees_attachment_bytes(monkeypatch):
    """AO-adjacent: attachments do not go through the classifier prompt, and
    never through candidate RAG."""
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_context_attached(kind="sql", content=SQL, now=100.0)
        await h.settle()
        await say(h, "How would you optimize this query?", now=100.5)

        assert completer.calls == 1
        assert SQL not in completer.prompts[0], "attachment leaked to classifier"
        assert "sql" in completer.prompts[0], "classifier not told material exists"
        # It does reach the *answer* prompt, verbatim.
        assert SQL in prompts(h)[0]
    finally:
        h.dispose()


@asyncio_test
async def test_attachment_content_never_enters_transcript_or_rag(monkeypatch):
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_context_attached(kind="table", content=SCHEMA, now=100.0)
        await h.settle()

        transcripts = h.result.of(EventType.TRANSCRIPT_FINAL)
        assert all(SCHEMA not in e.data.get("text", "") for e in transcripts)
        # MockRetriever is the RAG side; nothing was ingested into it.
        assert prompts(h) == []
    finally:
        h.dispose()


# ================================= context selection: follow-ups & chains


@asyncio_test
async def test_a_follow_up_receives_previous_conversation(monkeypatch):
    completer = CountingCompleter(
        reply(),
        reply(relationship="follow_up", needs_previous_context=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Redis?", now=100.0)
        await say(h, "Why is it fast?", now=110.0)

        assert completer.calls == 2, "one understanding call per turn"
        assert len(prompts(h)) == 2, "one answer call per turn"
        assert "Redis" in prompts(h)[1], "follow-up lost previous conversation"
    finally:
        h.dispose()


@asyncio_test
async def test_an_unrelated_new_question_does_not_inherit_old_context(monkeypatch):
    """AU: relationship drives selection, so a fresh subject starts clean."""
    completer = CountingCompleter(
        reply(),
        reply(relationship="new_question", needs_previous_context=False),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "How does Kafka handle partitions?", now=100.0)
        await say(h, "What is Azure Databricks?", now=200.0)

        assert "Kafka" not in prompts(h)[1], "unrelated context polluted the turn"
        assert "What is Azure Databricks?" in prompts(h)[1]
    finally:
        h.dispose()


@asyncio_test
async def test_a_long_follow_up_chain_has_no_count_limit(monkeypatch):
    """K / L / AR: ten follow-ups, ten turns, ten distinct question ids."""
    completer = CountingCompleter(
        *([reply()] + [reply(relationship="follow_up", needs_previous_context=True)] * 10)
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "How would you design an Azure Databricks pipeline?", now=100.0)
        for i, follow_up in enumerate([
            "Why Databricks?", "What if the data doubles?",
            "How would you handle failures?", "What about cost?",
            "How would you monitor it?", "And security?",
            "What about schema drift?", "How would you test it?",
            "What about backfills?", "And disaster recovery?",
        ]):
            await say(h, follow_up, now=110.0 + i * 10)

        assert completer.calls == 11
        assert len(prompts(h)) == 11
        detected = h.result.of(EventType.QUESTION_DETECTED)
        ids = [e.turn_id for e in detected]
        assert len(ids) == 11
        assert len(set(ids)) == 11, "turn ids merged or leaked across follow-ups"
    finally:
        h.dispose()


@asyncio_test
async def test_coding_progression_keeps_the_underlying_problem(monkeypatch):
    """AS / item 8: each step is its own turn, but the original problem stays
    reachable through conversation history."""
    completer = CountingCompleter(
        reply(intent="coding"),
        reply(relationship="new_method", needs_previous_context=True,
              needs_previous_code=True),
        reply(relationship="constraint_change", needs_previous_context=True,
              needs_previous_code=True),
        reply(relationship="follow_up", needs_previous_context=True),
        reply(relationship="new_implementation", needs_previous_context=True,
              needs_previous_code=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Write a Python function to find duplicate numbers.", now=100.0)
        await say(h, "Now show me another way using a set.", now=120.0)
        await say(h, "Can you do it without extra space?", now=140.0)
        await say(h, "What's the complexity?", now=160.0)
        await say(h, "Okay, now implement it in Java.", now=180.0)

        assert completer.calls == 5
        assert len(prompts(h)) == 5, "each step is its own answer turn"
        # The original problem is still present by the last step.
        assert "duplicate numbers" in prompts(h)[-1], "lost the underlying problem"
        # And the last step's own words are there too, unmerged.
        assert "implement it in Java" in prompts(h)[-1]
    finally:
        h.dispose()


@asyncio_test
async def test_a_repeated_question_still_gets_an_answer(monkeypatch):
    """AT / item 23: duplicate means "same question", not "suppress"."""
    completer = CountingCompleter(
        reply(),
        reply(relationship="duplicate", needs_previous_context=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)
        await say(h, "What is Azure Databricks?", now=200.0)

        assert len(prompts(h)) == 2, "a repeat must still be answered"
        assert completer.calls == 2
    finally:
        h.dispose()


# ============================== AE, AF, AH: failure and staleness


@asyncio_test
async def test_a_classifier_timeout_still_produces_exactly_one_answer(monkeypatch):
    """AE + item 27: a failure must not cost a duplicate answer call."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "question_understanding_timeout_ms", 10)
    completer = CountingCompleter(reply(), delay=0.4)
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert completer.calls == 1
        assert len(prompts(h)) == 1, "fallback caused a duplicate answer call"
        assert h.result.of(EventType.ANSWER_COMPLETED)
    finally:
        h.dispose()


@asyncio_test
async def test_a_classifier_provider_error_still_answers(monkeypatch):
    """AF: never surfaced as an interview failure."""
    completer = CountingCompleter(error=RuntimeError("groq down"))
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert len(prompts(h)) == 1
        assert h.result.of(EventType.ANSWER_COMPLETED)
        assert not h.result.of(EventType.ANSWER_ERROR)
    finally:
        h.dispose()


@asyncio_test
async def test_malformed_classifier_output_still_answers(monkeypatch):
    """AD at the session level."""
    completer = CountingCompleter("{not json")
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert len(prompts(h)) == 1
        assert h.result.of(EventType.ANSWER_COMPLETED)
        # The fallback path keeps prior behaviour: history is not dropped.
        assert "HOW THIS QUESTION WAS UNDERSTOOD" in prompts(h)[0]
    finally:
        h.dispose()


@asyncio_test
async def test_a_superseded_turn_produces_no_answer(monkeypatch):
    """AH / item 19: the stale understanding is cancelled with its turn and
    cannot come back to trigger an answer."""
    completer = CountingCompleter(reply(), reply(), delay=0.15)
    h = harness_with(completer, monkeypatch, chunk_delay=0.02)
    try:
        # First question starts, its understanding is still in flight.
        await say(h, "What is Azure Databricks?", now=100.0, settle=False)
        await asyncio.sleep(0.02)
        # A newer question supersedes it mid-classification.
        await say(h, "What is Delta Lake?", now=101.0, settle=False)
        await h.settle()

        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "the stale turn also answered"
        # The surviving answer is the newest question's.
        assert "Delta Lake" in prompts(h)[-1]
    finally:
        h.dispose()


def capture_metrics(monkeypatch) -> list[dict]:
    """Record metric lines emitted by the understanding layer.

    Patches the *importing* module: `question_understanding` does
    `from app.core.metrics import log_metric`, so it holds its own reference
    and patching `app.core.metrics` would not reach it.
    """
    import app.realtime.question_understanding as module

    seen: list[dict] = []
    original = module.log_metric

    def record(event, **fields):
        seen.append({"event": event, **fields})
        return original(event, **fields)

    monkeypatch.setattr(module, "log_metric", record)
    return seen


@asyncio_test
async def test_understanding_latency_is_reported_separately(monkeypatch):
    """Item 18/24: the classifier's cost must not hide inside another metric."""
    seen = capture_metrics(monkeypatch)
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        understanding_lines = [s for s in seen
                               if s["event"] == "question_understanding_completed"]
        assert len(understanding_lines) == 1
        line = understanding_lines[0]
        assert "understanding_latency_ms" in line
        assert line["classifier_success"] is True
        assert line["classifier_fallback"] is False
        assert line["relationship"] == Relationship.NEW_QUESTION.value
        # Metadata only -- no question text in the diagnostics.
        assert "Azure Databricks" not in str(line)
    finally:
        h.dispose()


def terminal_events(seen: list[dict]) -> list[str]:
    """The one terminal understanding event this turn produced."""
    names = {
        "question_understanding_completed",
        "question_understanding_timeout",
        "question_understanding_failed",
        "question_understanding_cancelled",
    }
    return [s["event"] for s in seen if s["event"] in names]


@asyncio_test
async def test_a_provider_error_is_reported_as_failed_not_completed(monkeypatch):
    """H11: distinct event names, so a failure can be counted without parsing
    flags off a shared line."""
    seen = capture_metrics(monkeypatch)
    completer = CountingCompleter(error=RuntimeError("boom"))
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert terminal_events(seen) == ["question_understanding_failed"]
        line = next(s for s in seen if s["event"] == "question_understanding_failed")
        assert line["classifier_fallback"] is True
        assert line["classifier_success"] is False
        assert line["source"] == UnderstandingSource.FALLBACK.value
        assert line["failure"].startswith("provider_error")
    finally:
        h.dispose()


@asyncio_test
async def test_a_timeout_is_reported_as_timeout(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "question_understanding_timeout_ms", 10)
    seen = capture_metrics(monkeypatch)
    completer = CountingCompleter(reply(), delay=0.4)
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        assert terminal_events(seen) == ["question_understanding_timeout"]
        line = next(s for s in seen if s["event"] == "question_understanding_timeout")
        assert line["classifier_timeout"] is True
    finally:
        h.dispose()


@asyncio_test
async def test_every_turn_emits_started_then_exactly_one_terminal(monkeypatch):
    seen = capture_metrics(monkeypatch)
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        started = [s for s in seen if s["event"] == "question_understanding_started"]
        assert len(started) == 1
        assert terminal_events(seen) == ["question_understanding_completed"]
        # No retry storm: one classification attempt, full stop.
        assert completer.calls == 1
    finally:
        h.dispose()


@asyncio_test
async def test_cancellation_is_not_reported_as_an_ordinary_fallback(monkeypatch):
    """Test 34. A superseded turn must be distinguishable from a classifier
    that failed -- one is normal control flow, the other is a degradation.

    Synchronised on an event rather than a sleep: cancelling a session task
    can land before the classifier is even entered (during the history read),
    which would make a sleep-based version pass or fail on timing luck.
    """
    from app.realtime.question_understanding import QuestionUnderstander

    seen = capture_metrics(monkeypatch)
    entered = asyncio.Event()

    class Blocking:
        async def complete_json(self, prompt, *, model, timeout_seconds):
            entered.set()
            await asyncio.Event().wait()   # never resolves; only cancellation ends it
            return reply()

    understander = QuestionUnderstander(Blocking())
    task = asyncio.create_task(
        understander.understand("slow question", session_id="s", question_id=1)
    )
    await entered.wait()          # the classifier is provably in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert terminal_events(seen) == ["question_understanding_cancelled"]
    cancelled = next(s for s in seen if s["event"] == "question_understanding_cancelled")
    # Not a fallback: no degraded understanding was produced or used.
    assert "classifier_fallback" not in cancelled
    assert "source" not in cancelled

# ==================================== hardening: anchor, re-ask, staging


@asyncio_test
async def test_the_thread_anchor_follows_new_questions_only(monkeypatch):
    """H4: a new question opens a thread; refinements of it do not move the
    anchor, which is what keeps the original problem reachable."""
    completer = CountingCompleter(
        reply(intent="coding"),                                   # opens a thread
        reply(relationship="new_method", needs_previous_code=True),
        reply(relationship="constraint_change"),
        reply(),                                                  # a new thread
    )
    h = harness_with(completer, monkeypatch)
    try:
        # The anchor holds the detector's normalised question -- the same
        # string the turns table stores and history renders as "Q: ...", which
        # is what makes anchor lookup in `select_context` match.
        anchor = "Write a Python function to find duplicate numbers?"
        await say(h, "Write a Python function to find duplicate numbers.", now=100.0)
        assert h.live._thread_anchor == anchor

        await say(h, "Now show me another way using a set.", now=120.0)
        assert h.live._thread_anchor == anchor, "a refinement moved the anchor"
        await say(h, "Can you do it without extra space?", now=140.0)
        assert h.live._thread_anchor == anchor

        # A genuinely new question opens a new thread.
        await say(h, "What is Azure Databricks?", now=300.0)
        assert h.live._thread_anchor == "What is Azure Databricks?"
    finally:
        h.dispose()


@asyncio_test
async def test_a_fallback_does_not_move_the_thread_anchor(monkeypatch):
    """Moving it on a guess would silently drop the original problem."""
    completer = CountingCompleter(reply(intent="coding"), "garbage not json")
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Write a function to find duplicates.", now=100.0)
        anchor = h.live._thread_anchor
        assert anchor == "Write a function to find duplicates?"

        await say(h, "Now do it in Java.", now=120.0)
        assert h.live._thread_anchor == anchor, "a fallback moved the anchor"
    finally:
        h.dispose()


@asyncio_test
async def test_a_six_step_coding_progression_keeps_the_original_problem(monkeypatch):
    """H4 end to end: six turns, six answers, and step six still carries the
    problem introduced in step one."""
    problem = "Write a Python function to find duplicate numbers."
    completer = CountingCompleter(
        reply(intent="coding"),
        reply(relationship="new_method", needs_previous_code=True),
        reply(relationship="constraint_change", needs_previous_code=True),
        reply(relationship="follow_up", needs_previous_context=True),
        reply(relationship="new_implementation", needs_previous_code=True),
        reply(relationship="new_implementation", needs_previous_code=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, problem, now=100.0)
        for i, step in enumerate([
            "Show me another way using a set.",
            "Can you do it without extra space?",
            "What's the complexity?",
            "Now implement it in Java.",
            "Now write it for a stream of integers.",
        ]):
            await say(h, step, now=120.0 + i * 20)

        assert completer.calls == 6, "one understanding call per step"
        assert len(prompts(h)) == 6, "each step is its own answer turn"

        final = prompts(h)[-1]
        assert "duplicate numbers" in final, "step six lost the original problem"
        assert "stream of integers" in final, "step six lost its own words"
        # Not every intervening answer is dragged along.
        assert final.count("Q: ") <= 4, "context selection dumped the whole thread"
    finally:
        h.dispose()


@asyncio_test
async def test_a_late_paste_reask_keeps_the_relationship_detail(monkeypatch):
    """H6: the re-ask must carry the deterministic detail, or a follow-up
    degrades into a new question whenever the classifier falls back."""
    # Classifier fails on both calls, so `detail` is the only thing that can
    # preserve the follow-up reading -- which is exactly the regression.
    completer = CountingCompleter("garbage", "garbage", "garbage")
    h = harness_with(completer, monkeypatch, chunk_delay=0.02)
    try:
        await say(h, "What is a hash map?", now=100.0)
        # A short follow-up: the deterministic layer marks it detail=follow_up.
        await say(h, "Why?", now=110.0, settle=False)
        assert h.live._live_question is not None
        assert h.live._live_question.detail == "follow_up", (
            "the live turn did not remember its relationship"
        )

        # A paste arrives while that follow-up is still answering.
        await h.live.on_context_attached(kind="table", content=SCHEMA, now=110.5)
        await h.settle()

        # Two questions were asked, so two answers is correct. What matters
        # is that the late paste did not add a third turn, and that the
        # surviving prompt carries the material.
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 2, [e.turn_id for e in completed]
        assert SCHEMA in prompts(h)[-1]
    finally:
        h.dispose()


@asyncio_test
async def test_an_unrelated_question_does_not_inherit_old_attachments(monkeypatch):
    """H5: attachment binding stays the backend's turn-scoped rule -- the
    classifier does not get to decide global relevance."""
    completer = CountingCompleter(reply(), reply())
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_context_attached(kind="sql", content=SQL, now=100.0)
        await h.settle()
        await say(h, "Explain this SQL.", now=100.5)
        assert SQL in prompts(h)[0]

        await say(h, "What is Azure?", now=200.0)

        assert SQL not in prompts(h)[1], "attachment leaked into an unrelated turn"
    finally:
        h.dispose()


@asyncio_test
async def test_each_stage_of_the_hot_path_is_timed_separately(monkeypatch):
    """H1/H11: history, understanding and selection each reported, so no
    cost hides inside another."""
    import app.realtime.session as session_module

    seen: list[dict] = []
    original = session_module.log_metric

    def record(event, **fields):
        seen.append({"event": event, **fields})
        return original(event, **fields)

    monkeypatch.setattr(session_module, "log_metric", record)
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)

        prepared = next(s for s in seen if s["event"] == "llm_request_prepared")
        for field in (
            "history_latency_ms", "understanding_latency_ms",
            "context_selection_ms", "relationship", "understanding_source",
        ):
            assert field in prepared, f"missing {field}"
        assert "Azure Databricks" not in str(prepared), "question text in metrics"
    finally:
        h.dispose()


@asyncio_test
async def test_retrieval_and_understanding_overlap(monkeypatch):
    """H1: the two slow independent calls run concurrently, not serially.

    Both legs are made to take a measurable amount of time; if they were
    serial the wall clock would be their sum rather than roughly the larger.
    """
    import time as time_module

    from app.retrieval.base import Retriever

    class SlowRetriever(Retriever):
        async def retrieve(self, question, knowledge_types=None, top_k=None):
            await asyncio.sleep(0.20)
            return []

    completer = CountingCompleter(reply(), delay=0.20)
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0),
        retriever=SlowRetriever(),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    h.live._understander = QuestionUnderstander(completer)
    try:
        # A RAG-routed category, so retrieval actually runs.
        started = time_module.monotonic()
        await say(h, "Tell me about your experience with Spark.", now=100.0)
        elapsed = time_module.monotonic() - started

        assert completer.calls == 1
        assert len(prompts(h)) == 1
        # Serial would be >= 0.40s; concurrent is ~0.20s. The bound is loose
        # so this cannot fail on a slow machine, but it still fails outright
        # if the two are sequenced.
        assert elapsed < 0.36, f"retrieval and understanding ran serially ({elapsed:.3f}s)"
    finally:
        h.dispose()
