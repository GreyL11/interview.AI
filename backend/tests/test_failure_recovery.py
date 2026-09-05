"""What happens when the provider, the classifier or the session gives out.

An interview does not stop because a network did. Every case here asserts the
same three invariants, because failing any one of them ends the session for the
candidate rather than just the turn:

    the turn reaches a terminal state (the UI never sits in "answering")
    the next question is still processable
    nothing from the broken turn reaches the UI afterwards

The fourth invariant is observability: a turn that never produced a token still
has to appear in the latency population, or the only latencies ever measured
are the successful ones -- which is exactly the population that hides a
provider problem.
"""

import asyncio

import pytest

from app.core.metrics import LatencyTrace
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.schemas.answer import Answer
from app.sessions.schemas import TranscriptSource
from tests.fakes import BrokenLLM, SlowStreamingLLM
from tests.replay_harness import ReplayHarness
from tests.test_understanding_session import CountingCompleter, reply

asyncio_test = pytest.mark.asyncio

TERMINAL = {EventType.ANSWER_COMPLETED, EventType.ANSWER_ERROR, EventType.ANSWER_CANCELLED}


class KindedLLM(LLMClient):
    """Fails mid-stream with a classified provider error, after optionally
    streaming some text first -- the interrupted-stream case, which is not the
    same as failing to open one."""

    def __init__(self, kind: LLMErrorKind, chunks_before_failure: int = 0) -> None:
        self.kind = kind
        self.chunks_before_failure = chunks_before_failure
        self.prompts: list[str] = []

    async def generate_answer(self, prompt: str) -> Answer:
        raise LLMError("failed", kind=self.kind)

    async def stream_answer(self, prompt: str):
        self.prompts.append(prompt)
        payload = '{"summary": "partial text that already reached the screen'
        for i in range(self.chunks_before_failure):
            yield payload[i * 12:(i + 1) * 12]
        raise LLMError("The provider is unavailable.", kind=self.kind)


class HangingLLM(LLMClient):
    """Opens a stream and never produces anything -- what a session close or a
    supersede has to be able to interrupt."""

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0
        self.prompts: list[str] = []
        #: Set the moment the provider is actually reached, so a test can wait
        #: on the fact instead of guessing how many event-loop turns it takes.
        self.entered = asyncio.Event()

    async def generate_answer(self, prompt: str) -> Answer:
        raise NotImplementedError

    async def stream_answer(self, prompt: str):
        self.prompts.append(prompt)
        self.started += 1
        self.entered.set()
        try:
            await asyncio.Event().wait()
            yield ""  # pragma: no cover
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def harness(llm, monkeypatch, completer=None) -> ReplayHarness:
    h = ReplayHarness(
        llm=llm, memory_factory=SqliteSessionMemory, monkeypatch=monkeypatch
    )
    if completer is not None:
        from app.realtime.question_understanding import QuestionUnderstander

        h.live._understander = QuestionUnderstander(completer)
    return h


async def say(h, text, now):
    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    await h.settle()


def phases(h) -> list[EventType]:
    return [e.type for e in h.result.events if e.type in TERMINAL]


# ================================================== 17. provider failure


@pytest.mark.parametrize(
    "kind",
    [
        LLMErrorKind.TIMEOUT,
        LLMErrorKind.NETWORK,
        LLMErrorKind.SERVER,
        LLMErrorKind.AUTH,
        LLMErrorKind.RATE_LIMIT,
        LLMErrorKind.MALFORMED,
    ],
)
@asyncio_test
async def test_every_provider_failure_terminates_the_turn(monkeypatch, kind):
    h = harness(KindedLLM(kind), monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)

        assert phases(h) == [EventType.ANSWER_ERROR], (
            f"{kind.value} left the turn un-terminated"
        )
        errors = h.result.of(EventType.ANSWER_ERROR)
        assert errors[0].data["code"] == "LLMError"
        assert errors[0].data["message"]
    finally:
        h.dispose()


@asyncio_test
async def test_a_failed_turn_does_not_poison_the_next_one(monkeypatch):
    """The whole point: one bad turn costs one answer, not the interview."""
    h = harness(BrokenLLM(), monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)
        assert phases(h) == [EventType.ANSWER_ERROR]

        # Provider recovers.
        h.live._llm = SlowStreamingLLM(chunk_delay=0)
        await say(h, "What is a hash map?", now=200.0)

        assert phases(h) == [EventType.ANSWER_ERROR, EventType.ANSWER_COMPLETED]
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "the recovered turn did not complete"
    finally:
        h.dispose()


@asyncio_test
async def test_a_stream_that_breaks_after_partial_text_still_terminates(monkeypatch):
    """Interrupted mid-stream is not the same as failing to open: text has
    already reached the screen, and the turn still has to end."""
    h = harness(KindedLLM(LLMErrorKind.NETWORK, chunks_before_failure=3), monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)

        assert h.result.of(EventType.ANSWER_DELTA), "no partial text streamed"
        assert phases(h) == [EventType.ANSWER_ERROR]
    finally:
        h.dispose()


@asyncio_test
async def test_a_provider_message_carries_no_secret(monkeypatch):
    """The message shown to the user is the provider layer's own classified
    text, never a key, URL or request body."""
    h = harness(KindedLLM(LLMErrorKind.AUTH), monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)

        message = h.result.of(EventType.ANSWER_ERROR)[0].data["message"]
        for leak in ("gsk_", "Bearer", "api.groq.com", "Authorization"):
            assert leak not in message
    finally:
        h.dispose()


# ================================================== 18/19. cancellation


@asyncio_test
async def test_a_superseded_turn_cannot_emit_after_its_replacement(monkeypatch):
    llm = SlowStreamingLLM(chunk_delay=0.01)
    h = harness(llm, monkeypatch)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What is a covering index?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        # Do not settle: the first answer is mid-stream when the second lands.
        await say(h, "What is a hash map?", now=101.0)

        turns = {e.data.get("question", "") for e in h.result.of(EventType.QUESTION_DETECTED)}
        assert len(turns) == 2

        first, second = sorted(
            e.turn_id for e in h.result.of(EventType.QUESTION_DETECTED)
        )
        # Nothing from the stale turn may arrive after the live turn started.
        live_started = next(
            i for i, e in enumerate(h.result.events)
            if e.type is EventType.ANSWER_STARTED and e.turn_id == second
        )
        late = [
            e for e in h.result.events[live_started:]
            if e.turn_id == first and e.type is EventType.ANSWER_DELTA
        ]
        assert late == [], f"stale turn emitted {len(late)} late deltas"
        # Deliberately not asserting that the provider *observed* the
        # cancellation: if the stale task had not yet been suspended inside
        # `stream_answer`, it is cancelled before entering it and the fake's
        # counter stays 0. Either way no late delta may reach the UI, which is
        # the invariant that matters and is asserted above.
        assert len(h.result.of(EventType.ANSWER_COMPLETED)) == 1
    finally:
        h.dispose()


@asyncio_test
async def test_a_stale_turn_cannot_mutate_current_question_state(monkeypatch):
    llm = SlowStreamingLLM(chunk_delay=0.01)
    h = harness(llm, monkeypatch)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What is a covering index?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await say(h, "What is a hash map?", now=101.0)

        detected = sorted(e.turn_id for e in h.result.of(EventType.QUESTION_DETECTED))
        assert h.live._current_turn_id == detected[-1], (
            "a stale turn overwrote the current question"
        )
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert [e.turn_id for e in completed] == [detected[-1]]
    finally:
        h.dispose()


@asyncio_test
async def test_a_question_superseded_during_understanding_never_answers(monkeypatch):
    """Cancellation is not a classifier failure: `understand` re-raises it
    rather than degrading to a fallback, which is what stops a stale turn
    from proceeding to generate."""
    entered = asyncio.Event()

    class BlockingCompleter:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.cancelled = 0

        async def complete_json(self, prompt, *, model, timeout_seconds):
            self.prompts.append(prompt)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return reply()

    completer = BlockingCompleter()
    h = harness(SlowStreamingLLM(chunk_delay=0), monkeypatch, completer)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What is a covering index?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await asyncio.wait_for(entered.wait(), timeout=2)

        # Second question arrives while the first is still classifying.
        h.live._understander._completer = CountingCompleter(reply())
        await say(h, "What is a hash map?", now=101.0)

        assert completer.cancelled == 1, "the stale classification kept running"
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "the stale turn produced an answer"
        assert len(h.llm.prompts) == 1, "the stale turn reached the answer provider"
    finally:
        h.dispose()


# ============================================ 20/21. session close mid-flight


@asyncio_test
async def test_closing_during_understanding_leaves_nothing_running(monkeypatch):
    entered = asyncio.Event()

    class BlockingCompleter:
        def __init__(self) -> None:
            self.cancelled = 0

        async def complete_json(self, prompt, *, model, timeout_seconds):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return reply()

    completer = BlockingCompleter()
    h = harness(SlowStreamingLLM(chunk_delay=0), monkeypatch, completer)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What is a covering index?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await asyncio.wait_for(entered.wait(), timeout=2)

        await h.live.close()

        assert completer.cancelled == 1
        assert h.result.of(EventType.ANSWER_COMPLETED) == []
        assert h.live._task is None or h.live._task.done()
    finally:
        h.dispose()


@asyncio_test
async def test_closing_during_a_provider_stream_leaves_nothing_running(monkeypatch):
    llm = HangingLLM()
    h = harness(llm, monkeypatch)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "What is a covering index?", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await asyncio.wait_for(llm.entered.wait(), timeout=2)
        assert llm.started == 1, "the stream never opened"

        await h.live.close()

        assert llm.cancelled == 1, "the provider stream outlived the session"
        assert h.live._task is None or h.live._task.done()
        assert h.result.of(EventType.ANSWER_COMPLETED) == []
    finally:
        h.dispose()


@asyncio_test
async def test_closing_an_idle_session_is_harmless(monkeypatch):
    h = harness(SlowStreamingLLM(chunk_delay=0), monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)
        await h.live.close()
        await h.live.close()  # idempotent
    finally:
        h.dispose()


# ============================================== 22. the latency population


def test_a_turn_that_never_produced_a_token_still_reports(monkeypatch):
    """The gap this closes: only `emit_first_token` used to write a trace, so
    a timed-out or superseded turn left no line and vanished from every
    latency aggregate."""
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.core.metrics.log_metric",
        lambda name, **fields: captured.append((name, fields)),
    )
    trace = LatencyTrace(speech_end_at=0.0, session_id="s", utterance_id=7)
    trace.emit_terminal(3, "llm_timeout")

    assert len(captured) == 1
    name, fields = captured[0]
    assert name == "question_latency_trace"
    assert fields["outcome"] == "llm_timeout"
    assert fields["question_id"] == 3
    assert fields["utterance_id"] == 7


def test_a_turn_reports_exactly_once(monkeypatch):
    """A cancelled turn that had already streamed text has its story told by
    the first-token line; a second line would double count it."""
    captured: list[str] = []
    monkeypatch.setattr(
        "app.core.metrics.log_metric",
        lambda name, **fields: captured.append(fields.get("outcome", "")),
    )
    trace = LatencyTrace(speech_end_at=0.0)
    trace.emit_first_token(1)
    trace.emit_terminal(1, "cancelled")

    assert captured == ["first_token"]


@asyncio_test
async def test_a_failed_turn_reports_its_outcome(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.core.metrics.log_metric",
        lambda name, **fields: (
            captured.append(fields) if name == "question_latency_trace" else None
        ),
    )
    h = harness(KindedLLM(LLMErrorKind.TIMEOUT), monkeypatch)
    try:
        trace = LatencyTrace(speech_end_at=100.0, session_id=h.session_id)
        await h.live.consider(
            "What is a covering index?", TranscriptSource.LOOPBACK, trace, now=100.0
        )
        await h.settle()

        outcomes = [f["outcome"] for f in captured]
        assert outcomes == ["llm_timeout"], outcomes
    finally:
        h.dispose()
