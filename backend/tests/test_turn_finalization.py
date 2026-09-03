"""Turn finalization: when is an interviewer question actually finished?

Three finalities are distinct, and this file is about the middle one:

    STT final  !=  interviewer turn final  !=  LLM question final

A Whisper final only says a VAD segment closed. These tests pin which
utterances go to the model immediately, which are held for a continuation,
and how many provider calls each conversation costs.

Deterministic by construction: every timestamp is an explicit speech-clock
value (`at_ms` / `now=`), never a sleep. `ReplayHarness` sets both hold
budgets to 1ms so a held question resolves on the next event-loop turn --
the *decision* to hold is asserted at the detector level, the *consequence*
(one provider call, not two) at the session level.
"""

import pytest

from app.core.config import settings
from app.realtime.events import EventType
from app.realtime.question_detector import Finality, QuestionDetector
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayEvent, ReplayHarness

# Not a module-level pytestmark: the detector tests below are pure and sync,
# and marking them asyncio would be a lie pytest warns about.
asyncio_test = pytest.mark.asyncio


@pytest.fixture
def harness(monkeypatch):
    h = ReplayHarness(monkeypatch=monkeypatch)
    yield h
    h.dispose()


# ------------------------------------------------------------ the decision
# Pure, no timing at all: given this exact wording, what finality?


@pytest.mark.parametrize(
    "text",
    [
        "What is dependency injection?",
        "How does Redis work?",
        "Tell me about your last project.",
        "What is the difference between a list and a tuple?",
        # Complete coding prompts: imperative, but nothing is missing.
        "Write a function that reverses a linked list.",
        "Write a function to reverse a linked list",
        "Find duplicate elements in an array.",
        "Find two numbers whose sum equals a target.",
        # A scenario opener that Whisper closed with "?" is closed. The
        # opener heuristic must not outrank explicit terminal punctuation.
        "If you had to choose between Redis and Memcached?",
        "How would you design a URL shortener?",
    ],
)
def test_a_complete_request_is_never_held(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"
    assert detection.finality is Finality.COMPLETE
    assert detection.hold_ms == 0


@pytest.mark.parametrize(
    "text",
    [
        # Bare trigger: the interviewer named the kind of request, not the request.
        "Can you explain",
        "Tell me about",
        "Could you describe",
        # Trailing into a clause that says nothing yet.
        "Can you explain how",
        "Can you explain what happens when",
        "How would you scale this if",
        "Tell me about a project you built and",
        # A premise with nothing asked for yet.
        "Given an array of integers",
    ],
)
def test_a_provably_unfinished_request_accumulates(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.finality is Finality.ACCUMULATING
    assert detection.hold_ms > 0


@pytest.mark.parametrize(
    "text",
    [
        "Given an array of integers, I want you to find two numbers",
        "Given an array of integers, find two numbers",
        "Find some edge cases",
    ],
)
def test_a_request_naming_a_quantity_but_no_constraint_is_potentially_complete(text):
    """Grammatically whole, but worded the way interviewers word a problem
    whose constraint is still coming. Held briefly -- not as long as a
    provable fragment, because this one may already be finished."""
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.finality is Finality.POTENTIALLY_COMPLETE
    assert 0 < detection.hold_ms


def test_the_ambiguous_hold_is_shorter_than_the_provable_one():
    """Evidence strength has to be reflected in the wait, or the tiers are
    decorative: a maybe-complete request must not wait longer than one that
    is definitely incomplete."""
    maybe = QuestionDetector().inspect("Given an array of integers, find two numbers")
    provable = QuestionDetector().inspect("Can you explain")
    assert maybe.hold_ms < provable.hold_ms


@pytest.mark.parametrize("followup", ["Why?", "How?", "What about scalability?", "Can you elaborate?"])
def test_a_followup_is_never_held(followup):
    """A follow-up leans on the previous turn; there is no continuation
    coming that would make it more answerable."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)

    detection = detector.inspect(followup, now=101.0)
    assert detection.accepted
    assert detection.finality is Finality.COMPLETE
    assert detection.hold_ms == 0


def test_a_merged_continuation_becomes_complete():
    """The point of holding: once the rest of the sentence lands, the
    combined text is answerable and must stop being held."""
    detector = QuestionDetector(coalesce_ms=1000)
    first = detector.inspect("Can you explain", now=100.0)
    assert first.finality is Finality.ACCUMULATING

    second = detector.inspect("how dependency injection works in FastAPI", now=100.4)
    assert second.finality is Finality.COMPLETE
    assert second.hold_ms == 0
    assert second.supersedes


# ------------------------------------------------------ the word-count floor
# `question_min_words` gates more than "is this too short to bother with": a
# fragment below the floor is routed to the follow-up bypass, and the
# follow-up path is excluded from merging. So the floor silently decides
# whether a short trailing continuation joins the question it continues.


def test_the_shipped_word_floor_allows_two_word_continuations():
    """Regression guard for config drift. This was found set to 3 in a live
    .env, which is why it is asserted rather than assumed."""
    assert settings.question_min_words <= 2, (
        "question_min_words > 2 strands two-word continuations like "
        '"in FastAPI?" on the follow-up path, where they cannot merge'
    )


def test_a_two_word_continuation_merges_at_the_shipped_floor():
    detector = QuestionDetector(min_words=2, coalesce_ms=1200)
    detector.inspect("Can you explain", now=100.0)
    detector.inspect("how dependency injection works", now=100.4)

    merged = detector.inspect("in FastAPI?", now=100.8)
    assert merged.supersedes
    assert merged.text == "Can you explain how dependency injection works in FastAPI?"


def test_raising_the_word_floor_to_three_strands_the_continuation():
    """The failure mode the guard above exists for, pinned so the cost of
    changing the floor is visible rather than surprising."""
    detector = QuestionDetector(min_words=3, coalesce_ms=1200)
    detector.inspect("Can you explain how dependency injection works", now=100.0)

    stranded = detector.inspect("in FastAPI?", now=100.4)
    assert not stranded.supersedes
    assert stranded.text == "in FastAPI?"
    assert "dependency injection" not in stranded.text


# ------------------------------------------------------- the speech clock
# Grouping must follow when the interviewer spoke, not when Whisper replied.


def test_speech_gap_groups_fragments_that_whisper_returned_far_apart():
    """Fragments spoken 400ms apart belong to one request. Whisper may have
    returned them seconds apart under backlog; the timestamps passed here
    are speech-clock, and that is what the window measures."""
    detector = QuestionDetector(coalesce_ms=1000)
    detector.inspect("Can you explain", now=100.0)
    merged = detector.inspect("how connection pooling works", now=100.4)

    assert merged.supersedes
    assert "Can you explain" in merged.text
    assert "connection pooling" in merged.text


def test_a_real_pause_separates_questions_whisper_returned_together():
    """The inverse, and the one that actually guards the property: these two
    finals are delivered back to back in wall-clock time (no sleep between
    them), but the interviewer paused 6s between them. Arrival-time grouping
    would merge them; speech-time grouping must not."""
    detector = QuestionDetector(coalesce_ms=1000, context_window_ms=4000)
    detector.inspect("What is a hash map?", now=100.0)
    second = detector.inspect("What is a bloom filter?", now=106.0)

    assert second.accepted
    assert not second.supersedes
    assert "hash map" not in second.text
    assert second.text == "What is a bloom filter?"


# ------------------------------------------------- provider call accounting
# The objective: one exact assembled question, one provider call.


@asyncio_test
async def test_nothing_reaches_the_provider_while_a_question_is_held(monkeypatch):
    """The guarantee the hold exists for, asserted without waiting for it:
    with the budget set effectively to forever, an unfinished fragment must
    be parked with no provider call made.

    The other tests in this section run with a 1ms budget so they finish, so
    this is the one that proves the hold is load-bearing rather than a
    formality."""
    h = ReplayHarness(monkeypatch=monkeypatch, stabilization_ms=60_000)
    try:
        await h.live.on_transcript(
            "Can you explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )

        assert h.live._pending_ask is not None, "fragment should be parked"
        assert h.llm.prompts == [], "no provider call may happen while held"
        assert h.live._current_turn_id is None, "no turn may be opened while held"
    finally:
        h.live._cancel_pending_ask()
        h.dispose()


@asyncio_test
async def test_a_question_split_across_three_fragments_costs_one_call(monkeypatch):
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="Can you explain"),
            ReplayEvent(at_ms=400, text="how dependency injection works"),
            ReplayEvent(at_ms=800, text="in FastAPI?"),
        ], settle_between=False)

        assert len(h.llm.prompts) == 1, h.llm.prompts
        # Exact interviewer wording, in order, reaches the model -- assembled,
        # not paraphrased or summarised.
        prompt = h.llm.prompts[0]
        for fragment in ("Can you explain", "how dependency injection works", "in FastAPI"):
            assert fragment in prompt
        assert prompt.index("Can you explain") < prompt.index("how dependency injection")
        assert prompt.index("how dependency injection") < prompt.index("in FastAPI")
        assert len(result.of(EventType.ANSWER_COMPLETED)) == 1
    finally:
        await h.close()


@asyncio_test
async def test_a_complete_question_costs_one_call_and_is_not_delayed(harness):
    result = await harness.play([ReplayEvent(at_ms=0, text="What is dependency injection?")])

    assert len(harness.llm.prompts) == 1
    assert harness.live._pending_ask is None, "a complete question must never be held"
    assert result.detected_questions() == ["What is dependency injection?"]


@asyncio_test
async def test_a_complete_coding_prompt_costs_one_call_and_is_not_delayed(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="Write a function that reverses a linked list."),
    ])

    assert len(harness.llm.prompts) == 1
    assert harness.live._pending_ask is None
    assert len(result.of(EventType.ANSWER_COMPLETED)) == 1


@asyncio_test
async def test_a_semantically_open_coding_prompt_absorbs_its_constraint(monkeypatch):
    """The case that used to cost two calls: a problem setup answered
    immediately, then re-asked when its constraint arrived."""
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        await h.play([
            ReplayEvent(at_ms=0, text="Given an array of integers, find two numbers"),
            ReplayEvent(at_ms=600, text="whose sum equals a target value."),
        ], settle_between=False)

        assert len(h.llm.prompts) == 1, h.llm.prompts
        prompt = h.llm.prompts[0]
        assert "find two numbers" in prompt
        assert "sum equals a target value" in prompt
    finally:
        await h.close()


@asyncio_test
async def test_a_correction_replaces_rather_than_duplicating(monkeypatch):
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="How would you scale this to a thousand QPS?"),
            ReplayEvent(at_ms=400, text="Actually, assume ten thousand QPS."),
        ], settle_between=False)

        # Exactly one answer survives, and it carries the corrected figure.
        assert len(result.of(EventType.ANSWER_COMPLETED)) == 1
        assert "ten thousand" in h.llm.prompts[-1]
        assert set(result.completed_turn_ids()).isdisjoint(result.cancelled_turn_ids())
    finally:
        await h.close()


@asyncio_test
async def test_two_unrelated_questions_are_two_turns(harness):
    """Separated on the speech clock by well over every window, so they must
    not be assembled into one request."""
    result = await harness.play([
        ReplayEvent(at_ms=0, text="What is a hash map?"),
        ReplayEvent(at_ms=20_000, text="What is a bloom filter?"),
    ])

    assert result.detected_questions() == [
        "What is a hash map?",
        "What is a bloom filter?",
    ]
    assert len(result.of(EventType.ANSWER_COMPLETED)) == 2
    assert len(harness.llm.prompts) == 2


@asyncio_test
async def test_a_followup_is_answered_without_waiting(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="What is a hash map?"),
        ReplayEvent(at_ms=5_000, text="Why?"),
    ])

    assert len(result.of(EventType.ANSWER_COMPLETED)) == 2
    assert harness.live._pending_ask is None
    assert result.detected_questions()[-1] == "Why?"
