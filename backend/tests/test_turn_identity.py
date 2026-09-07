"""Which utterances are the same turn, and which are a new one.

Two failures on opposite sides of one question.

**One turn read as two.** An identical final delivered twice -- an STT
redelivery, a double-clicked manual submit -- opened a second turn, spent a
second understanding call and a second answer call, and left the first answer
running. The candidate saw the same question and the same answer twice, and
the session paid twice for one question.

**Two utterances read as none.** A one-word opener ("Explain") is below the
word floor and leans on nothing, so it was rejected; the utterance that
finished it ("Kafka replication.") names no request of its own, so it was
rejected too. The question was lost outright -- the one outcome worse than
answering it late.

The dangerous fix for the first is deduplicating by question text, which
would silently drop a question an interviewer legitimately asks again. So
every "must still be answered" case below is as load-bearing as the "must be
dropped" ones, and they are in the same file on purpose.
"""

import pytest

from app.core.config import settings
from app.realtime.events import EventType
from app.realtime.question_detector import QuestionDetector
from app.sessions.schemas import TranscriptSource
from tests.replay_harness import ReplayHarness
from tests.test_understanding_session import CountingCompleter, reply

asyncio_test = pytest.mark.asyncio

QUESTION = "How would you design a rate limiter?"

#: Comfortably inside `question_redelivery_ms`, in speech-clock seconds.
COPY_GAP = 0.2
#: Comfortably outside it, and outside every merge window too.
LATER = 30.0


@pytest.fixture
def harness(monkeypatch):
    h = ReplayHarness(monkeypatch=monkeypatch)
    yield h
    h.dispose()


# ------------------------------------------------------------- must be dropped


def test_a_redelivered_final_is_dropped():
    detector = QuestionDetector()
    assert detector.inspect(QUESTION, now=100.0).accepted
    repeat = detector.inspect(QUESTION, now=100.0 + COPY_GAP)
    assert not repeat.accepted
    assert repeat.detail == "duplicate_redelivery"


def test_a_burst_of_copies_is_dropped_without_extending_its_own_window():
    """Measured from the one real question, not from the previous copy -- or a
    stream of redeliveries would hold the window open indefinitely and a
    genuine re-ask arriving during it would be swallowed."""
    detector = QuestionDetector()
    detector.inspect(QUESTION, now=100.0)
    for offset in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        assert not detector.inspect(QUESTION, now=100.0 + offset).accepted
    # 1.6s is past the window even though every copy in between was inside it.
    assert detector.inspect(QUESTION, now=101.6).accepted


def test_a_copy_is_recognised_through_punctuation_and_case_drift():
    """Two passes over the same audio can differ in exactly those two ways."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)
    drifted = detector.inspect("what is a hash map.", now=100.0 + COPY_GAP)
    assert not drifted.accepted
    assert drifted.detail == "duplicate_redelivery"


def test_a_manual_resubmit_is_dropped_too():
    """A double-clicked submit reaches the detector through a different
    source, and is the same duplicate -- so the guard does not depend on
    `buffer_context`, which every other session-scoped behaviour does."""
    detector = QuestionDetector()
    assert detector.inspect(QUESTION, now=100.0, buffer_context=False).accepted
    repeat = detector.inspect(QUESTION, now=100.0 + COPY_GAP, buffer_context=False)
    assert not repeat.accepted


# ---------------------------------------------------------- must be answered


def test_the_same_question_asked_again_later_is_answered_again():
    detector = QuestionDetector()
    assert detector.inspect(QUESTION, now=100.0).accepted
    assert detector.inspect(QUESTION, now=100.0 + LATER).accepted


def test_the_same_text_after_a_different_question_is_answered():
    detector = QuestionDetector()
    assert detector.inspect(QUESTION, now=100.0).accepted
    assert detector.inspect("What is a semaphore?", now=103.0).accepted
    again = detector.inspect(QUESTION, now=106.0)
    assert again.accepted, "the live turn is a different question now"


def test_the_same_text_after_a_topic_switch_is_answered():
    detector = QuestionDetector()
    assert detector.inspect("How would you optimize this query?", now=100.0).accepted
    assert detector.inspect("Now let's switch to Kafka.", now=110.0).accepted
    again = detector.inspect("How would you optimize this query?", now=120.0)
    assert again.accepted


def test_a_different_question_immediately_after_is_answered():
    """The window is about identity, not about recency -- anything else would
    be a debounce on the interviewer."""
    detector = QuestionDetector()
    assert detector.inspect(QUESTION, now=100.0).accepted
    assert detector.inspect("What is a semaphore?", now=100.0 + COPY_GAP).accepted


def test_an_out_of_order_final_is_not_read_as_a_copy():
    """A negative speech-clock gap must not pass the window test."""
    detector = QuestionDetector()
    detector.inspect(QUESTION, now=105.0)
    assert detector.inspect(QUESTION, now=100.0).accepted


# ------------------------------------------------------------------ end to end


@asyncio_test
async def test_a_redelivered_final_costs_one_answer_and_one_of_each_call(harness):
    completer = CountingCompleter(reply(), reply())
    harness.live._understander.__init__(completer)

    for now in (100.0, 100.0 + COPY_GAP):
        await harness.live.on_transcript(
            QUESTION, TranscriptSource.LOOPBACK, is_final=True, now=now
        )
        await harness.settle()

    result = harness.result
    assert len(result.of(EventType.QUESTION_DETECTED)) == 1
    assert len(result.completed_turn_ids()) == 1
    assert not result.cancelled_turn_ids()
    assert completer.calls == 1, "one understanding call for one question"
    assert harness.llm.started == 1, "one answer call for one question"


@asyncio_test
async def test_a_genuine_re_ask_costs_a_second_answer(harness):
    for now in (100.0, 100.0 + LATER):
        await harness.live.on_transcript(
            QUESTION, TranscriptSource.LOOPBACK, is_final=True, now=now
        )
        await harness.settle()

    assert len(harness.result.of(EventType.QUESTION_DETECTED)) == 2
    assert len(harness.result.completed_turn_ids()) == 2


# ------------------------------------- a request split across two utterances


@pytest.mark.parametrize(
    ("opener", "rest", "expected"),
    [
        ("Explain", "Kafka replication.", "Explain Kafka replication?"),
        ("Explain", "Kafka replication", "Explain Kafka replication?"),
        ("Optimize", "this query.", "Optimize this query?"),
        ("Describe", "the retry policy.", "Describe the retry policy?"),
    ],
)
@pytest.mark.parametrize("gap", [0.1, 0.5, 1.5, 4.0, 7.9])
def test_a_held_opener_is_joined_to_the_utterance_that_finishes_it(
    opener, rest, expected, gap
):
    """Every gap inside `question_context_window_ms`, since which side of a
    VAD boundary the object lands on is not something the interviewer
    controls. The joined text is their own words in their own order."""
    detector = QuestionDetector()
    assert not detector.inspect(opener, now=100.0).accepted
    finished = detector.inspect(rest, now=100.0 + gap)
    assert finished.accepted, f"lost after a {gap}s gap"
    assert finished.text == expected


def test_a_held_opener_expires():
    detector = QuestionDetector()
    detector.inspect("Explain", now=100.0)
    stale = detector.inspect(
        "Kafka replication.", now=100.0 + settings.question_context_window_ms / 1000 + 1
    )
    assert not stale.accepted, "an opener that old belongs to a different question"


def test_an_opener_alone_never_produces_a_question():
    for opener in ("Explain", "Optimize", "Show", "Describe"):
        detector = QuestionDetector()
        assert not detector.inspect(opener, now=100.0).accepted


def test_only_a_request_opener_is_held():
    """Not every rejected utterance becomes setup -- "Only legitimate task/
    opening fragments should be retained"."""
    for chatter in ("Okay.", "Right.", "Mm-hmm.", "Next question.", "Sure."):
        detector = QuestionDetector()
        assert not detector.inspect(chatter, now=100.0).accepted
        assert detector.inspect("Kafka replication.", now=100.5).accepted is False, (
            f"{chatter!r} must not manufacture a question"
        )


def test_a_held_opener_is_not_joined_onto_filler():
    detector = QuestionDetector()
    detector.inspect("Explain", now=100.0)
    assert not detector.inspect("Okay.", now=100.5).accepted
    assert not detector.inspect("That makes sense.", now=101.0).accepted


def test_a_held_opener_never_rewrites_a_question_that_stands_on_its_own():
    """The join is a fallback for an utterance that is otherwise
    undetectable. A real question keeps its own exact wording."""
    detector = QuestionDetector()
    detector.inspect("Explain", now=100.0)
    real = detector.inspect("What is Kafka?", now=100.5)
    assert real.accepted
    assert real.text == "What is Kafka?"


def test_a_held_opener_is_dropped_once_a_question_lands():
    """It must not reach across a completed turn and glue itself to something
    said afterwards.

    What happens to the later utterance instead is the pre-existing merge
    behaviour -- it lands inside the coalesce window and is read as a
    follow-on clause of the question that was just accepted. That is not what
    this test is about; the assertion is only that "Explain" is gone.
    """
    detector = QuestionDetector()
    detector.inspect("Explain", now=100.0)
    assert detector.inspect("What is Kafka?", now=100.5).accepted
    later = detector.inspect("Kafka replication.", now=101.0)
    assert "explain" not in later.text.lower()
    assert "explain" not in later.effective_text.lower()


def test_a_typed_question_neither_holds_nor_uses_an_opener():
    """`buffer_context=False` isolates the candidate's own input from every
    session-scoped mechanism, and this is one."""
    detector = QuestionDetector()
    detector.inspect("Explain", now=100.0, buffer_context=False)
    assert not detector.inspect(
        "Kafka replication.", now=100.5, buffer_context=False
    ).accepted


@asyncio_test
async def test_a_split_opener_reaches_exactly_one_answer(harness):
    completer = CountingCompleter(reply())
    harness.live._understander.__init__(completer)

    await harness.live.on_transcript(
        "Explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
    )
    await harness.settle()
    await harness.live.on_transcript(
        "Kafka replication.", TranscriptSource.LOOPBACK, is_final=True, now=100.5
    )
    await harness.settle()

    result = harness.result
    assert result.detected_questions() == ["Explain Kafka replication?"]
    assert len(result.completed_turn_ids()) == 1
    assert completer.calls == 1
    assert harness.llm.started == 1


@asyncio_test
async def test_an_opener_with_nothing_after_it_answers_nothing(harness):
    await harness.live.on_transcript(
        "Explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
    )
    await harness.settle()

    assert not harness.result.of(EventType.QUESTION_DETECTED)
    assert not harness.result.completed_turn_ids()
    assert harness.llm.started == 0
