import pytest

from app.realtime.events import RejectionReason
from app.realtime.question_detector import QuestionDetector
from app.schemas.classification import Category


@pytest.mark.parametrize(
    "text,category",
    [
        ("How would you handle duplicate records in a data pipeline?", Category.SCENARIO),
        ("Write a function to reverse a linked list", Category.CODING),
        ("Tell me about a time you had a team conflict", Category.BEHAVIORAL),
        ("How would you design a URL shortener?", Category.SYSTEM_DESIGN),
    ],
)
def test_accepts_real_questions(text, category):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.classification.category == category


def test_accepts_imperative_coding_prompt():
    """Not phrased as a question, but it is the task being set."""
    detection = QuestionDetector().inspect("Write a function to reverse a linked list")
    assert detection.accepted
    assert detection.classification.requires_code


@pytest.mark.parametrize("filler", ["yeah", "mm-hmm", "right", "ok sure"])
def test_rejects_short_acknowledgements(filler):
    detection = QuestionDetector().inspect(filler)
    assert not detection.accepted
    assert detection.reason in (RejectionReason.TOO_SHORT, RejectionReason.NOT_A_QUESTION)


def test_rejects_statements():
    detection = QuestionDetector().inspect("thanks that makes a lot of sense to me")
    assert not detection.accepted
    assert detection.reason == RejectionReason.NOT_A_QUESTION


def test_coalesces_a_rapid_correction():
    detector = QuestionDetector(coalesce_ms=1000)
    first = detector.inspect("How would you scale this service?", now=100.0)
    assert first.accepted

    second = detector.inspect("Actually assume ten thousand QPS", now=100.4)
    assert second.accepted
    assert second.supersedes
    assert "scale this service" in second.text
    assert "ten thousand QPS" in second.text


def test_does_not_coalesce_after_the_window():
    detector = QuestionDetector(coalesce_ms=1000)
    detector.inspect("How would you scale this service?", now=100.0)
    later = detector.inspect("What is a database index anyway?", now=105.0)

    assert later.accepted
    assert not later.supersedes
    assert "scale this service" not in later.text


def test_reset_clears_coalescing_state():
    detector = QuestionDetector(coalesce_ms=1000)
    detector.inspect("How would you scale this service?", now=100.0)
    detector.reset()
    following = detector.inspect("What is a database index anyway?", now=100.2)
    assert not following.supersedes


def test_min_words_is_configurable():
    assert not QuestionDetector(min_words=10).inspect("What is a database index?").accepted


def test_low_confidence_is_rejected():
    detector = QuestionDetector(min_confidence=0.99)
    detection = detector.inspect("What is a database index?")
    assert not detection.accepted
    assert detection.reason == RejectionReason.LOW_CONFIDENCE
