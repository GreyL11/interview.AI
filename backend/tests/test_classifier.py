import pytest

from app.intelligence.classifier import classify
from app.schemas.classification import Category, Domain


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How would you handle duplicate records in a data pipeline?", Category.SCENARIO),
        ("Write a function to reverse a linked list", Category.CODING),
        ("Write a query to find the top 5 customers by revenue", Category.SQL),
        ("Tell me about a time you had a team conflict", Category.BEHAVIORAL),
        ("How would you design a URL shortener?", Category.SYSTEM_DESIGN),
        ("Why is this failing with a stack trace on startup?", Category.DEBUGGING),
        ("And what about at higher scale?", Category.FOLLOW_UP),
        ("What is a database index?", Category.TECHNICAL_KNOWLEDGE),
        ("thanks that makes sense", Category.UNKNOWN),
    ],
)
def test_category_detection(question, expected):
    assert classify(question).category == expected


def test_non_question_flagged():
    result = classify("thanks that makes sense")
    assert result.is_question is False
    assert result.category == Category.UNKNOWN


def test_domain_detection():
    assert classify("How do you dedupe rows in an ETL pipeline?").domain == Domain.DATA_ENGINEERING


def test_flags_for_personal_category():
    result = classify("Tell me about a time you had a team conflict")
    assert result.requires_personal_context is True
    assert result.requires_rag is True
    assert result.requires_code is False


def test_flags_for_coding_category():
    result = classify("Write a function to reverse a linked list")
    assert result.requires_code is True
    assert result.requires_reasoning is True
    assert result.requires_rag is False


@pytest.mark.parametrize(
    "question",
    [
        "What's wrong with this code?",
        "Why does this implementation fail?",
        "How would you fix the race condition?",
        "This isn't working, why not?",
    ],
)
def test_verbal_debugging_phrasings_are_classified_as_debugging(question):
    """These are natural spoken phrasings for a debugging follow-up on code
    the interviewer just described -- none of them contain the literal
    "bug"/"crash"/"stack trace" keywords the original regex required."""
    assert classify(question).category == Category.DEBUGGING


def test_a_positive_why_does_this_work_question_is_not_misclassified_as_debugging():
    """The debugging regex targets fail/broken specifically so a positive
    "why does X work" mechanism question -- common and unrelated to
    debugging -- is not swept in by a generic "why does...work" match."""
    result = classify("Why does event-driven architecture work well for this?")
    assert result.category != Category.DEBUGGING
