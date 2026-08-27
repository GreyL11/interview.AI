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
