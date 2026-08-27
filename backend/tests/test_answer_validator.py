import pytest

from app.intelligence.answer_validator import AnswerValidationError, validate
from app.intelligence.classifier import classify
from app.schemas.answer import Answer


def test_empty_summary_rejected():
    with pytest.raises(AnswerValidationError):
        validate(Answer(summary="  "), classify("What is a database index?"))


def test_unsupported_personal_claim_warned():
    classification = classify("What is a database index?")
    answer = Answer(summary="I built a sharded index at my last job.")
    assert validate(answer, classification).warnings


def test_no_warning_when_personal_context_expected():
    classification = classify("Tell me about a time you had a team conflict")
    answer = Answer(summary="I led the mediation between two engineers.")
    assert validate(answer, classification).warnings == []
