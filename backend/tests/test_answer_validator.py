import pytest

from app.intelligence.answer_validator import AnswerValidationError, validate
from app.intelligence.classifier import classify
from app.schemas.answer import Answer


def test_empty_summary_rejected():
    with pytest.raises(AnswerValidationError):
        validate(Answer(summary="  "), classify("What is a database index?"))


def test_unsupported_personal_claim_warned_on_technical_question():
    answer = Answer(summary="I built a sharded index at my last job.")
    result = validate(answer, classify("What is a database index?"), context_found=False)
    assert result.warnings


def test_personal_claim_warned_when_no_context_was_found():
    """The gap context_found closes: a behavioural question expects personal
    context, so the old flag-based check stayed silent. With an empty knowledge
    base there is nothing backing the claim, and it must be flagged."""
    answer = Answer(summary="I led the mediation between two engineers.")
    result = validate(
        answer, classify("Tell me about a time you had a team conflict"), context_found=False
    )
    assert result.warnings


def test_no_warning_when_context_supports_the_claim():
    answer = Answer(summary="I led the mediation between two engineers.")
    result = validate(
        answer, classify("Tell me about a time you had a team conflict"), context_found=True
    )
    assert result.warnings == []


def test_conditional_voice_never_warns():
    answer = Answer(summary="I would make the pipeline idempotent.")
    result = validate(answer, classify("How would you dedupe records?"), context_found=False)
    assert result.warnings == []


def test_claims_in_key_points_and_detail_are_checked():
    answer = Answer(
        summary="Here is the approach.",
        key_points=["I shipped this at Acme"],
        detailed_answer="Background.",
    )
    assert validate(answer, classify("What is caching?"), context_found=False).warnings


def test_existing_warnings_are_preserved():
    answer = Answer(summary="Fine.", warnings=["pre-existing"])
    result = validate(answer, classify("What is caching?"), context_found=True)
    assert "pre-existing" in result.warnings
