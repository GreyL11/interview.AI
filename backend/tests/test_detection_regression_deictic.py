"""A referent-less fragment must not trigger an answer; a grounded one must."""
import pytest

from app.realtime.question_detector import QuestionDetector

BARE = ["What is this?", "What is this", "What about that?", "What is that?"]


@pytest.mark.parametrize("text", BARE)
def test_bare_deictic_without_referent_is_held_back(text):
    assert not QuestionDetector().inspect(text, now=100.0).accepted


@pytest.mark.parametrize("text", BARE)
def test_bare_deictic_after_recent_question_is_follow_up(text):
    d = QuestionDetector()
    assert d.inspect("How would you design a cache for a service?", now=99.0).accepted
    assert d.inspect(text, now=100.0).accepted


@pytest.mark.parametrize("text", [
    "What is Kafka?", "Explain this.", "What is this used for?",
    "Can you explain that", "What is a mutex",
])
def test_self_contained_questions_still_trigger(text):
    assert QuestionDetector().inspect(text, now=100.0).accepted


def test_split_deictic_fragment_still_merges_with_continuation():
    d = QuestionDetector()
    assert not d.inspect("What is this", now=100.0).accepted
    r = d.inspect("design pattern used for in Spring", now=100.6, accumulating=True)
    assert r.accepted and "design pattern" in r.effective_text


def test_split_explain_it_merges_with_continuation():
    d = QuestionDetector()
    d.inspect("Can you explain it", now=100.0)
    r = d.inspect("in terms of memory", now=100.6, accumulating=True)
    assert r.accepted and "explain it" in r.effective_text and "memory" in r.effective_text


def test_stage_percentiles():
    from app.core.metrics import _STAGE_SAMPLES, record_stage, stage_percentiles
    _STAGE_SAMPLES.clear()
    for v in range(1, 101):
        record_stage("x", v)
    p = stage_percentiles()["x"]
    assert 48 <= p["p50"] <= 53 and 93 <= p["p95"] <= 97
