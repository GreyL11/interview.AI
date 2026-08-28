"""Tests for the offline analysis tools in scripts/.

These are developer tools, not request-path code, but they are the thing
future decisions get made from -- a silently wrong percentile or a dataset
that stops loading would be worse than no tool at all.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
FIXTURE_LOG = Path(__file__).resolve().parent / "fixtures_latency_traces.log"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summarize_latency = _load("summarize_latency")
evaluate_detector = _load("evaluate_detector")


# ------------------------------------------------------- latency summarizer


def test_parses_traces_and_skips_incomplete_ones():
    traces = summarize_latency.parse_traces(FIXTURE_LOG.read_text().splitlines())

    # The fixture holds 6 trace lines; one lacks a first visible token.
    assert len(traces) == 5
    assert all(summarize_latency.COMPLETION_FIELD in t for t in traces)


def test_unrelated_metric_lines_are_ignored():
    lines = ["... metric question_stabilization_started session_id=abc delay_ms=400"]
    assert summarize_latency.parse_traces(lines) == []


def test_each_metric_reports_its_own_sample_count():
    """retrieval_ms is absent on non-RAG routes; it must be summarized over
    only the traces that actually have it, never averaged as if it were 0."""
    traces = summarize_latency.parse_traces(FIXTURE_LOG.read_text().splitlines())
    rows = {r["metric"]: r for r in summarize_latency.summarize(traces)}

    assert rows["retrieval_ms"]["n"] == 1
    assert rows["total_question_to_first_visible_token_ms"]["n"] == 5


def test_percentiles_are_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert summarize_latency.percentile(values, 0.5) == 30.0
    assert summarize_latency.percentile(values, 0.95) == 100.0


def test_missing_file_exits_non_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["summarize_latency.py", "does_not_exist.log"])
    assert summarize_latency.main() == 2


def test_a_log_with_no_traces_exits_non_zero(monkeypatch, tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("nothing interesting here\n")
    monkeypatch.setattr(sys, "argv", ["summarize_latency.py", str(empty)])
    assert summarize_latency.main() == 1


# ------------------------------------------------------- detector evaluation


def test_dataset_is_valid_and_every_case_is_well_formed():
    cases = json.loads((SCRIPTS / "detector_eval_dataset.json").read_text())["cases"]

    assert len(cases) >= 30
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in cases:
        assert case["utterances"], case["id"]
        assert isinstance(case["expect_detected"], bool), case["id"]
        assert case.get("group"), case["id"]


def test_every_dataset_group_is_represented():
    cases = json.loads((SCRIPTS / "detector_eval_dataset.json").read_text())["cases"]
    groups = {c["group"] for c in cases}

    assert groups >= {
        "real_question", "statement", "acknowledgement", "partial",
        "followup", "coding", "correction", "noisy_stt",
    }


def test_evaluation_runs_and_meets_the_committed_thresholds():
    """Guards the dataset the same way --strict does on the command line, so a
    detector regression fails the normal test run too, not only a manual step."""
    cases = json.loads((SCRIPTS / "detector_eval_dataset.json").read_text())["cases"]
    results = evaluate_detector.evaluate(cases)

    assert results["total"] == len(cases)
    assert results["accuracy"] >= evaluate_detector.DEFAULT_MIN_ACCURACY
    assert results["precision"] >= evaluate_detector.DEFAULT_MIN_PRECISION
    assert results["recall"] >= evaluate_detector.DEFAULT_MIN_RECALL
    assert results["category_accuracy"] >= evaluate_detector.DEFAULT_MIN_CATEGORY_ACCURACY


def test_multi_turn_cases_share_one_detector_instance():
    """A follow-up case only behaves correctly if earlier utterances actually
    ran through the same detector -- otherwise 'Why?' has no recent question
    and the case would silently pass for the wrong reason."""
    with_context = {
        "id": "t", "group": "followup",
        "utterances": [{"text": "What is a hash map?"}, {"text": "Why?", "gap_ms": 2000}],
        "expect_detected": True,
    }
    without_context = {
        "id": "t2", "group": "followup",
        "utterances": [{"text": "Why?"}],
        "expect_detected": False,
    }

    assert evaluate_detector.run_case(with_context)[0].accepted is True
    assert evaluate_detector.run_case(without_context)[0].accepted is False


@pytest.mark.parametrize("bad_threshold_arg", ["--min-accuracy", "--min-category-accuracy"])
def test_impossible_threshold_makes_the_runner_fail(monkeypatch, bad_threshold_arg):
    monkeypatch.setattr(sys, "argv", ["evaluate_detector.py", bad_threshold_arg, "1.5"])
    assert evaluate_detector.main() == 1
