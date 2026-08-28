"""QuestionDetector accuracy evaluation against a curated dataset.

This is a REGRESSION/EVALUATION tool, not a production-accuracy claim: the
dataset is a small, curated set of representative cases (see
detector_eval_dataset.json), not a statistically representative sample of
real interviews. Numbers below describe how the detector performs against
THIS dataset today, so a future change can be checked against it.

Each case replays its `utterances` in order through one fresh
QuestionDetector, using synthetic `now` timestamps (via `gap_ms`) so
window/stabilization logic is exercised deterministically -- no real
sleeping, no hidden state shared between cases.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_detector.py
    .venv/Scripts/python.exe scripts/evaluate_detector.py --strict
    .venv/Scripts/python.exe scripts/evaluate_detector.py --min-accuracy 0.9 --min-category-accuracy 0.8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.realtime.question_detector import Detection, QuestionDetector  # noqa: E402

DATASET_PATH = Path(__file__).parent / "detector_eval_dataset.json"

# Regression floor, not a real-world accuracy claim: these are the minimum
# scores this dataset must keep scoring for `evaluate_detector.py --strict`
# (or an explicit --min-* override) to pass. Set from the dataset's own
# measured performance, with a little headroom -- see the report this script
# prints for the actual current numbers.
DEFAULT_MIN_ACCURACY = 0.90
DEFAULT_MIN_PRECISION = 0.90
DEFAULT_MIN_RECALL = 0.90
DEFAULT_MIN_CATEGORY_ACCURACY = 0.75


def run_case(case: dict) -> tuple[Detection, list[Detection]]:
    """Replay one case's utterances through a fresh detector. Returns the
    final Detection (what expectations are checked against) and the full
    per-utterance history (for debugging)."""
    detector = QuestionDetector()
    now = 0.0
    history: list[Detection] = []
    for utt in case["utterances"]:
        source = utt.get("source", "LOOPBACK")
        now += utt.get("gap_ms", 300) / 1000
        detection = detector.inspect(utt["text"], now, buffer_context=source == "LOOPBACK")
        history.append(detection)
    return history[-1], history


def evaluate(cases: list[dict]) -> dict:
    tp = tn = fp = fn = 0
    category_checked = category_correct = 0
    failures: list[dict] = []

    for case in cases:
        final, _ = run_case(case)
        expected = case["expect_detected"]
        actual = final.accepted

        if expected and actual:
            tp += 1
        elif not expected and not actual:
            tn += 1
        elif not expected and actual:
            fp += 1
            failures.append({"id": case["id"], "kind": "false_positive",
                              "expected": expected, "actual": actual,
                              "transcript": case["utterances"][-1]["text"]})
        else:
            fn += 1
            failures.append({"id": case["id"], "kind": "false_negative",
                              "expected": expected, "actual": actual,
                              "transcript": case["utterances"][-1]["text"]})

        expect_category = case.get("expect_category")
        if expect_category and expected and actual:
            category_checked += 1
            actual_category = final.classification.category.value if final.classification else None
            if actual_category == expect_category:
                category_correct += 1
            else:
                failures.append({"id": case["id"], "kind": "wrong_category",
                                  "expected": expect_category, "actual": actual_category,
                                  "transcript": case["utterances"][-1]["text"]})

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    accuracy = (tp + tn) / total if total else 1.0
    category_accuracy = category_correct / category_checked if category_checked else None

    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "accuracy": accuracy,
        "category_checked": category_checked, "category_correct": category_correct,
        "category_accuracy": category_accuracy,
        "failures": failures,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_report(results: dict) -> None:
    print("Question Detection Evaluation")
    print("-" * 40)
    print(f"Cases:               {results['total']}")
    print(f"True Positives:      {results['tp']}")
    print(f"True Negatives:      {results['tn']}")
    print(f"False Positives:     {results['fp']}")
    print(f"False Negatives:     {results['fn']}")
    print()
    print(f"Precision:           {_pct(results['precision'])}")
    print(f"Recall:              {_pct(results['recall'])}")
    print(f"Accuracy:            {_pct(results['accuracy'])}")
    print()
    print("Category Classification Evaluation (only cases correctly detected AND with an expected category)")
    print("-" * 40)
    print(f"Cases checked:       {results['category_checked']}")
    print(f"Category Accuracy:   {_pct(results['category_accuracy'])}")

    if results["failures"]:
        print()
        print("Failures:")
        for f in results["failures"]:
            print(f"  [{f['kind']}] {f['id']}: expected={f['expected']!r} actual={f['actual']!r} "
                  f"text={f['transcript']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--strict", action="store_true",
                         help="Exit non-zero if any of the default minimum thresholds are violated.")
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-category-accuracy", type=float, default=None)
    args = parser.parse_args()

    cases = json.loads(args.dataset.read_text())["cases"]
    results = evaluate(cases)
    print_report(results)

    thresholds = {
        "accuracy": args.min_accuracy if args.min_accuracy is not None
        else (DEFAULT_MIN_ACCURACY if args.strict else None),
        "precision": args.min_precision if args.min_precision is not None
        else (DEFAULT_MIN_PRECISION if args.strict else None),
        "recall": args.min_recall if args.min_recall is not None
        else (DEFAULT_MIN_RECALL if args.strict else None),
        "category_accuracy": args.min_category_accuracy if args.min_category_accuracy is not None
        else (DEFAULT_MIN_CATEGORY_ACCURACY if args.strict else None),
    }
    violations = [
        f"{metric} {results[metric]:.3f} < {minimum:.3f}"
        for metric, minimum in thresholds.items()
        if minimum is not None and results[metric] is not None and results[metric] < minimum
    ]
    if violations:
        print()
        print("THRESHOLD VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
