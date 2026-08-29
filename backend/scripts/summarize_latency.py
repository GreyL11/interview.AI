"""Summarize real-machine latency from collected `question_latency_trace` logs.

Consumes the structured trace lines the running app already emits -- no new
instrumentation, no database, no external observability stack. Each trace is
one accepted question's full path from speech-end to first visible token.

    metric question_latency_trace question_id=7 speech_end_to_stt_final_ms=812 ...

Only lines with total_question_to_first_visible_token_ms present are counted:
a trace missing it never reached a visible answer (cancelled, superseded, or
failed) and including it would understate real latency. Each metric is
summarized over the traces that actually carry it, and the per-metric sample
count is reported so a metric present in only some traces (e.g. retrieval,
which is skipped for non-RAG routes) is never silently averaged against ones
where it did not apply.

Usage:
    python -m app --port 8000 > run.log 2>&1        # collect
    .venv/Scripts/python.exe scripts/summarize_latency.py run.log
    ... | .venv/Scripts/python.exe scripts/summarize_latency.py -
"""

import argparse
import re
import sys
from pathlib import Path

TRACE_MARKER = "metric question_latency_trace"
_FIELD = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)")

#: Reported in pipeline order, so the table reads as the actual journey.
METRICS = [
    "speech_end_to_stt_final_ms",
    "stt_queue_wait_ms",
    "stt_inference_ms",
    "stt_final_to_question_detected_ms",
    "question_detected_to_ask_ms",
    "previous_answer_cancel_wait_ms",
    "retrieval_ms",
    "prompt_build_ms",
    "llm_task_to_request_ms",
    "llm_request_to_first_response_ms",
    "llm_request_to_first_text_token_ms",
    "first_token_to_websocket_send_ms",
    "total_question_to_first_visible_token_ms",
]

COMPLETION_FIELD = "total_question_to_first_visible_token_ms"


def parse_traces(lines) -> list[dict[str, float]]:
    traces = []
    for line in lines:
        if TRACE_MARKER not in line:
            continue
        payload = line.split(TRACE_MARKER, 1)[1]
        fields = {k: float(v) for k, v in _FIELD.findall(payload)}
        if COMPLETION_FIELD in fields:
            traces.append(fields)
    return traces


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Exact and obvious for the small sample sizes
    a single-user desktop session produces -- interpolation would imply more
    precision than a few dozen samples support."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def summarize(traces: list[dict[str, float]]) -> list[dict]:
    rows = []
    for metric in METRICS:
        values = [t[metric] for t in traces if metric in t]
        if not values:
            continue
        rows.append({
            "metric": metric,
            "n": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": max(values),
        })
    return rows


def print_report(traces: list[dict[str, float]], rows: list[dict]) -> None:
    print(f"Latency summary over {len(traces)} completed question trace(s)")
    print("(traces without a first visible token -- cancelled/superseded/failed -- are excluded)")
    print()
    header = f"{'Metric':<42} {'n':>4} {'p50':>8} {'p95':>8} {'max':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['metric']:<42} {row['n']:>4} "
              f"{row['p50']:>8.0f} {row['p95']:>8.0f} {row['max']:>8.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", type=str,
                        help="Path to a log file, or '-' to read stdin.")
    args = parser.parse_args()

    if args.logfile == "-":
        lines = sys.stdin.readlines()
    else:
        path = Path(args.logfile)
        if not path.exists():
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        lines = path.read_text(errors="replace").splitlines()

    traces = parse_traces(lines)
    if not traces:
        print("No completed question_latency_trace lines found.", file=sys.stderr)
        print("Run a session first, and make sure LOG_LEVEL=INFO.", file=sys.stderr)
        return 1

    print_report(traces, summarize(traces))
    return 0


if __name__ == "__main__":
    sys.exit(main())
