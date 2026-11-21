"""Per-stage latency benchmark for representative interview scenarios.

Exercises the real question-detection, understanding, retrieval-routing,
prompt-construction and event-emission code with a fake, zero-delay LLM, so
what it measures is application-controlled overhead ONLY (including the
detector's real stabilization holds). It does not and cannot represent STT,
VAD, network or model latency. For those, run the app and read the
`question_latency_trace` / `latency_percentiles` lines from the log.

Each scenario runs --runs times on a fresh session; stage timings are reported
as p50/p95/p99/avg/max with perf_counter resolution. Event-loop lag is the
worst overshoot of a 1 ms heartbeat while the scenario runs.

Usage:
    venv/Scripts/python.exe scripts/benchmark_pipeline.py [--runs 30]
"""

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.metrics import LatencyTrace  # noqa: E402
from app.documents.schemas import utcnow  # noqa: E402
from app.llm.base import LLMClient  # noqa: E402
from app.memory.session_memory import InMemorySessionMemory  # noqa: E402
from app.realtime.events import EventType  # noqa: E402
from app.realtime.session import LiveSession  # noqa: E402
from app.retrieval.mock_retriever import MockRetriever  # noqa: E402
from app.schemas.answer import Answer  # noqa: E402
from app.sessions.schemas import Session, TranscriptSource  # noqa: E402
from app.storage.database import Database  # noqa: E402
from app.storage.session_repository import SessionRepository  # noqa: E402


class InstantLLM(LLMClient):
    """Zero-delay fake: isolates application overhead from model latency."""

    async def generate_answer(self, prompt: str) -> Answer:
        raise NotImplementedError

    async def stream_answer(self, prompt: str):
        answer = Answer(summary="Answer.", key_points=["a", "b"], detailed_answer="detail")
        yield answer.model_dump_json()


@dataclass
class Scenario:
    name: str
    utterances: list[str]
    #: Wall-clock gap between utterances, seconds. 0 for "spoken back to back".
    gaps: list[float] = field(default_factory=list)


SCENARIOS = [
    Scenario("Short question", ["Explain caching?"]),
    Scenario(
        "Multi-sentence question",
        ["Can you explain how database indexing works and when we should use "
         "composite indexes?"],
    ),
    Scenario(
        "Coding question",
        ["Write a program to count how many times each character appears in a string."],
    ),
    Scenario("Follow-up", ["Explain caching?", "Why?"], gaps=[2.0]),
    Scenario("Correction", ["Explain Docker...", "No, explain Kubernetes."], gaps=[0.05]),
    Scenario(
        "Setup + question",
        [
            "Using this string, write a character count program.",
            "How many times is each character repeated?",
        ],
        gaps=[0.3],
    ),
]

#: column -> (start mark, end mark) over perf_counter timestamps of the LAST utterance.
STAGES = {
    "final->detected": ("final", "detected"),
    "detected->1st delta": ("detected", "delta"),
    "final->1st delta": ("final", "delta"),
}


async def _loop_lag(stop: asyncio.Event, out: list[float]) -> None:
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.001)
        out.append((time.perf_counter() - t) * 1000 - 1.0)


async def run_once(scenario: Scenario) -> dict[str, float]:
    db_path = Path(tempfile.gettempdir()) / f"_bench_{uuid.uuid4().hex}.db"
    db = Database(db_path)
    marks: dict[str, float] = {}
    try:
        sessions = SessionRepository(db)
        session_id = str(uuid.uuid4())
        sessions.create(Session(session_id=session_id, started_at=utcnow()))
        live = LiveSession(
            session_id=session_id, sessions=sessions, retriever=MockRetriever(),
            llm=InstantLLM(), memory=InMemorySessionMemory(),
        )

        async def on_event(ev) -> None:
            now = time.perf_counter()
            if ev.type == EventType.QUESTION_DETECTED:
                marks["detected"] = now
            elif ev.type == EventType.ANSWER_DELTA and "delta" not in marks:
                marks["delta"] = now

        live.subscribe(on_event)
        lag: list[float] = []
        stop = asyncio.Event()
        lag_task = asyncio.create_task(_loop_lag(stop, lag))
        cpu0 = time.process_time()
        for i, text in enumerate(scenario.utterances):
            marks.clear()
            now = time.monotonic()
            trace = LatencyTrace(speech_end_at=now, stt_final_at=now)
            marks["final"] = time.perf_counter()
            await live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, trace=trace)
            if live._pending_ask is not None:
                await asyncio.gather(live._pending_ask, return_exceptions=True)
            if live._task is not None:
                await asyncio.gather(live._task, return_exceptions=True)
            if i < len(scenario.gaps):
                await asyncio.sleep(scenario.gaps[i])
        cpu = (time.process_time() - cpu0) * 1000
        stop.set()
        await lag_task
        await live.close()
    finally:
        db.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(f"{db_path}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass  # ponytail: Windows may still hold a worker-thread handle; temp dir.
    result = {
        col: (marks[b] - marks[a]) * 1000
        for col, (a, b) in STAGES.items() if a in marks and b in marks
    }
    result["cpu ms"] = cpu
    result["loop lag max"] = max(lag, default=0.0)
    return result


def _summary(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) < 2:
        return f"{values[0]:.2f}"
    q = statistics.quantiles(values, n=100, method="inclusive")
    return (f"p50 {q[49]:7.2f}  p95 {q[94]:7.2f}  p99 {q[98]:7.2f}  "
            f"avg {statistics.fmean(values):7.2f}  max {max(values):7.2f}")


async def main(runs: int) -> None:
    print(f"\nMOCKED benchmark, {runs} runs/scenario -- application overhead only "
          "(zero-delay fake LLM, no STT/network). Times in ms.\n")
    for scenario in SCENARIOS:
        await run_once(scenario)  # warm-up: imports, regex compiles, sqlite
        samples = [await run_once(scenario) for _ in range(runs)]
        print(scenario.name)
        for col in [*STAGES, "cpu ms", "loop lag max"]:
            print(f"  {col:<20} {_summary([s[col] for s in samples if col in s])}")

    tracemalloc.start()
    for scenario in SCENARIOS:
        await run_once(scenario)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nPeak Python heap across one pass of all scenarios: {peak / 1024:.0f} KiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    asyncio.run(main(parser.parse_args().runs))
