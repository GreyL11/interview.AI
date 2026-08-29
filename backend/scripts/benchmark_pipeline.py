"""Per-stage latency benchmark for representative interview scenarios.

Exercises the real question-detection, retrieval-routing and
prompt-construction code with a fake, near-zero-delay LLM, so what it measures
is application-controlled overhead ONLY. It does not and cannot represent real
provider network or model latency -- do not read the "first_token_ms" column
here as a production estimate. For that, run the app and read the
`question_latency_trace` metric line from the log.

Usage:
    venv/Scripts/python.exe scripts/benchmark_pipeline.py
"""

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.metrics import LatencyTrace  # noqa: E402
from app.documents.schemas import utcnow  # noqa: E402
from app.llm.base import LLMClient  # noqa: E402
from app.memory.session_memory import InMemorySessionMemory  # noqa: E402
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
    Scenario(
        "Follow-up",
        ["Explain caching?", "Why?"],
        gaps=[2.0],
    ),
    Scenario(
        "Correction",
        ["Explain Docker...", "No, explain Kubernetes."],
        gaps=[0.05],
    ),
    Scenario(
        "Setup + question",
        [
            "Using this string, write a character count program.",
            "How many times is each character repeated?",
        ],
        gaps=[0.3],
    ),
]


async def run_scenario(scenario: Scenario) -> dict[str, float | None]:
    db_path = Path(f"_bench_{uuid.uuid4().hex}.db")
    db = Database(db_path)
    try:
        sessions = SessionRepository(db)
        session_id = str(uuid.uuid4())
        sessions.create(Session(session_id=session_id, started_at=utcnow()))
        live = LiveSession(
            session_id=session_id, sessions=sessions, retriever=MockRetriever(),
            llm=InstantLLM(), memory=InMemorySessionMemory(),
        )

        traces: list[dict] = []
        import app.core.metrics as metrics_module
        original = metrics_module.log_metric

        def capture(event, **fields):
            if event == "question_latency_trace":
                traces.append(fields)
            return original(event, **fields)

        metrics_module.log_metric = capture
        try:
            for i, text in enumerate(scenario.utterances):
                now = time.monotonic()
                # No real STT in this benchmark -- treat "final text is ready"
                # as the STT-final moment, so question-detection time is still
                # measured (the actual STT stage is not; see PERFORMANCE.md).
                trace = LatencyTrace(speech_end_at=now, stt_final_at=now)
                await live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, trace=trace)
                if live._pending_ask is not None:
                    await live._pending_ask
                if live._task is not None:
                    await asyncio.gather(live._task, return_exceptions=True)
                if i < len(scenario.gaps):
                    await asyncio.sleep(scenario.gaps[i])
        finally:
            metrics_module.log_metric = original

        if not traces:
            return {"question_detection_ms": None, "retrieval_ms": None,
                    "prompt_build_ms": None, "first_token_ms": None, "total_ms": None}

        last = traces[-1]
        return {
            "question_detection_ms": last.get("stt_final_to_question_detected_ms"),
            "retrieval_ms": last.get("retrieval_ms"),
            "prompt_build_ms": last.get("prompt_build_ms"),
            "first_token_ms": last.get("llm_request_to_first_text_token_ms"),
            "total_ms": last.get("total_question_to_first_visible_token_ms"),
        }
    finally:
        db.close()
        db_path.unlink(missing_ok=True)


def _fmt(value) -> str:
    return "-" if value is None else f"{value}ms"


async def run_mocked_benchmark() -> None:
    print(
        "\nMOCKED benchmark -- application overhead only "
        "(near-zero-delay fake LLM, no network). Not a provider latency estimate.\n"
    )
    header = f"{'Scenario':<24} {'Detection':>10} {'Retrieval':>10} {'Prompt':>8} {'1st Token':>10} {'Total':>8}"
    print(header)
    print("-" * len(header))
    for scenario in SCENARIOS:
        result = await run_scenario(scenario)
        print(
            f"{scenario.name:<24} {_fmt(result['question_detection_ms']):>10} "
            f"{_fmt(result['retrieval_ms']):>10} {_fmt(result['prompt_build_ms']):>8} "
            f"{_fmt(result['first_token_ms']):>10} {_fmt(result['total_ms']):>8}"
        )


async def main() -> None:
    await run_mocked_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
