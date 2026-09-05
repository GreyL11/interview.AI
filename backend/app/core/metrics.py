"""Structured latency metrics.

One line per pipeline stage, at INFO, in `metric <event> k=v k=v` form so the
whole speech-end -> first-token path can be reconstructed from a log file with
grep alone. Deliberately not a metrics framework: this is a desktop app with
one user, and a log line is the thing a support bundle already carries.
"""

import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

logger = get_logger("app.metrics")


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, str) and " " in value:
        return f'"{value}"'
    return str(value)


def log_metric(event: str, **fields: Any) -> None:
    """Emit one timing event. Fields that are None are omitted, so callers can
    pass optional identifiers unconditionally."""
    rendered = " ".join(
        f"{key}={_format(value)}" for key, value in fields.items() if value is not None
    )
    logger.info("metric %s %s", event, rendered)


def elapsed_ms(started: float, now: float) -> int:
    """Monotonic-clock delta in whole milliseconds."""
    return int((now - started) * 1000)


@dataclass
class LatencyTrace:
    """One interviewer utterance's timing, from speech-end through the first
    visible answer token, correlated by question_id in one summary line.

    Threaded explicitly through the pipeline (STT worker -> LiveSession)
    rather than looked up from a dict by ID: there is then never a table of
    in-flight traces to leak or expire if a stage never completes. All
    timestamps are `time.monotonic()`, matching the rest of the codebase, so
    a value recorded on the STT worker thread can still be safely subtracted
    from one recorded later on the event loop.
    """

    speech_end_at: float
    #: Which session/utterance produced this trace, so a log line is joinable
    #: back to the STT-side `stt_job_*`/`speech_*_detected` events (those carry
    #: `utterance_id` but not `question_id`, and vice versa on the answer side).
    session_id: str | None = None
    utterance_id: int | None = None
    stt_queue_wait_ms: int | None = None
    stt_inference_ms: int | None = None
    stt_final_at: float | None = None
    question_detected_at: float | None = None
    ask_started_at: float | None = None
    cancel_wait_ms: int | None = None
    #: The bounded classifier call. Timed here as well as in
    #: `llm_request_prepared` so one line carries the whole critical path.
    understanding_ms: int | None = None
    retrieval_ms: int | None = None
    prompt_build_ms: int | None = None
    llm_request_at: float | None = None
    #: The provider's first streamed chunk of any kind. In this app's configuration
    #: (plain JSON text streaming, no function calling) this is the same
    #: moment as the first *text* chunk -- see the note on `stream_answer`.
    llm_first_response_at: float | None = None
    #: First chunk containing text. Structurally this is *not* the same as
    #: the first useful answer token: the response is streamed JSON, so the
    #: first several chunks are usually the `{"summary": "` preamble before
    #: any of the actual summary text appears.
    llm_first_text_token_at: float | None = None
    #: Set once a trace line has been written, so a turn produces exactly one.
    #: A cancelled turn that had already streamed text has its story told by
    #: the first-token line; emitting a second one on the way out would double
    #: count it in any latency aggregate.
    reported: bool = False

    def _ms(self, start: float | None, end: float | None) -> int | None:
        if start is None or end is None:
            return None
        return elapsed_ms(start, end)

    def emit_terminal(self, question_id: int, outcome: str) -> None:
        """Write the trace for a turn that never reached a visible token.

        Without this, a question that timed out, failed or was superseded left
        no `question_latency_trace` line at all -- so the only latencies ever
        measured were the successful ones, which is exactly the population
        that hides a provider problem. `outcome` distinguishes them.
        """
        if self.reported:
            return
        self.reported = True
        log_metric(
            "question_latency_trace",
            question_id=question_id,
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            outcome=outcome,
            speech_end_to_stt_final_ms=self._ms(self.speech_end_at, self.stt_final_at),
            stt_queue_wait_ms=self.stt_queue_wait_ms,
            stt_inference_ms=self.stt_inference_ms,
            stt_final_to_question_detected_ms=self._ms(
                self.stt_final_at, self.question_detected_at
            ),
            question_detected_to_ask_ms=self._ms(
                self.question_detected_at, self.ask_started_at
            ),
            previous_answer_cancel_wait_ms=self.cancel_wait_ms,
            understanding_ms=self.understanding_ms,
            retrieval_ms=self.retrieval_ms,
            prompt_build_ms=self.prompt_build_ms,
            llm_task_to_request_ms=self._ms(self.ask_started_at, self.llm_request_at),
            # How long the provider was given before it went wrong. Null here
            # means it never produced a chunk at all.
            llm_request_to_first_response_ms=self._ms(
                self.llm_request_at, self.llm_first_response_at
            ),
            total_question_to_outcome_ms=self._ms(self.speech_end_at, time.monotonic()),
        )

    def emit_first_token(self, question_id: int) -> None:
        """Log the one consolidated trace line, at the moment the first
        *visible* answer token is forwarded onto the WebSocket -- the actual
        "time to first useful token" the user experiences, which can lag the
        raw provider first-token time by however long the JSON preamble takes
        to stream past (see `llm_first_text_token_at`)."""
        now = time.monotonic()
        self.reported = True
        log_metric(
            "question_latency_trace",
            question_id=question_id,
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            outcome="first_token",
            speech_end_to_stt_final_ms=self._ms(self.speech_end_at, self.stt_final_at),
            stt_queue_wait_ms=self.stt_queue_wait_ms,
            stt_inference_ms=self.stt_inference_ms,
            stt_final_to_question_detected_ms=self._ms(
                self.stt_final_at, self.question_detected_at
            ),
            question_detected_to_ask_ms=self._ms(self.question_detected_at, self.ask_started_at),
            previous_answer_cancel_wait_ms=self.cancel_wait_ms,
            understanding_ms=self.understanding_ms,
            retrieval_ms=self.retrieval_ms,
            prompt_build_ms=self.prompt_build_ms,
            llm_task_to_request_ms=self._ms(self.ask_started_at, self.llm_request_at),
            llm_request_to_first_response_ms=self._ms(
                self.llm_request_at, self.llm_first_response_at
            ),
            llm_request_to_first_text_token_ms=self._ms(
                self.llm_request_at, self.llm_first_text_token_at
            ),
            first_token_to_websocket_send_ms=self._ms(self.llm_first_text_token_at, now),
            total_question_to_first_visible_token_ms=self._ms(self.speech_end_at, now),
        )
