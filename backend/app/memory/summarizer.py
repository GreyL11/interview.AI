import json

from app.core.logging import get_logger
from app.documents.schemas import utcnow
from app.realtime.question_understanding import StructuredCompleter
from app.sessions.schemas import SessionSummary
from app.storage.session_repository import SessionRepository

logger = get_logger(__name__)

_PROMPT = """Summarise this interview practice conversation for use as background context.

Rules:
- At most 120 words.
- Record what was discussed and any facts the candidate established about themselves.
- Do not invent details that are not present.
- Also list the technical topics covered.

Respond with ONLY this JSON:
{{"summary": "...", "topics": ["topic1", "topic2"]}}

Conversation:
{conversation}
"""

#: What the provider is constrained to produce. The prompt above says the same
#: thing in words, and still has to: the schema only reaches a model that
#: supports constrained decoding (see `app.llm.base.json_response_format`).
_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "topics"],
}

#: Generous on purpose. Summarising runs off the answer path, so it is never
#: the thing an interviewer waits on -- the only cost of a slow call is a
#: later summary.
_TIMEOUT_SECONDS = 20.0


class SessionSummarizer:
    """Compresses turns that have aged out of the verbatim window.

    Runs off the answer path: summarising is never allowed to add latency to a
    live question, and a failure here degrades context quality rather than
    breaking the session.
    """

    def __init__(self, sessions: SessionRepository, llm: StructuredCompleter) -> None:
        self._sessions = sessions
        self._llm = llm

    def needs_summary(self, session_id: str, verbatim_from_seq: int) -> bool:
        if verbatim_from_seq <= 0:
            return False
        existing = self._sessions.get_summary(session_id)
        covered = existing.covered_through_seq if existing else 0
        return verbatim_from_seq > covered

    async def summarize(self, session_id: str, through_seq: int) -> SessionSummary | None:
        turns = [t for t in self._sessions.get_answered_turns(session_id) if t.seq < through_seq]
        if not turns:
            return None

        existing = self._sessions.get_summary(session_id)
        lines: list[str] = []
        if existing and existing.summary:
            lines.append(f"Previous summary: {existing.summary}")
        for turn in turns:
            answer = turn.answer.summary if turn.answer else ""
            lines.append(f"Q: {turn.question}\nA: {answer}")

        try:
            payload = json.loads(
                await self._llm.complete_json(
                    _PROMPT.format(conversation="\n\n".join(lines)),
                    model="",
                    timeout_seconds=_TIMEOUT_SECONDS,
                    schema=_RESPONSE_SCHEMA,
                )
            )
        except Exception as exc:
            # Any provider failure, and any response a model on the
            # unconstrained path mangled. The verbatim window has already
            # dropped these turns, so losing the summary costs context, not
            # correctness. CancelledError is a BaseException and still
            # propagates.
            logger.warning("session_summary_failed session=%s error=%s", session_id, exc)
            return None

        if not isinstance(payload, dict):
            logger.warning("session_summary_not_an_object session=%s", session_id)
            return None

        summary = SessionSummary(
            session_id=session_id,
            summary=payload.get("summary", "")[:2000],
            topics=[str(t) for t in payload.get("topics", [])][:12],
            covered_through_seq=through_seq,
            updated_at=utcnow(),
        )
        self._sessions.upsert_summary(summary)
        logger.info(
            "session_summarized session=%s through_seq=%d topics=%d",
            session_id, through_seq, len(summary.topics),
        )
        return summary
