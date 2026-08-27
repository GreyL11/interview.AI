import json

from app.core.config import settings
from app.core.logging import get_logger
from app.documents.schemas import utcnow
from app.llm.base import LLMClient, LLMError
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


class SessionSummarizer:
    """Compresses turns that have aged out of the verbatim window.

    Runs off the answer path: summarising is never allowed to add latency to a
    live question, and a failure here degrades context quality rather than
    breaking the session.
    """

    def __init__(self, sessions: SessionRepository, llm: LLMClient) -> None:
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
            answer = await self._llm.generate_answer(
                _PROMPT.format(conversation="\n\n".join(lines))
            )
            payload = _extract(answer.summary, answer.detailed_answer)
        except LLMError as exc:
            # Fallback: the verbatim window has already dropped these turns, so
            # losing the summary costs context, not correctness.
            logger.warning("session_summary_failed session=%s error=%s", session_id, exc)
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


def _extract(*candidates: str) -> dict:
    """The summariser reuses the standard Answer-shaped LLM call, so the JSON we
    want may land in either field. Try both before giving up."""
    for text in candidates:
        if not text:
            continue
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"summary": candidates[0] if candidates else "", "topics": []}
