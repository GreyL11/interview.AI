from app.core.config import settings
from app.memory.base import SessionMemory
from app.memory.session_memory import estimate_tokens
from app.storage.session_repository import SessionRepository


class SqliteSessionMemory(SessionMemory):
    """Conversation history backed by the turns table.

    Reads the most recent answered turns and walks backwards until the token
    budget is spent, so the verbatim window is bounded by size rather than by a
    fixed turn count — one rambling system-design answer costs the same budget
    as several short ones.
    """

    def __init__(self, sessions: SessionRepository, max_tokens: int | None = None) -> None:
        self._sessions = sessions
        self._max_tokens = max_tokens if max_tokens is not None else settings.memory_max_tokens

    def get_history(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []

        turns = self._sessions.get_answered_turns(session_id)
        window: list[str] = []
        spent = 0
        for turn in reversed(turns):
            answer = turn.answer.summary if turn.answer else ""
            pair = [f"Q: {turn.question}", f"A: {answer}"]
            cost = sum(estimate_tokens(p) for p in pair)
            if spent + cost > self._max_tokens and window:
                break
            window[:0] = pair
            spent += cost
        return window

    def verbatim_from_seq(self, session_id: str) -> int:
        """Lowest seq still held verbatim. Turns below this are the summarizer's
        responsibility."""
        turns = self._sessions.get_answered_turns(session_id)
        kept = len(self.get_history(session_id)) // 2
        if kept >= len(turns):
            return 0
        return turns[len(turns) - kept].seq if kept else (turns[-1].seq + 1 if turns else 0)

    def append_turn(self, session_id: str | None, question: str, answer_summary: str) -> None:
        # No-op: turns are persisted by the realtime pipeline via SessionRepository,
        # which records far more than a summary string. Kept to satisfy the
        # interface for callers that don't know which implementation they hold.
        return

    def summary(self, session_id: str | None) -> str:
        if not session_id:
            return ""
        stored = self._sessions.get_summary(session_id)
        return stored.summary if stored else ""

    def topics(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []
        stored = self._sessions.get_summary(session_id)
        return stored.topics if stored else []
