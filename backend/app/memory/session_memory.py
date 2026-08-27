MAX_TURNS_PER_SESSION = 10


class SessionMemory:
    """In-process conversation history keyed by session_id.

    ponytail: single global dict, not persisted or thread-safe beyond the
    GIL. Fine for one-process Phase 1; move to SQLite when sessions need to
    survive a restart or be shared across workers.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[str]] = {}

    def get_history(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []
        return self._sessions.get(session_id, [])

    def append_turn(self, session_id: str | None, question: str, answer_summary: str) -> None:
        if not session_id:
            return
        history = self._sessions.setdefault(session_id, [])
        history.append(f"Q: {question}")
        history.append(f"A: {answer_summary}")
        del history[: max(0, len(history) - MAX_TURNS_PER_SESSION * 2)]


session_memory = SessionMemory()
