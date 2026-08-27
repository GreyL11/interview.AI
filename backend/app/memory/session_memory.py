from app.core.config import settings
from app.memory.base import SessionMemory


def estimate_tokens(text: str) -> int:
    # ponytail: 4 chars/token approximation, same as the chunker. Used only for
    # budgeting; being 20% off costs a little prompt room, never correctness.
    return max(1, len(text) // 4)


class InMemorySessionMemory(SessionMemory):
    """Process-local conversation history, bounded by a token budget.

    Kept as the default for the request/response `POST /question` path and for
    tests. Live sessions use SqliteSessionMemory so history survives a restart.
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        self._max_tokens = max_tokens if max_tokens is not None else settings.memory_max_tokens
        self._sessions: dict[str, list[str]] = {}
        self._summaries: dict[str, str] = {}
        self._topics: dict[str, list[str]] = {}

    def get_history(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []
        return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str | None, question: str, answer_summary: str) -> None:
        if not session_id:
            return
        history = self._sessions.setdefault(session_id, [])
        history.append(f"Q: {question}")
        history.append(f"A: {answer_summary}")
        self._trim(history)

    def _trim(self, history: list[str]) -> None:
        # Drop oldest Q/A pairs until the window fits. Pairs, not lines, so a
        # question never survives without its answer.
        while len(history) > 2 and sum(estimate_tokens(h) for h in history) > self._max_tokens:
            del history[:2]

    def summary(self, session_id: str | None) -> str:
        return self._summaries.get(session_id or "", "")

    def topics(self, session_id: str | None) -> list[str]:
        return list(self._topics.get(session_id or "", []))

    def set_summary(self, session_id: str, summary: str, topics: list[str]) -> None:
        self._summaries[session_id] = summary
        self._topics[session_id] = topics


#: Default instance used by the POST /question path.
session_memory = InMemorySessionMemory()
