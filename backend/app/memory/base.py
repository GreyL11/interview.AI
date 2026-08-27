from abc import ABC, abstractmethod


class SessionMemory(ABC):
    """Conversation context for follow-up questions.

    Implementations must keep the returned context bounded — an interview
    session can run for an hour, and an unbounded history would grow the prompt
    until it is slow, expensive, and eventually rejected.
    """

    @abstractmethod
    def get_history(self, session_id: str | None) -> list[str]:
        """Recent turns, oldest first, already within the token budget."""
        ...

    @abstractmethod
    def append_turn(self, session_id: str | None, question: str, answer_summary: str) -> None:
        ...

    @abstractmethod
    def summary(self, session_id: str | None) -> str:
        """Compressed context for turns that fell out of the verbatim window."""
        ...

    @abstractmethod
    def topics(self, session_id: str | None) -> list[str]:
        ...

    def bounded_context(self, session_id: str | None) -> list[str]:
        """Summary (if any) followed by the verbatim window."""
        parts: list[str] = []
        summary = self.summary(session_id)
        if summary:
            parts.append(f"[Earlier in this session] {summary}")
        parts.extend(self.get_history(session_id))
        return parts
