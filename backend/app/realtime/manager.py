import asyncio

from app.core.logging import get_logger
from app.realtime.session import LiveSession

logger = get_logger(__name__)


class SessionManager:
    """Registry of live sessions.

    Sessions outlive their WebSocket: a dropped connection pauses the view, not
    the session, so a reconnecting client can catch up rather than start over.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._lock = asyncio.Lock()

    async def register(self, session: LiveSession) -> LiveSession:
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> LiveSession | None:
        return self._sessions.get(session_id)

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            try:
                await self.close(session_id)
            except Exception:
                logger.exception("session_close_failed id=%s", session_id)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


session_manager = SessionManager()
