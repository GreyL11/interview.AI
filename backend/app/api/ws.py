from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.deps import (
    get_llm_client,
    get_retriever,
    get_session_memory,
    get_session_repository,
    get_summarizer,
)
from app.core.logging import get_logger
from app.documents.schemas import utcnow
from app.realtime.events import CancelReason, ClientMessage, Event, EventType, event
from app.realtime.manager import session_manager
from app.realtime.session import LiveSession
from app.sessions.schemas import Session, TranscriptSource
from app.storage.session_repository import SessionRepository

logger = get_logger(__name__)

router = APIRouter()

WS_UNAUTHORIZED = 4001
WS_UNKNOWN_SESSION = 4004


def _authorized(token: str | None) -> bool:
    # The backend binds to localhost, but any local process could still reach it.
    # A token issued by the desktop shell at spawn time keeps other processes out.
    expected = settings.api_token
    return not expected or token == expected


async def _build_session(session_id: str, sessions: SessionRepository) -> LiveSession:
    return await session_manager.register(
        LiveSession(
            session_id=session_id,
            sessions=sessions,
            retriever=get_retriever(),
            llm=get_llm_client(),
            memory=get_session_memory(),
            summarizer=get_summarizer(),
        )
    )


@router.websocket("/ws/session/{session_id}")
async def session_socket(
    websocket: WebSocket,
    session_id: str,
    token: str | None = None,
    since_seq: int = 0,
) -> None:
    if not _authorized(token):
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    sessions = get_session_repository()
    if sessions.get(session_id) is None:
        sessions.create(Session(session_id=session_id, started_at=utcnow()))

    live = session_manager.get(session_id) or await _build_session(session_id, sessions)

    await websocket.accept()

    async def emit(ev: Event) -> None:
        await websocket.send_text(ev.model_dump_json())

    live.subscribe(emit)
    try:
        # Reconnect: replay what the client missed rather than restarting it.
        missed = live.replay_since(since_seq)
        for ev in missed:
            await emit(ev)
        if not missed:
            await live.emit(event(EventType.SESSION_STARTED, session_id=session_id))

        await _pump(websocket, live)

    except WebSocketDisconnect:
        logger.info("ws_disconnected session=%s", session_id)
    finally:
        live.unsubscribe(emit)


async def _pump(websocket: WebSocket, live: LiveSession) -> None:
    while True:
        raw = await websocket.receive_text()
        try:
            message = ClientMessage.model_validate_json(raw)
        except Exception as exc:
            await websocket.send_text(
                event(EventType.ERROR, code="BadMessage", message=str(exc)).model_dump_json()
            )
            continue

        if message.type == EventType.PING:
            await websocket.send_text(event(EventType.PONG).model_dump_json())

        elif message.type == EventType.QUESTION_MANUAL:
            text = str(message.data.get("text", "")).strip()
            if text:
                await live.on_transcript(text, TranscriptSource.MANUAL, is_final=True)

        elif message.type == EventType.ANSWER_CANCEL:
            await live.cancel(CancelReason.USER_STOP)

        elif message.type == EventType.SESSION_STOP:
            await session_manager.close(live.session_id)
            return

        else:
            await websocket.send_text(
                event(
                    EventType.ERROR,
                    code="UnsupportedMessage",
                    message=f"Unsupported client message: {message.type}",
                ).model_dump_json()
            )
