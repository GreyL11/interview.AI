import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_session_repository
from app.documents.schemas import utcnow
from app.sessions.schemas import Session, SessionDetail, SessionListItem
from app.storage.session_repository import SessionRepository

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=Session, status_code=201)
async def create_session(
    title: str = "",
    sessions: SessionRepository = Depends(get_session_repository),
) -> Session:
    return sessions.create(
        Session(session_id=str(uuid.uuid4()), started_at=utcnow(), title=title)
    )


@router.get("", response_model=list[SessionListItem])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    sessions: SessionRepository = Depends(get_session_repository),
) -> list[SessionListItem]:
    return sessions.list(limit=min(limit, 200), offset=offset)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    sessions: SessionRepository = Depends(get_session_repository),
) -> SessionDetail:
    detail = sessions.detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
    return detail


@router.post("/{session_id}/end", response_model=Session)
async def end_session(
    session_id: str,
    sessions: SessionRepository = Depends(get_session_repository),
) -> Session:
    if sessions.get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
    sessions.end(session_id)
    return sessions.get(session_id)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    sessions: SessionRepository = Depends(get_session_repository),
) -> dict:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
    return {"session_id": session_id, "deleted": True}
