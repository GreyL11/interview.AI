import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import documents, health, question, sessions, settings_api, ws
from app.core.auth import TokenMiddleware
from app.core.config import settings
from app.core.logging import get_logger
from app.documents.service import DocumentError
from app.intelligence.answer_validator import AnswerValidationError
from app.llm.base import LLMError
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for directory in (settings.documents_dir, settings.faiss_path.parent, settings.db_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    # Crash recovery: anything still PROCESSING or ACTIVE means the process died
    # partway through. Surface it as a terminal state rather than leaving it stuck.
    from app.core.deps import (
        get_document_repository,
        get_llm_client,
        get_session_repository,
    )

    # Before anything builds a provider: a desktop install ships no .env, so
    # the only place its keys can come from is the OS credential store.
    from app.core.secret_config import load_persisted_secrets

    load_persisted_secrets()

    stuck = get_document_repository().reset_stuck_processing()
    if stuck:
        logger.warning("recovered_stuck_documents count=%d", stuck)

    orphaned = get_session_repository().close_stale_active()
    if orphaned:
        logger.warning("closed_orphaned_sessions count=%d", orphaned)

    # Provider SDK import and client construction were measured at ~2.4s, paid
    # lazily on the first question of a session. Move it here, in a thread so
    # startup does not block on it.
    warm = asyncio.create_task(asyncio.to_thread(get_llm_client().warmup))

    yield

    warm.cancel()

    from app.realtime.manager import session_manager

    await session_manager.close_all()


app = FastAPI(title="Interview Coach API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(tauri\.localhost|localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TokenMiddleware)

app.include_router(health.router)
app.include_router(question.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(settings_api.router)
app.include_router(ws.router)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(AnswerValidationError)
async def answer_validation_error_handler(request: Request, exc: AnswerValidationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(DocumentError)
async def document_error_handler(request: Request, exc: DocumentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
