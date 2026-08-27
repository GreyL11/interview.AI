from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import documents, health, question
from app.core.config import settings
from app.core.logging import get_logger
from app.documents.service import DocumentError
from app.intelligence.answer_validator import AnswerValidationError
from app.llm.base import LLMError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for directory in (settings.documents_dir, settings.faiss_path.parent, settings.db_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    # Crash recovery: anything still PROCESSING means the process died mid-ingest.
    from app.core.deps import get_document_repository

    stuck = get_document_repository().reset_stuck_processing()
    if stuck:
        logger.warning("recovered_stuck_documents count=%d", stuck)

    yield


app = FastAPI(title="Interview Coach API", version="0.2.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(question.router)
app.include_router(documents.router)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(AnswerValidationError)
async def answer_validation_error_handler(request: Request, exc: AnswerValidationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(DocumentError)
async def document_error_handler(request: Request, exc: DocumentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
