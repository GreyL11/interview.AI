from functools import lru_cache

from fastapi import APIRouter, Depends

from app.intelligence.orchestrator import Orchestrator
from app.llm.gemini_client import GeminiClient
from app.retrieval.mock_retriever import MockRetriever
from app.schemas.answer import StructuredResponse
from app.schemas.question import QuestionRequest

router = APIRouter()


@lru_cache
def get_orchestrator() -> Orchestrator:
    # Lazy + cached: constructing GeminiClient reads GEMINI_API_KEY, and we
    # want a clear error at request time (handled by the app-level LLMError
    # handler) rather than an import-time crash when the key isn't set yet
    # (e.g. running tests).
    return Orchestrator(retriever=MockRetriever(), llm=GeminiClient())


@router.post("/question", response_model=StructuredResponse)
async def ask_question(
    request: QuestionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> StructuredResponse:
    return await orchestrator.handle(request.question, request.session_id)
