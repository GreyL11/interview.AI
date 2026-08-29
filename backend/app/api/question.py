from fastapi import APIRouter, Depends

from app.core.deps import get_llm_client
from app.intelligence.orchestrator import Orchestrator
from app.retrieval.mock_retriever import MockRetriever
from app.schemas.answer import StructuredResponse
from app.schemas.question import QuestionRequest

router = APIRouter()


def get_orchestrator() -> Orchestrator:
    # Built per request, deliberately: it holds nothing expensive (the LLM
    # client and retriever are themselves cached), and caching it here would
    # pin the *old* LLM client after Settings saves a new key -- the settings
    # API can only clear `get_llm_client`, not every object holding a reference
    # to what it returned.
    return Orchestrator(retriever=MockRetriever(), llm=get_llm_client())


@router.post("/question", response_model=StructuredResponse)
async def ask_question(
    request: QuestionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> StructuredResponse:
    return await orchestrator.handle(request.question, request.session_id)
