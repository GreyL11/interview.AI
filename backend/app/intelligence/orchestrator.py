import time

from app.core.logging import get_logger
from app.intelligence.answer_validator import validate
from app.intelligence.classifier import classify
from app.intelligence.router import Route, route_for
from app.llm.base import LLMClient
from app.llm.prompts import build_prompt
from app.memory.session_memory import session_memory
from app.retrieval.base import Retriever
from app.schemas.answer import StructuredResponse
from app.schemas.classification import Classification

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self._retriever = retriever
        self._llm = llm

    async def handle(self, question: str, session_id: str | None) -> StructuredResponse:
        start = time.monotonic()
        logger.info("request_received session_id=%s", session_id)

        classification = classify(question)
        logger.info("classification_completed category=%s confidence=%.2f", classification.category, classification.confidence)

        route = route_for(classification.category)
        logger.info("route_selected route=%s", route)

        context = await self._gather_context(question, classification, route)
        history = session_memory.get_history(session_id) if route == Route.FOLLOW_UP else []

        prompt = build_prompt(question, classification.category, context, history)
        answer = await self._llm.generate_answer(prompt)
        answer = validate(answer, classification)

        session_memory.append_turn(session_id, question, answer.summary)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("request_completed elapsed_ms=%.1f", elapsed_ms)

        return StructuredResponse(question=question, classification=classification, answer=answer)

    async def _gather_context(self, question: str, classification: Classification, route: Route) -> list[str]:
        if route == Route.RAG or classification.requires_rag:
            return await self._retriever.retrieve(question)
        return []
