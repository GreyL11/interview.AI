import time

from app.core.logging import get_logger
from app.documents.schemas import PERSONAL_KNOWLEDGE_TYPES, KnowledgeType, RetrievedChunk
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

#: Routes allowed to pull the user's personal documents. Technical routes are
#: deliberately absent: a SYSTEM_DESIGN answer must not quietly acquire
#: first-person claims sourced from a resume.
_PERSONAL_ROUTES = {Route.RAG}


class Orchestrator:
    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self._retriever = retriever
        self._llm = llm

    async def handle(self, question: str, session_id: str | None) -> StructuredResponse:
        start = time.monotonic()
        logger.info("request_received session_id=%s", session_id)

        classification = classify(question)
        logger.info(
            "classification_completed category=%s confidence=%.2f",
            classification.category, classification.confidence,
        )

        route = route_for(classification.category)
        logger.info("route_selected route=%s", route)

        chunks = await self._gather_context(question, classification, route)
        context_found = bool(chunks)
        context = [c.as_context() for c in chunks]

        history = session_memory.get_history(session_id) if route == Route.FOLLOW_UP else []

        prompt = build_prompt(question, classification.category, context, history)
        answer = await self._llm.generate_answer(prompt)
        answer = validate(answer, classification, context_found=context_found)

        session_memory.append_turn(session_id, question, answer.summary)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request_completed elapsed_ms=%.1f context_found=%s chunks=%d",
            elapsed_ms, context_found, len(chunks),
        )

        return StructuredResponse(
            question=question, classification=classification, answer=answer
        )

    async def _gather_context(
        self, question: str, classification: Classification, route: Route
    ) -> list[RetrievedChunk]:
        if route in _PERSONAL_ROUTES or classification.requires_rag:
            return await self._retriever.retrieve(
                question, knowledge_types=list(PERSONAL_KNOWLEDGE_TYPES)
            )
        if route == Route.FOLLOW_UP:
            # A follow-up inherits the previous question's subject, which may well
            # be personal ("and how did you test it?"). Retrieve, but let the
            # similarity threshold decide whether anything is actually relevant.
            return await self._retriever.retrieve(
                question, knowledge_types=list(PERSONAL_KNOWLEDGE_TYPES)
            )
        return []


def knowledge_types_for(route: Route) -> list[KnowledgeType] | None:
    return list(PERSONAL_KNOWLEDGE_TYPES) if route in _PERSONAL_ROUTES else None
