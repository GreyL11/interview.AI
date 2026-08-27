from app.documents.schemas import KnowledgeType, RetrievedChunk
from app.retrieval.base import Retriever


class MockRetriever(Retriever):
    """Always-empty retriever. Kept past Phase 1 because it is the cleanest way
    to test the no-personal-context path — the case where the LLM must stay in
    the conditional voice."""

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_types: list[KnowledgeType] | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        return []
