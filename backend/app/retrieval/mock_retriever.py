from app.retrieval.base import Retriever


class MockRetriever(Retriever):
    """Phase 1 stand-in for the real RAG pipeline. Returns no context, so the
    LLM is instructed to speak in the conditional ("I would...") rather than
    invent personal experience."""

    async def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        return []
