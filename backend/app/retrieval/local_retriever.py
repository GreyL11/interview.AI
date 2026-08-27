import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.documents.schemas import KnowledgeType, RetrievedChunk
from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import Retriever
from app.storage.chunk_repository import ChunkRepository
from app.vector_store.base import VectorStore

logger = get_logger(__name__)


class LocalRetriever(Retriever):
    """embed_query -> FAISS -> SQLite join -> chunks. Nothing leaves the machine."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        chunks: ChunkRepository,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._chunks = chunks

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_types: list[KnowledgeType] | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else settings.rag_top_k
        threshold = (
            min_similarity if min_similarity is not None else settings.rag_min_similarity
        )
        if k <= 0:
            return []

        vector = await asyncio.to_thread(self._embedder.embed_query, question)

        # Over-fetch: the SQLite join drops hits whose document isn't READY or
        # doesn't match the knowledge_type filter, so asking FAISS for exactly k
        # would silently return fewer than k usable chunks.
        hits = await asyncio.to_thread(
            self._vectors.search, vector, k * settings.rag_overfetch
        )
        if not hits:
            logger.info("retrieval_empty reason=no_vectors question_len=%d", len(question))
            return []

        resolved = await asyncio.to_thread(
            self._chunks.resolve_vectors, hits, knowledge_types
        )
        above = [c for c in resolved if c.score >= threshold][:k]

        logger.info(
            "retrieval_completed hits=%d resolved=%d kept=%d threshold=%.2f",
            len(hits), len(resolved), len(above), threshold,
        )
        return above
