from functools import lru_cache

from app.chunking.semantic_chunker import SemanticChunker
from app.core.config import settings
from app.documents.parsers.base import ParserRegistry
from app.documents.parsers.docx import DocxParser
from app.documents.parsers.markdown import MarkdownParser
from app.documents.parsers.pdf import PdfParser
from app.documents.parsers.text import TextParser
from app.documents.service import DocumentService
from app.embeddings.base import EmbeddingProvider
from app.embeddings.onnx_embedder import OnnxEmbedder
from app.llm.base import LLMClient
from app.llm.registry import ProviderConfig, build_client
from app.memory.sqlite_memory import SqliteSessionMemory
from app.memory.summarizer import SessionSummarizer
from app.realtime.question_understanding import QuestionUnderstander
from app.retrieval.base import Retriever
from app.retrieval.local_retriever import LocalRetriever
from app.storage.chunk_repository import ChunkRepository
from app.storage.database import Database
from app.storage.document_repository import DocumentRepository
from app.storage.session_repository import SessionRepository
from app.stt.base import SttEngine
from app.stt.faster_whisper_engine import FasterWhisperEngine
from app.vector_store.base import VectorStore
from app.vector_store.faiss_store import FaissVectorStore

# Composition root. Everything is @lru_cache'd so the process shares one
# database, one FAISS index, and one loaded embedding model. Tests override the
# FastAPI dependencies rather than these.


@lru_cache
def get_database() -> Database:
    return Database(settings.db_path)


@lru_cache
def get_document_repository() -> DocumentRepository:
    return DocumentRepository(get_database())


@lru_cache
def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository(get_database())


@lru_cache
def get_embedder() -> EmbeddingProvider:
    return OnnxEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    return FaissVectorStore(get_embedder().dimension, settings.faiss_path)


@lru_cache
def get_parsers() -> ParserRegistry:
    return ParserRegistry([TextParser(), MarkdownParser(), PdfParser(), DocxParser()])


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        db=get_database(),
        documents=get_document_repository(),
        chunks=get_chunk_repository(),
        parsers=get_parsers(),
        chunker=SemanticChunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_retriever() -> Retriever:
    return LocalRetriever(get_embedder(), get_vector_store(), get_chunk_repository())


@lru_cache
def get_session_repository() -> SessionRepository:
    return SessionRepository(get_database())


@lru_cache
def get_session_memory() -> SqliteSessionMemory:
    return SqliteSessionMemory(get_session_repository())


def api_key_for(provider: str) -> str:
    """Read `<provider>_api_key` off Settings, if that field exists.

    The same convention `app.core.secrets`/`app.api.settings_api` already use
    -- a provider's key lives in the field named after it, discovered by name
    rather than through a per-provider branch.
    """
    return getattr(settings, f"{provider}_api_key", "") or ""


def _answer_config() -> ProviderConfig:
    return ProviderConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=api_key_for(settings.llm_provider),
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _understanding_config() -> ProviderConfig:
    """Question Understanding's own provider config -- independent of the
    answer stage's, but defaulting to it field-by-field when unset. See
    `Settings.question_understanding_provider`."""
    provider = settings.question_understanding_provider or settings.llm_provider
    return ProviderConfig(
        provider=provider,
        model=settings.question_understanding_model or settings.llm_model,
        api_key=api_key_for(provider),
        base_url=settings.question_understanding_base_url or settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


@lru_cache
def get_llm_client() -> LLMClient:
    """Answer generation's provider, behind the shared LLMClient interface."""
    return build_client(_answer_config())


@lru_cache
def get_question_understanding_client() -> LLMClient:
    """Question Understanding's provider. Reuses the answer client (and its
    connection pool) whenever the two stages are configured identically,
    which is the common case; only builds a second client when they
    genuinely differ."""
    config = _understanding_config()
    if config == _answer_config():
        return get_llm_client()
    return build_client(config)


@lru_cache
def get_summarizer() -> SessionSummarizer:
    return SessionSummarizer(get_session_repository(), get_llm_client())


@lru_cache
def get_question_understander() -> QuestionUnderstander:
    """The understanding layer, wired to whichever provider Settings names for
    it. Duck-typed against `StructuredCompleter` (see
    app.realtime.question_understanding): any client without a `complete_json`
    method degrades this to the deterministic fallback rather than erroring."""
    client = get_question_understanding_client()
    completer = client if hasattr(client, "complete_json") else None
    return QuestionUnderstander(completer)


@lru_cache
def get_stt_engine() -> SttEngine:
    return FasterWhisperEngine()
