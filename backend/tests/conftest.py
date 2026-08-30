import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings as _settings
from app.core.secrets import InMemorySecretStore, set_secret_store

# Tests assert against the code's own defaults, so they must not inherit the
# developer's .env -- otherwise a stale local value (an old STT priority, a
# different QUESTION_MIN_WORDS) fails tests that have nothing to do with the
# change being made, and a real regression can hide behind a local override.
# The singleton is mutated in place because modules already hold a reference
# to it; rebinding the name here would not reach them.
_pristine = Settings(_env_file=None)
for _name in type(_pristine).model_fields:
    setattr(_settings, _name, getattr(_pristine, _name))

# The same rule, one layer down, and the one that was missed: the OS credential
# store is developer state too.
#
# Every `TestClient(app)` runs the application lifespan, and the lifespan calls
# `load_persisted_secrets()`. Against the real store that reads whatever key the
# developer has saved and writes it into `settings` -- *after* a test's own
# monkeypatch has run -- so a machine with a key saved behaves differently from
# one without. Two tests failed exactly this way, and only on machines where
# someone had used the app.
#
# It also *wrote* to the real store: the obsolete-secret migration deletes
# entries, so running the suite could remove a credential from the developer's
# own Credential Manager.
#
# Installed here, at import time, because conftest is imported before any test
# or fixture runs -- a fixture would be too late for the lifespan of a client
# built by an earlier fixture in the same test.
set_secret_store(InMemorySecretStore())

from app.api.question import get_orchestrator  # noqa: E402
from app.chunking.semantic_chunker import SemanticChunker
from app.core.deps import get_document_service
from app.documents.parsers.base import ParserRegistry
from app.documents.parsers.docx import DocxParser
from app.documents.parsers.markdown import MarkdownParser
from app.documents.parsers.pdf import PdfParser
from app.documents.parsers.text import TextParser
from app.documents.service import DocumentService
from app.intelligence.orchestrator import Orchestrator
from app.main import app
from app.retrieval.local_retriever import LocalRetriever
from app.retrieval.mock_retriever import MockRetriever
from app.storage.chunk_repository import ChunkRepository
from app.storage.database import Database
from app.storage.document_repository import DocumentRepository
from app.vector_store.faiss_store import FaissVectorStore
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def parsers() -> ParserRegistry:
    return ParserRegistry([TextParser(), MarkdownParser(), PdfParser(), DocxParser()])


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def database(tmp_path) -> Database:
    db = Database(tmp_path / "metadata" / "app.db")
    yield db
    db.close()


@pytest.fixture
def vector_store(tmp_path, embedder) -> FaissVectorStore:
    return FaissVectorStore(embedder.dimension, tmp_path / "faiss" / "index.faiss")


@pytest.fixture
def documents_repo(database) -> DocumentRepository:
    return DocumentRepository(database)


@pytest.fixture
def chunks_repo(database) -> ChunkRepository:
    return ChunkRepository(database)


@pytest.fixture
def service(
    tmp_path, database, documents_repo, chunks_repo, parsers, embedder, vector_store
) -> DocumentService:
    return DocumentService(
        db=database,
        documents=documents_repo,
        chunks=chunks_repo,
        parsers=parsers,
        chunker=SemanticChunker(chunk_size=400, chunk_overlap=80),
        embedder=embedder,
        vector_store=vector_store,
        storage_dir=tmp_path / "documents",
    )


@pytest.fixture
def retriever(embedder, vector_store, chunks_repo) -> LocalRetriever:
    return LocalRetriever(embedder, vector_store, chunks_repo)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(fake_llm):
    """API client with the LLM faked out. Retrieval stays empty here; tests that
    need real retrieval use the `service`/`retriever` fixtures directly."""
    app.dependency_overrides[get_orchestrator] = lambda: Orchestrator(
        retriever=MockRetriever(), llm=fake_llm
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def documents_client(fake_llm, service):
    """API client wired to a tmp_path-backed document service."""
    app.dependency_overrides[get_orchestrator] = lambda: Orchestrator(
        retriever=MockRetriever(), llm=fake_llm
    )
    app.dependency_overrides[get_document_service] = lambda: service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
