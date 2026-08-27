import asyncio
import shutil
import uuid
from pathlib import Path

from app.chunking.base import Chunker
from app.core.config import settings
from app.core.logging import get_logger
from app.documents.parsers.base import ParserRegistry
from app.documents.schemas import (
    EXTENSION_TO_FILE_TYPE,
    DeleteResponse,
    Document,
    DocumentStatus,
    FileType,
    IngestResponse,
    KnowledgeType,
    utcnow,
)
from app.embeddings.base import EmbeddingProvider
from app.storage.chunk_repository import ChunkRepository
from app.storage.database import Database
from app.storage.document_repository import DocumentRepository
from app.vector_store.base import VectorStore

logger = get_logger(__name__)


class DocumentError(Exception):
    """Raised for caller-fixable problems: unsupported type, unknown id."""


def detect_file_type(filename: str) -> FileType:
    suffix = Path(filename).suffix.lower()
    file_type = EXTENSION_TO_FILE_TYPE.get(suffix)
    if file_type is None:
        supported = ", ".join(sorted(EXTENSION_TO_FILE_TYPE))
        raise DocumentError(f"Unsupported file type '{suffix}'. Supported: {supported}")
    return file_type


class DocumentService:
    def __init__(
        self,
        db: Database,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        storage_dir: Path | None = None,
    ) -> None:
        self._db = db
        self._documents = documents
        self._chunks = chunks
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._vectors = vector_store
        self._storage_dir = storage_dir or settings.documents_dir
        self._ingest_lock = asyncio.Lock()

    # ---------------------------------------------------------------- upload

    async def upload(self, filename: str, content: bytes, knowledge_type: KnowledgeType) -> Document:
        file_type = detect_file_type(filename)
        document_id = str(uuid.uuid4())

        target_dir = self._storage_dir / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / Path(filename).name
        await asyncio.to_thread(path.write_bytes, content)

        document = Document(
            document_id=document_id,
            filename=Path(filename).name,
            file_type=file_type,
            knowledge_type=knowledge_type,
            title=Path(filename).stem,
            source=str(path),
            created_at=utcnow(),
            status=DocumentStatus.UPLOADED,
        )
        self._documents.create(document)
        logger.info(
            "document_uploaded id=%s type=%s knowledge=%s bytes=%d",
            document_id, file_type.value, knowledge_type.value, len(content),
        )
        return document

    # ---------------------------------------------------------------- ingest

    async def ingest(self, document_id: str) -> IngestResponse:
        document = self._documents.get(document_id)
        if document is None:
            raise DocumentError(f"Unknown document '{document_id}'")

        # Serialised: ingestion mutates a single shared FAISS index, and vector
        # id allocation must not interleave with another ingest's index write.
        async with self._ingest_lock:
            return await self._ingest_locked(document)

    async def _ingest_locked(self, document: Document) -> IngestResponse:
        document_id = document.document_id
        self._documents.update_status(document_id, DocumentStatus.PROCESSING)

        # Re-ingest must not double up chunks or leak the previous vectors.
        await self._purge_derived_data(document_id)

        vector_ids: list[int] = []
        try:
            path = Path(document.source)
            if not path.exists():
                raise DocumentError(f"Stored file is missing: {path}")

            normalized = await asyncio.to_thread(
                self._parsers.parse, path, document.file_type, document_id
            )
            chunks = self._chunker.chunk(
                normalized,
                knowledge_type=document.knowledge_type.value,
                source=document.filename,
            )
            if not chunks:
                raise DocumentError("Document produced no chunks")

            vectors = await asyncio.to_thread(self._embedder.embed, [c.text for c in chunks])

            vector_ids = self._db.allocate_vector_ids(len(chunks))
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.vector_id = vector_id

            await asyncio.to_thread(self._vectors.add, vector_ids, vectors)
            self._chunks.create_many(chunks)
            await asyncio.to_thread(self._vectors.persist)

            # READY is set last. Until this line the retrieval join filters every
            # one of these chunks out, so a failure above is invisible, not partial.
            self._documents.update_status(
                document_id, DocumentStatus.READY, chunk_count=len(chunks)
            )
            logger.info("document_ingested id=%s chunks=%d", document_id, len(chunks))
            return IngestResponse(
                document_id=document_id, status=DocumentStatus.READY, chunk_count=len(chunks)
            )

        except Exception as exc:
            logger.exception("document_ingest_failed id=%s", document_id)
            await self._rollback(document_id, vector_ids)
            self._documents.update_status(
                document_id, DocumentStatus.FAILED, error=str(exc), chunk_count=0
            )
            return IngestResponse(
                document_id=document_id,
                status=DocumentStatus.FAILED,
                chunk_count=0,
                error=str(exc),
            )

    async def _rollback(self, document_id: str, vector_ids: list[int]) -> None:
        """Best-effort cleanup. Correctness does not depend on this succeeding —
        the READY filter already hides the data — but leaving orphans behind
        would waste disk and memory."""
        try:
            self._chunks.delete_by_document(document_id)
            if vector_ids:
                await asyncio.to_thread(self._vectors.remove, vector_ids)
                await asyncio.to_thread(self._vectors.persist)
        except Exception:
            logger.exception("rollback_incomplete id=%s", document_id)

    async def _purge_derived_data(self, document_id: str) -> None:
        existing = self._chunks.get_vector_ids(document_id)
        if existing:
            await asyncio.to_thread(self._vectors.remove, existing)
        self._chunks.delete_by_document(document_id)

    # ---------------------------------------------------------------- queries

    def get(self, document_id: str) -> Document:
        document = self._documents.get(document_id)
        if document is None:
            raise DocumentError(f"Unknown document '{document_id}'")
        return document

    def list(
        self,
        knowledge_type: KnowledgeType | None = None,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        return self._documents.list(knowledge_type=knowledge_type, status=status)

    # ---------------------------------------------------------------- delete

    async def delete(self, document_id: str) -> DeleteResponse:
        document = self._documents.get(document_id)
        if document is None:
            raise DocumentError(f"Unknown document '{document_id}'")

        async with self._ingest_lock:
            vector_ids = self._chunks.get_vector_ids(document_id)
            removed = 0
            if vector_ids:
                removed = await asyncio.to_thread(self._vectors.remove, vector_ids)
                await asyncio.to_thread(self._vectors.persist)

            chunk_count = self._chunks.delete_by_document(document_id)
            self._documents.delete(document_id)

            folder = self._storage_dir / document_id
            if folder.exists():
                await asyncio.to_thread(shutil.rmtree, folder, True)

        logger.info(
            "document_deleted id=%s chunks=%d vectors=%d", document_id, chunk_count, removed
        )
        return DeleteResponse(
            document_id=document_id,
            deleted=True,
            chunks_removed=chunk_count,
            vectors_removed=removed,
        )
