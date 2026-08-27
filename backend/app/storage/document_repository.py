# Deferred annotations: this class defines a `list()` method, which shadows the
# builtin inside the class body and breaks `list[...]` annotations at runtime.
from __future__ import annotations

import sqlite3

from app.documents.schemas import (
    Document,
    DocumentStatus,
    FileType,
    KnowledgeType,
    utcnow,
)
from app.storage.database import Database


def _to_document(row: sqlite3.Row) -> Document:
    return Document(
        document_id=row["document_id"],
        filename=row["filename"],
        file_type=FileType(row["file_type"]),
        knowledge_type=KnowledgeType(row["knowledge_type"]),
        title=row["title"],
        source=row["source"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        status=DocumentStatus(row["status"]),
        error=row["error"],
        chunk_count=row["chunk_count"],
    )


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, document: Document) -> Document:
        with self._db.write() as conn:
            conn.execute(
                """INSERT INTO documents
                   (document_id, filename, file_type, knowledge_type, title, source,
                    created_at, ingested_at, status, error, chunk_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    document.document_id,
                    document.filename,
                    document.file_type.value,
                    document.knowledge_type.value,
                    document.title,
                    document.source,
                    document.created_at.isoformat(),
                    document.ingested_at.isoformat() if document.ingested_at else None,
                    document.status.value,
                    document.error,
                    document.chunk_count,
                ),
            )
        return document

    def get(self, document_id: str) -> Document | None:
        row = self._db.connect().execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _to_document(row) if row else None

    def list(
        self,
        knowledge_type: KnowledgeType | None = None,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        sql = "SELECT * FROM documents"
        clauses, params = [], []
        if knowledge_type is not None:
            clauses.append("knowledge_type = ?")
            params.append(knowledge_type.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = self._db.connect().execute(sql, params).fetchall()
        return [_to_document(r) for r in rows]

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        ingested_at = utcnow().isoformat() if status == DocumentStatus.READY else None
        with self._db.write() as conn:
            conn.execute(
                """UPDATE documents
                   SET status = ?,
                       error = ?,
                       chunk_count = COALESCE(?, chunk_count),
                       ingested_at = COALESCE(?, ingested_at)
                   WHERE document_id = ?""",
                (status.value, error, chunk_count, ingested_at, document_id),
            )

    def delete(self, document_id: str) -> bool:
        with self._db.write() as conn:
            cur = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cur.rowcount > 0

    def reset_stuck_processing(self) -> int:
        """Crash recovery: a document left PROCESSING means the process died
        mid-ingest. Mark it FAILED so it is visibly re-ingestable rather than
        silently stuck."""
        with self._db.write() as conn:
            cur = conn.execute(
                "UPDATE documents SET status = ?, error = ? WHERE status = ?",
                (
                    DocumentStatus.FAILED.value,
                    "Ingestion interrupted by shutdown; re-ingest required.",
                    DocumentStatus.PROCESSING.value,
                ),
            )
        return cur.rowcount
