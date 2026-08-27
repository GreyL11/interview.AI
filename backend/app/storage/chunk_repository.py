import json
import sqlite3

from app.documents.schemas import (
    Chunk,
    DocumentStatus,
    KnowledgeType,
    RetrievedChunk,
)
from app.storage.database import Database


def _to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        vector_id=row["vector_id"],
        token_count=row["token_count"],
        metadata=json.loads(row["metadata"]),
    )


class ChunkRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_many(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._db.write() as conn:
            conn.executemany(
                """INSERT INTO chunks
                   (chunk_id, document_id, chunk_index, text, vector_id, token_count, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.text,
                        c.vector_id,
                        c.token_count,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def get_by_document(self, document_id: str) -> list[Chunk]:
        rows = self._db.connect().execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [_to_chunk(r) for r in rows]

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._db.connect().execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        ).fetchall()
        return [_to_chunk(r) for r in rows]

    def get_vector_ids(self, document_id: str) -> list[int]:
        rows = self._db.connect().execute(
            "SELECT vector_id FROM chunks WHERE document_id = ? AND vector_id IS NOT NULL",
            (document_id,),
        ).fetchall()
        return [r["vector_id"] for r in rows]

    def delete_by_document(self, document_id: str) -> int:
        with self._db.write() as conn:
            cur = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        return cur.rowcount

    def resolve_vectors(
        self,
        scored_vector_ids: list[tuple[int, float]],
        knowledge_types: list[KnowledgeType] | None = None,
    ) -> list[RetrievedChunk]:
        """Turn FAISS hits into chunks, joined to their parent document.

        This join is the safety net for the whole design: it requires the parent
        document to be READY, so vectors belonging to a half-ingested, failed, or
        deleted document simply produce no row and vanish from results. No
        soft-delete flag, no index rebuild, no filtering code.
        """
        if not scored_vector_ids:
            return []

        scores = dict(scored_vector_ids)
        placeholders = ",".join("?" * len(scores))
        params: list = list(scores.keys())

        sql = f"""
            SELECT c.chunk_id, c.document_id, c.text, c.vector_id,
                   d.knowledge_type, d.title
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE c.vector_id IN ({placeholders})
              AND d.status = ?
        """
        params.append(DocumentStatus.READY.value)

        if knowledge_types:
            kt_placeholders = ",".join("?" * len(knowledge_types))
            sql += f" AND d.knowledge_type IN ({kt_placeholders})"
            params.extend(kt.value for kt in knowledge_types)

        rows = self._db.connect().execute(sql, params).fetchall()

        retrieved = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=scores[r["vector_id"]],
                knowledge_type=KnowledgeType(r["knowledge_type"]),
                title=r["title"],
            )
            for r in rows
        ]
        # SQL loses the similarity ordering; restore it.
        retrieved.sort(key=lambda c: c.score, reverse=True)
        return retrieved

    def count(self) -> int:
        return self._db.connect().execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
