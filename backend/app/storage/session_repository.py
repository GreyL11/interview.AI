# Deferred annotations: this class defines a `list()` method, which shadows the
# builtin inside the class body and breaks `list[...]` annotations at runtime.
from __future__ import annotations

import json
import sqlite3

from app.documents.schemas import utcnow
from app.schemas.answer import Answer
from app.sessions.schemas import (
    RetrievalHit,
    Session,
    SessionDetail,
    SessionListItem,
    SessionStatus,
    SessionSummary,
    TranscriptEntry,
    TranscriptSource,
    Turn,
    TurnStatus,
)
from app.storage.database import Database


def _to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        status=SessionStatus(row["status"]),
        title=row["title"],
        config=json.loads(row["config"]),
    )


def _to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        seq=row["seq"],
        question=row["question"],
        category=row["category"],
        domain=row["domain"],
        confidence=row["confidence"],
        answer=Answer.model_validate_json(row["answer"]) if row["answer"] else None,
        context_found=bool(row["context_found"]),
        status=TurnStatus(row["status"]),
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )


def _to_transcript(row: sqlite3.Row) -> TranscriptEntry:
    return TranscriptEntry(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        source=TranscriptSource(row["source"]),
        is_final=bool(row["is_final"]),
        text=row["text"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---------------------------------------------------------------- sessions

    def create(self, session: Session) -> Session:
        with self._db.write() as conn:
            conn.execute(
                """INSERT INTO sessions (session_id, started_at, ended_at, status, title, config)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session.session_id,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.status.value,
                    session.title,
                    json.dumps(session.config),
                ),
            )
        return session

    def get(self, session_id: str) -> Session | None:
        row = self._db.connect().execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return _to_session(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> list[SessionListItem]:
        rows = self._db.connect().execute(
            """SELECT s.*, (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
               FROM sessions s ORDER BY s.started_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        return [
            SessionListItem(
                session_id=r["session_id"],
                started_at=r["started_at"],
                ended_at=r["ended_at"],
                status=SessionStatus(r["status"]),
                title=r["title"],
                turn_count=r["turn_count"],
            )
            for r in rows
        ]

    def end(self, session_id: str) -> None:
        with self._db.write() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, ended_at = ? WHERE session_id = ?",
                (SessionStatus.ENDED.value, utcnow().isoformat(), session_id),
            )

    def set_title(self, session_id: str, title: str) -> None:
        with self._db.write() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id)
            )

    def delete(self, session_id: str) -> bool:
        with self._db.write() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0

    def close_stale_active(self) -> int:
        """A session left ACTIVE means the process died mid-session."""
        with self._db.write() as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = ?, ended_at = ? WHERE status = ?",
                (SessionStatus.ENDED.value, utcnow().isoformat(), SessionStatus.ACTIVE.value),
            )
        return cur.rowcount

    # ------------------------------------------------------------------- turns

    def next_seq(self, session_id: str) -> int:
        row = self._db.connect().execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["n"]

    def create_turn(self, turn: Turn) -> Turn:
        with self._db.write() as conn:
            cur = conn.execute(
                """INSERT INTO turns
                   (session_id, seq, question, category, domain, confidence, answer,
                    context_found, status, latency_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    turn.session_id,
                    turn.seq,
                    turn.question,
                    turn.category,
                    turn.domain,
                    turn.confidence,
                    turn.answer.model_dump_json() if turn.answer else None,
                    int(turn.context_found),
                    turn.status.value,
                    turn.latency_ms,
                    turn.created_at.isoformat(),
                ),
            )
        return turn.model_copy(update={"turn_id": cur.lastrowid})

    def complete_turn(
        self,
        turn_id: int,
        answer: Answer,
        context_found: bool,
        latency_ms: int,
        hits: list[RetrievalHit] | None = None,
    ) -> None:
        with self._db.write() as conn:
            conn.execute(
                """UPDATE turns SET answer = ?, context_found = ?, status = ?, latency_ms = ?
                   WHERE turn_id = ?""",
                (
                    answer.model_dump_json(),
                    int(context_found),
                    TurnStatus.ANSWERED.value,
                    latency_ms,
                    turn_id,
                ),
            )
            if hits:
                conn.executemany(
                    "INSERT INTO retrieval_hits (turn_id, chunk_id, score, rank) VALUES (?,?,?,?)",
                    [(turn_id, h.chunk_id, h.score, h.rank) for h in hits],
                )

    def mark_turn(self, turn_id: int, status: TurnStatus) -> None:
        with self._db.write() as conn:
            conn.execute("UPDATE turns SET status = ? WHERE turn_id = ?", (status.value, turn_id))

    def get_turns(self, session_id: str) -> list[Turn]:
        rows = self._db.connect().execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [_to_turn(r) for r in rows]

    def get_answered_turns(self, session_id: str) -> list[Turn]:
        rows = self._db.connect().execute(
            "SELECT * FROM turns WHERE session_id = ? AND status = ? ORDER BY seq",
            (session_id, TurnStatus.ANSWERED.value),
        ).fetchall()
        return [_to_turn(r) for r in rows]

    def get_hits(self, turn_id: int) -> list[RetrievalHit]:
        rows = self._db.connect().execute(
            "SELECT chunk_id, score, rank FROM retrieval_hits WHERE turn_id = ? ORDER BY rank",
            (turn_id,),
        ).fetchall()
        return [RetrievalHit(chunk_id=r["chunk_id"], score=r["score"], rank=r["rank"]) for r in rows]

    # ------------------------------------------------------------- transcripts

    def add_transcript(self, entry: TranscriptEntry) -> TranscriptEntry:
        with self._db.write() as conn:
            cur = conn.execute(
                """INSERT INTO transcripts
                   (session_id, turn_id, source, is_final, text, started_at, ended_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    entry.session_id,
                    entry.turn_id,
                    entry.source.value,
                    int(entry.is_final),
                    entry.text,
                    entry.started_at.isoformat() if entry.started_at else None,
                    entry.ended_at.isoformat() if entry.ended_at else None,
                ),
            )
        return entry.model_copy(update={"id": cur.lastrowid})

    def get_transcript(self, session_id: str, finals_only: bool = True) -> list[TranscriptEntry]:
        sql = "SELECT * FROM transcripts WHERE session_id = ?"
        params: list = [session_id]
        if finals_only:
            sql += " AND is_final = 1"
        sql += " ORDER BY id"
        rows = self._db.connect().execute(sql, params).fetchall()
        return [_to_transcript(r) for r in rows]

    # --------------------------------------------------------------- summaries

    def get_summary(self, session_id: str) -> SessionSummary | None:
        row = self._db.connect().execute(
            "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return SessionSummary(
            session_id=row["session_id"],
            summary=row["summary"],
            topics=json.loads(row["topics"]),
            covered_through_seq=row["covered_through_seq"],
            updated_at=row["updated_at"],
        )

    def upsert_summary(self, summary: SessionSummary) -> None:
        with self._db.write() as conn:
            conn.execute(
                """INSERT INTO session_summaries
                   (session_id, summary, topics, covered_through_seq, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary = excluded.summary,
                       topics = excluded.topics,
                       covered_through_seq = excluded.covered_through_seq,
                       updated_at = excluded.updated_at""",
                (
                    summary.session_id,
                    summary.summary,
                    json.dumps(summary.topics),
                    summary.covered_through_seq,
                    summary.updated_at.isoformat(),
                ),
            )

    def detail(self, session_id: str) -> SessionDetail | None:
        session = self.get(session_id)
        if session is None:
            return None
        return SessionDetail(
            session=session,
            turns=self.get_turns(session_id),
            transcript=self.get_transcript(session_id),
            summary=self.get_summary(session_id),
        )
