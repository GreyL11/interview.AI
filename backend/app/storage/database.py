import sqlite3
import threading
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# ponytail: one process-wide write lock instead of a connection pool. SQLite in
# WAL mode allows concurrent readers but a single writer; serialising writes here
# is simpler and more reliable than catching "database is locked" everywhere.
# Revisit only if this stops being a single-user desktop app.
_write_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    file_type      TEXT NOT NULL,
    knowledge_type TEXT NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    ingested_at    TEXT,
    status         TEXT NOT NULL,
    error          TEXT,
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    progress       TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    vector_id   INTEGER UNIQUE,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS vector_seq (next_id INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    status     TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    config     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    question      TEXT NOT NULL,
    category      TEXT NOT NULL DEFAULT '',
    domain        TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 0,
    answer        TEXT,
    context_found INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    latency_ms    INTEGER,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_id    INTEGER,
    source     TEXT NOT NULL,
    is_final   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    started_at TEXT,
    ended_at   TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id  INTEGER NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    score    REAL NOT NULL,
    rank     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id          TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary             TEXT NOT NULL DEFAULT '',
    topics              TEXT NOT NULL DEFAULT '[]',
    covered_through_seq INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document  ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vector    ON chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_documents_lookup ON documents(knowledge_type, status);
CREATE INDEX IF NOT EXISTS idx_turns_session    ON turns(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_transcripts_sess ON transcripts(session_id, is_final);
CREATE INDEX IF NOT EXISTS idx_hits_turn        ON retrieval_hits(turn_id);
"""

CURRENT_VERSION = 3

#: Columns added after the first release, applied to databases that predate
#: them. SQLite has no "ADD COLUMN IF NOT EXISTS", and an existing install must
#: not be recreated -- it holds the user's documents and session history.
_ADDED_COLUMNS = {
    # v3: what a slow ingest is currently doing, so the UI can say "reading
    # scanned page 3 of 12" instead of an unexplained spinner.
    "documents": [("progress", "TEXT")],
}


class Database:
    """Owns the SQLite connection lifecycle. Connections are per-thread because
    ingestion runs on worker threads via asyncio.to_thread."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self.connect()
        with _write_lock, conn:
            # CREATE TABLE IF NOT EXISTS, so this is a no-op on an existing
            # database and creates a current-shaped one from scratch otherwise.
            conn.executescript(SCHEMA)
            self._add_missing_columns(conn)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION,))
            else:
                conn.execute("UPDATE schema_version SET version = ?", (CURRENT_VERSION,))
            if conn.execute("SELECT COUNT(*) AS n FROM vector_seq").fetchone()["n"] == 0:
                conn.execute("INSERT INTO vector_seq (next_id) VALUES (1)")

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection) -> None:
        """Bring an older database up to the current shape, in place.

        Driven off the live table info rather than the recorded version number,
        so it is idempotent and stays correct even if a database was created by
        a build whose version counter disagrees. Only ever *adds* nullable
        columns -- nothing here can destroy a user's data.
        """
        for table, columns in _ADDED_COLUMNS.items():
            existing = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            for name, declaration in columns:
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
                    logger.info("schema_column_added table=%s column=%s", table, name)

    def write(self):
        """Context manager for a serialised write transaction."""
        return _WriteTransaction(self.connect())

    def allocate_vector_ids(self, count: int) -> list[int]:
        """Reserve a contiguous block of vector ids. Ids are never reused, so a
        deleted vector's id can't be resurrected against a stale FAISS entry."""
        conn = self.connect()
        with _write_lock, conn:
            start = conn.execute("SELECT next_id FROM vector_seq").fetchone()["next_id"]
            conn.execute("UPDATE vector_seq SET next_id = ?", (start + count,))
        return list(range(start, start + count))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class _WriteTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        _write_lock.acquire()
        self._conn.execute("BEGIN")
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            _write_lock.release()
        return False
