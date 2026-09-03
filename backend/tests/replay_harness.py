"""Reusable transcript replay harness for realistic multi-turn scenarios.

Feeds a sequence of timestamped utterances through the REAL LiveSession /
QuestionDetector pipeline, with fake external dependencies (LLM, retriever,
storage) -- no production logic is reimplemented here.

Deterministic by construction, not by mocking time globally: `at_ms` in each
ReplayEvent becomes the `now` passed straight to
QuestionDetector.inspect(now=...) (already an injectable parameter), so
merge-window / stabilization-eligibility decisions are exercised exactly as
they would be in production, without any real sleeping. The one genuinely
time-based mechanism, the stabilization hold in LiveSession._delayed_ask, is
a real (but tiny, test-configured) asyncio.sleep -- see `stabilization_ms`.
"""

import asyncio
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.documents.schemas import utcnow
from app.llm.base import LLMClient
from app.memory.base import SessionMemory
from app.memory.session_memory import InMemorySessionMemory
from app.realtime.events import Event, EventType
from app.realtime.session import LiveSession
from app.retrieval.base import Retriever
from app.retrieval.mock_retriever import MockRetriever
from app.sessions.schemas import Session, TranscriptSource
from app.storage.database import Database
from app.storage.session_repository import SessionRepository
from tests.fakes import SlowStreamingLLM


@dataclass
class ReplayEvent:
    at_ms: int
    text: str
    source: str = "LOOPBACK"
    is_final: bool = True


@dataclass
class ReplayResult:
    """Everything a scenario assertion typically needs, already extracted."""

    events: list[Event] = field(default_factory=list)

    def of(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.type == event_type]

    def detected_questions(self) -> list[str]:
        return [e.data["question"] for e in self.of(EventType.QUESTION_DETECTED)]

    def rejected_texts(self) -> list[str]:
        return [e.data["text"] for e in self.of(EventType.QUESTION_REJECTED)]

    def answer_deltas(self, turn_id: int | None = None) -> list[str]:
        deltas = self.of(EventType.ANSWER_DELTA)
        if turn_id is not None:
            deltas = [e for e in deltas if e.turn_id == turn_id]
        return [e.data["summary"] for e in deltas]

    def completed_turn_ids(self) -> list[int]:
        return [e.turn_id for e in self.of(EventType.ANSWER_COMPLETED)]

    def cancelled_turn_ids(self) -> list[int]:
        return [e.turn_id for e in self.of(EventType.ANSWER_CANCELLED)]


class ReplayHarness:
    """Drives one LiveSession through a scripted conversation.

    `llm` defaults to a zero-delay SlowStreamingLLM (fast, deterministic);
    pass one with `chunk_delay > 0` to exercise genuine overlap between a
    still-streaming answer and the next question.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        memory: SessionMemory | None = None,
        memory_factory=None,
        stabilization_ms: int = 1,
        monkeypatch=None,
    ) -> None:
        self.db = Database(Path(tempfile.gettempdir()) / f"_replay_{uuid.uuid4().hex}.db")
        self.sessions = SessionRepository(self.db)
        self.session_id = str(uuid.uuid4())
        self.sessions.create(Session(session_id=self.session_id, started_at=utcnow()))
        self.llm = llm if llm is not None else SlowStreamingLLM(chunk_delay=0)
        if memory is None:
            # memory_factory takes the SessionRepository, for implementations
            # (SqliteSessionMemory) that read real persisted turns back.
            memory = (
                memory_factory(self.sessions) if memory_factory is not None
                else InMemorySessionMemory()
            )
        self.live = LiveSession(
            session_id=self.session_id,
            sessions=self.sessions,
            retriever=retriever if retriever is not None else MockRetriever(),
            llm=self.llm,
            memory=memory,
        )
        self.result = ReplayResult()
        self.live.subscribe(self._collect)

        # A tiny stabilization window keeps replay fast without disabling the
        # mechanism under test; use monkeypatch (pytest fixture) so it's
        # restored automatically, matching the rest of the test suite's style.
        # Both hold budgets, so a scenario's timing does not depend on which
        # Finality branch the detector picked -- see question_detector.Finality.
        for field in ("question_stabilization_ms", "question_hold_incomplete_ms"):
            if monkeypatch is not None:
                monkeypatch.setattr(settings, field, stabilization_ms)
            else:
                setattr(settings, field, stabilization_ms)

    async def _collect(self, ev: Event) -> None:
        self.result.events.append(ev)

    async def play(
        self, events: list[ReplayEvent], settle_between: bool = True
    ) -> ReplayResult:
        """Replay a conversation.

        `settle_between=True` (default) lets each answer finish before the
        next utterance -- the normal case, where the interviewer speaks again
        seconds later and the previous answer has already streamed.
        Pass False to model speech arriving *while* the previous answer is
        still streaming, which is what makes supersession/cancellation
        scenarios (correction, rapid-fire) meaningful.
        """
        for ev in events:
            source = TranscriptSource[ev.source]
            await self.live.on_transcript(
                ev.text, source, ev.is_final, now=ev.at_ms / 1000
            )
            if settle_between:
                await self.settle()
        await self.settle()
        return self.result

    async def settle(self, rounds: int = 5) -> None:
        """Let any pending stabilization timer and/or in-flight answer task
        actually finish (cancelled or completed) before assertions run."""
        for _ in range(rounds):
            pending = [
                t for t in (self.live._pending_ask, self.live._task)
                if t is not None and not t.done()
            ]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def dispose(self) -> None:
        """Sync teardown: release the sqlite handle and remove the temp file.
        Safe to call from a non-async fixture."""
        path = self.db.db_path
        self.db.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(f"{path}{suffix}").unlink(missing_ok=True)
            except OSError:
                # Database keeps thread-local connections and memory reads run
                # via asyncio.to_thread, so a worker thread's handle can still
                # be open on Windows. The file is in the OS temp dir; leaving
                # it is preferable to failing an otherwise-passing test.
                pass

    async def close(self) -> None:
        """Full teardown including LiveSession.close() -- needs a running
        loop, so call it from inside a test, not a sync fixture."""
        await self.live.close()
        self.dispose()
