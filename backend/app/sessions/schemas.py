from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.documents.schemas import utcnow
from app.schemas.answer import Answer


class SessionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"


class TurnStatus(StrEnum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TranscriptSource(StrEnum):
    #: Which capture device the audio came from. This is device attribution,
    #: not speaker diarization — reliable precisely because it makes no
    #: acoustic claim about who is talking.
    MIC = "MIC"
    LOOPBACK = "LOOPBACK"
    MANUAL = "MANUAL"


class Session(BaseModel):
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    title: str = ""
    config: dict = Field(default_factory=dict)


class Turn(BaseModel):
    turn_id: int | None = None
    session_id: str
    seq: int
    question: str
    category: str = ""
    domain: str = ""
    confidence: float = 0.0
    answer: Answer | None = None
    context_found: bool = False
    status: TurnStatus = TurnStatus.PENDING
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


class TranscriptEntry(BaseModel):
    id: int | None = None
    session_id: str
    turn_id: int | None = None
    source: TranscriptSource
    is_final: bool
    text: str
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    rank: int


class SessionSummary(BaseModel):
    """Rolling compression of turns that have fallen out of the verbatim window."""

    session_id: str
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    covered_through_seq: int = 0
    updated_at: datetime = Field(default_factory=utcnow)


class SessionListItem(BaseModel):
    session_id: str
    started_at: datetime
    ended_at: datetime | None
    status: SessionStatus
    title: str
    turn_count: int


class SessionDetail(BaseModel):
    session: Session
    turns: list[Turn]
    transcript: list[TranscriptEntry]
    summary: SessionSummary | None = None
