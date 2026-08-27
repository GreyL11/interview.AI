from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.documents.schemas import utcnow


class EventType(StrEnum):
    # server -> client
    SESSION_STARTED = "session.started"
    SESSION_STATUS = "session.status"
    SESSION_ENDED = "session.ended"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    QUESTION_DETECTED = "question.detected"
    QUESTION_REJECTED = "question.rejected"
    ANSWER_STARTED = "answer.started"
    ANSWER_RETRIEVING = "answer.retrieving"
    ANSWER_DELTA = "answer.delta"
    ANSWER_COMPLETED = "answer.completed"
    ANSWER_CANCELLED = "answer.cancelled"
    ANSWER_ERROR = "answer.error"
    ERROR = "error"

    # client -> server
    SESSION_STOP = "session.stop"
    QUESTION_MANUAL = "question.manual"
    ANSWER_CANCEL = "answer.cancel"
    AUDIO_START = "audio.start"
    AUDIO_STOP = "audio.stop"
    PING = "ping"
    PONG = "pong"


class RejectionReason(StrEnum):
    NOT_A_QUESTION = "not_a_question"
    TOO_SHORT = "too_short"
    LOW_CONFIDENCE = "low_confidence"


class CancelReason(StrEnum):
    SUPERSEDED = "superseded"
    USER_STOP = "user_stop"
    SESSION_ENDED = "session_ended"


class Event(BaseModel):
    """Every frame on the socket. `seq` is monotonic per session and drives
    reconnect replay; `turn_id` lets both sides discard events belonging to a
    question that has already been superseded."""

    type: EventType
    seq: int = 0
    ts: str = Field(default_factory=lambda: utcnow().isoformat())
    turn_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def event(event_type: EventType, turn_id: int | None = None, **data: Any) -> Event:
    return Event(type=event_type, turn_id=turn_id, data=data)


class ClientMessage(BaseModel):
    type: EventType
    turn_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)
