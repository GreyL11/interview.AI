import uuid

import pytest

from app.documents.schemas import utcnow
from app.schemas.answer import Answer
from app.sessions.schemas import (
    RetrievalHit,
    Session,
    SessionStatus,
    TranscriptEntry,
    TranscriptSource,
    Turn,
    TurnStatus,
)


@pytest.fixture
def sessions(database):
    from app.storage.session_repository import SessionRepository

    return SessionRepository(database)


def new_session(sessions, title="Practice") -> Session:
    return sessions.create(
        Session(session_id=str(uuid.uuid4()), started_at=utcnow(), title=title)
    )


def answer(summary="Make it idempotent.") -> Answer:
    return Answer(summary=summary, key_points=["a", "b"], detailed_answer="detail")


def test_create_and_get(sessions):
    created = new_session(sessions)
    fetched = sessions.get(created.session_id)
    assert fetched.session_id == created.session_id
    assert fetched.status == SessionStatus.ACTIVE


def test_get_missing_returns_none(sessions):
    assert sessions.get("nope") is None


def test_end_session(sessions):
    s = new_session(sessions)
    sessions.end(s.session_id)
    ended = sessions.get(s.session_id)
    assert ended.status == SessionStatus.ENDED
    assert ended.ended_at is not None


def test_list_includes_turn_counts(sessions):
    s = new_session(sessions)
    for i in range(3):
        sessions.create_turn(Turn(session_id=s.session_id, seq=i, question=f"q{i}"))

    listed = sessions.list()
    assert len(listed) == 1
    assert listed[0].turn_count == 3


def test_next_seq_increments(sessions):
    s = new_session(sessions)
    assert sessions.next_seq(s.session_id) == 0
    sessions.create_turn(Turn(session_id=s.session_id, seq=0, question="q"))
    assert sessions.next_seq(s.session_id) == 1


def test_complete_turn_persists_answer_and_hits(sessions):
    s = new_session(sessions)
    turn = sessions.create_turn(Turn(session_id=s.session_id, seq=0, question="How to dedupe?"))

    sessions.complete_turn(
        turn.turn_id,
        answer=answer(),
        context_found=True,
        latency_ms=1234,
        hits=[RetrievalHit(chunk_id="c1", score=0.8, rank=0)],
    )

    stored = sessions.get_turns(s.session_id)[0]
    assert stored.status == TurnStatus.ANSWERED
    assert stored.answer.summary == "Make it idempotent."
    assert stored.context_found is True
    assert stored.latency_ms == 1234
    assert sessions.get_hits(turn.turn_id)[0].chunk_id == "c1"


def test_answered_turns_excludes_cancelled(sessions):
    s = new_session(sessions)
    a = sessions.create_turn(Turn(session_id=s.session_id, seq=0, question="kept"))
    b = sessions.create_turn(Turn(session_id=s.session_id, seq=1, question="dropped"))

    sessions.complete_turn(a.turn_id, answer(), False, 10)
    sessions.mark_turn(b.turn_id, TurnStatus.CANCELLED)

    answered = sessions.get_answered_turns(s.session_id)
    assert [t.question for t in answered] == ["kept"]


def test_transcript_roundtrip(sessions):
    s = new_session(sessions)
    sessions.add_transcript(TranscriptEntry(
        session_id=s.session_id, source=TranscriptSource.LOOPBACK, is_final=False, text="partial"
    ))
    sessions.add_transcript(TranscriptEntry(
        session_id=s.session_id, source=TranscriptSource.LOOPBACK, is_final=True, text="final text"
    ))

    finals = sessions.get_transcript(s.session_id)
    assert [t.text for t in finals] == ["final text"]

    everything = sessions.get_transcript(s.session_id, finals_only=False)
    assert len(everything) == 2


def test_summary_upsert_overwrites(sessions):
    from app.sessions.schemas import SessionSummary

    s = new_session(sessions)
    sessions.upsert_summary(SessionSummary(
        session_id=s.session_id, summary="first", topics=["kafka"], covered_through_seq=2
    ))
    sessions.upsert_summary(SessionSummary(
        session_id=s.session_id, summary="second", topics=["kafka", "sql"], covered_through_seq=5
    ))

    stored = sessions.get_summary(s.session_id)
    assert stored.summary == "second"
    assert stored.covered_through_seq == 5
    assert stored.topics == ["kafka", "sql"]


def test_delete_cascades(sessions):
    s = new_session(sessions)
    turn = sessions.create_turn(Turn(session_id=s.session_id, seq=0, question="q"))
    sessions.complete_turn(turn.turn_id, answer(), False, 5,
                           hits=[RetrievalHit(chunk_id="c", score=0.5, rank=0)])
    sessions.add_transcript(TranscriptEntry(
        session_id=s.session_id, source=TranscriptSource.MIC, is_final=True, text="hi"
    ))

    assert sessions.delete(s.session_id) is True
    assert sessions.get_turns(s.session_id) == []
    assert sessions.get_transcript(s.session_id) == []
    assert sessions.get_hits(turn.turn_id) == []


def test_close_stale_active(sessions):
    a = new_session(sessions)
    b = new_session(sessions)
    sessions.end(b.session_id)

    assert sessions.close_stale_active() == 1
    assert sessions.get(a.session_id).status == SessionStatus.ENDED


def test_detail_bundles_everything(sessions):
    s = new_session(sessions)
    turn = sessions.create_turn(Turn(session_id=s.session_id, seq=0, question="q"))
    sessions.complete_turn(turn.turn_id, answer(), True, 20)
    sessions.add_transcript(TranscriptEntry(
        session_id=s.session_id, source=TranscriptSource.MIC, is_final=True, text="spoken"
    ))

    detail = sessions.detail(s.session_id)
    assert detail.session.session_id == s.session_id
    assert len(detail.turns) == 1
    assert len(detail.transcript) == 1


def test_detail_of_missing_session_is_none(sessions):
    assert sessions.detail("nope") is None
