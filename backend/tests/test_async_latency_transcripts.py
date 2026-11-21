import pytest

pytestmark = pytest.mark.asyncio

import threading

from app.realtime.events import EventType
from app.sessions.schemas import TranscriptSource
from tests.test_live_session import Collector, build, session_id, sessions  # noqa: F401
from tests.fakes import FakeLLM


async def test_transcript_write_is_off_the_critical_path_and_ordered(
    sessions, session_id, retriever  # noqa: F811
):
    gate = threading.Event()
    real = sessions.add_transcript

    def slow(entry):
        gate.wait(2)  # a blocked writer must not delay the emit
        return real(entry)

    sessions.add_transcript = slow
    live = build(sessions, session_id, retriever, FakeLLM())
    c = Collector()
    live.subscribe(c)

    await live.on_transcript("first", TranscriptSource.MIC, is_final=True)
    await live.on_transcript("second", TranscriptSource.MIC, is_final=True)
    assert len(c.of(EventType.TRANSCRIPT_FINAL)) == 2  # emitted before any write
    assert sessions.get_transcript(session_id) == []

    gate.set()
    await live.flush_transcripts()
    assert [t.text for t in sessions.get_transcript(session_id)] == ["first", "second"]


async def test_transcript_write_failure_is_logged_not_raised(
    sessions, session_id, retriever, caplog  # noqa: F811
):
    def boom(entry):
        raise RuntimeError("disk")

    sessions.add_transcript = boom
    live = build(sessions, session_id, retriever, FakeLLM())
    await live.on_transcript("hello", TranscriptSource.MIC, is_final=True)
    await live.flush_transcripts()
    assert "transcript_persist_failed" in caplog.text
