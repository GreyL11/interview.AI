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


async def test_superseding_ask_does_not_wait_for_cancelled_turn_write(
    sessions, session_id, retriever  # noqa: F811
):
    import asyncio

    from tests.fakes import SlowStreamingLLM

    gate = threading.Event()
    for name in ("mark_turn", "interrupt_turn"):
        real = getattr(sessions, name)

        def slow(*args, _real=real):
            gate.wait(2)  # a blocked status write must not hold the next question
            return _real(*args)

        setattr(sessions, name, slow)

    live = build(sessions, session_id, retriever, SlowStreamingLLM(chunk_delay=0.05))
    c = Collector()
    live.subscribe(c)

    first = await live.ask("Explain Docker")
    await asyncio.sleep(0.01)
    second = await asyncio.wait_for(live.ask("Explain Kubernetes"), timeout=1)
    assert [e.turn_id for e in c.of(EventType.ANSWER_CANCELLED)] == [first]
    assert [e.turn_id for e in c.of(EventType.QUESTION_DETECTED)] == [first, second]

    gate.set()
    await live.cancel()  # flushes the pending status writes
    statuses = {t.turn_id: t.status.value for t in sessions.get_turns(session_id)}
    assert statuses[first] in ("CANCELLED", "INTERRUPTED")
