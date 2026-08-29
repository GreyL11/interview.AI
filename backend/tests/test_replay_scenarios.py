"""Replay-based end-to-end scenarios: realistic interviewer conversations
driven through the real session pipeline via ReplayHarness.

These assert observable behavior (which questions were detected, which answer
won, what was cancelled) rather than internal state, so they stay valid across
refactors of the mechanisms underneath.
"""

import asyncio

import pytest

from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.sessions.schemas import TurnStatus
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayEvent, ReplayHarness

pytestmark = pytest.mark.asyncio


@pytest.fixture
def harness(monkeypatch):
    """Sync fixture: teardown only does sync work (close the sqlite handle,
    remove the temp file), so it needs no running event loop. Tests that
    specifically exercise `LiveSession.close()` call it themselves."""
    h = ReplayHarness(monkeypatch=monkeypatch)
    yield h
    h.dispose()


async def test_simple_question(harness):
    result = await harness.play([ReplayEvent(at_ms=0, text="Explain caching.")])

    assert result.detected_questions() == ["Explain caching?"]
    assert len(result.of(EventType.ANSWER_COMPLETED)) == 1
    detected = result.of(EventType.QUESTION_DETECTED)[0]
    assert detected.data["classification"]["category"] == "TECHNICAL_KNOWLEDGE"


async def test_correction_only_the_corrected_question_wins(monkeypatch):
    """The first answer must be cancelled and must not reach a final state."""
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="Explain caching."),
            ReplayEvent(at_ms=300, text="No, explain Redis caching."),
        ], settle_between=False)

        completed = result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1
        answered = [t for t in h.sessions.get_turns(h.session_id)
                    if t.status == TurnStatus.ANSWERED]
        assert len(answered) == 1
        assert "Redis" in answered[0].question
        # The superseded turn is recorded as cancelled, not silently dropped.
        assert result.cancelled_turn_ids()
        # No completed answer may belong to a cancelled turn.
        assert set(result.completed_turn_ids()).isdisjoint(result.cancelled_turn_ids())
    finally:
        h.dispose()


async def test_followup_uses_previous_conversation(monkeypatch):
    h = ReplayHarness(monkeypatch=monkeypatch, memory_factory=SqliteSessionMemory)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="Explain Redis caching."),
            ReplayEvent(at_ms=4000, text="What happens if Redis goes down?"),
            ReplayEvent(at_ms=8000, text="Why?"),
        ])

        assert len(result.detected_questions()) == 3
        # The last prompt carries earlier Q&A as context, so "Why?" is answerable.
        assert "Redis" in h.llm.prompts[-1]
    finally:
        h.dispose()


async def test_setup_context_reaches_llm_but_not_display(harness):
    """The setup-context buffer applies to utterances the detector *rejects*
    ("...just write a character count program" reads as narration, not a
    prompt); the next accepted question inherits them as effective context
    while the displayed question stays clean."""
    result = await harness.play([
        ReplayEvent(at_ms=0, text="By using this study, just write a character count program."),
        ReplayEvent(at_ms=1500, text="How many times is each character repeated?"),
    ])

    questions = result.detected_questions()
    assert questions == ["How many times is each character repeated?"]
    assert "character count program" in harness.llm.prompts[-1]


async def test_an_accepted_setup_sentence_is_not_carried_into_the_refinement(harness):
    """Documents a real current limitation, so a future fix has a test.

    Without "just", the same sentence parses as a complete imperative task and
    is *accepted* as its own question. The refinement that follows then merges
    and, because prompt extraction keeps only the last matching sentence, the
    original setup drops out of what the LLM finally sees. Distinguishing this
    from a genuine correction ("Write X." / "No, actually write Y.") needs a
    signal the detector does not currently have."""
    result = await harness.play([
        ReplayEvent(at_ms=0, text="By using this data, write a character count program."),
        ReplayEvent(at_ms=1500, text="How many times is each character repeated?"),
    ])

    assert result.detected_questions()[-1] == "How many times is each character repeated?"
    assert "character count program" not in harness.llm.prompts[-1]


async def test_pause_split_coding_question_merges(monkeypatch):
    """The continuation lands 2.5s later -- realistically while the fragment's
    answer is still streaming, so the fragment's answer is superseded and only
    the merged coding question produces a final answer."""
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="Given an array, find two numbers"),
            ReplayEvent(at_ms=2500, text="whose sum equals a target value."),
        ], settle_between=False)

        prompt = h.llm.prompts[-1]
        assert "find two numbers" in prompt
        assert "sum equals a target value" in prompt
        assert len(result.of(EventType.ANSWER_COMPLETED)) == 1
    finally:
        h.dispose()


async def test_rapid_fire_only_the_latest_wins(monkeypatch):
    h = ReplayHarness(llm=SlowStreamingLLM(chunk_delay=0.02), monkeypatch=monkeypatch)
    try:
        result = await h.play([
            ReplayEvent(at_ms=0, text="What is a hash map?"),
            ReplayEvent(at_ms=5000, text="How would you design a URL shortener?"),
            ReplayEvent(at_ms=10000, text="What is a database index?"),
        ], settle_between=False)

        assert len(result.of(EventType.ANSWER_COMPLETED)) == 1
        answered = [t for t in h.sessions.get_turns(h.session_id)
                    if t.status == TurnStatus.ANSWERED]
        assert len(answered) == 1
        assert "database index" in answered[0].question
    finally:
        h.dispose()


async def test_acknowledgement_does_not_trigger_an_answer(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="Explain caching."),
        ReplayEvent(at_ms=3000, text="Okay."),
    ])

    assert len(result.detected_questions()) == 1
    assert result.rejected_texts()


async def test_unrelated_statement_triggers_nothing(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="I worked with Redis last year.")
    ])

    assert result.detected_questions() == []
    assert result.of(EventType.ANSWER_COMPLETED) == []


async def test_topic_switch_does_not_contaminate(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="Explain caching."),
        ReplayEvent(at_ms=300, text="Actually, let's do longest palindromic substring."),
    ])

    questions = result.detected_questions()
    assert "palindromic" in questions[-1]
    assert "caching" not in questions[-1].lower()


async def test_short_followup_without_context_is_rejected(harness):
    result = await harness.play([ReplayEvent(at_ms=0, text="Why?")])

    assert result.detected_questions() == []
    assert result.rejected_texts() == ["Why?"]


async def test_mic_speech_never_produces_a_question(harness):
    result = await harness.play([
        ReplayEvent(at_ms=0, text="I think we could use a dictionary here.", source="MIC"),
    ])

    assert result.detected_questions() == []
    assert result.of(EventType.ANSWER_COMPLETED) == []
    # It is still recorded as a transcript -- isolation, not silence.
    assert result.of(EventType.TRANSCRIPT_FINAL)


# ---------------------------------------------- domain follow-up conversations


async def test_sql_question_then_method_followup(monkeypatch):
    """"Can you do it using a window function?" must resolve against the
    previous SQL problem, and both turns must stay SQL-aware."""
    harness = ReplayHarness(monkeypatch=monkeypatch, memory_factory=SqliteSessionMemory)
    try:
        result = await harness.play([
            ReplayEvent(at_ms=0, text="Find the second highest salary."),
            ReplayEvent(at_ms=6000, text="Can you do it using a window function?"),
            ReplayEvent(at_ms=12000, text="What's the performance tradeoff?"),
        ])

        categories = [
            e.data["classification"]["category"]
            for e in result.of(EventType.QUESTION_DETECTED)
        ]
        assert categories[0] == "SQL"
        assert categories[1] == "SQL"  # explicit method request stays SQL
        assert len(result.completed_turn_ids()) == 3

        # The narrow performance follow-up still sees the original problem.
        assert "second highest salary" in harness.llm.prompts[-1]
    finally:
        harness.dispose()


async def test_behavioral_question_then_learning_followup(monkeypatch):
    """The follow-up must be answerable from the previous STAR story, and must
    not itself be BEHAVIORAL -- that would re-trigger a whole new story."""
    harness = ReplayHarness(monkeypatch=monkeypatch, memory_factory=SqliteSessionMemory)
    try:
        result = await harness.play([
            ReplayEvent(at_ms=0, text="Tell me about a challenging project."),
            ReplayEvent(at_ms=6000, text="What did you learn from that?"),
        ])

        categories = [
            e.data["classification"]["category"]
            for e in result.of(EventType.QUESTION_DETECTED)
        ]
        assert categories[0] == "BEHAVIORAL"
        assert categories[1] != "BEHAVIORAL"
        assert "challenging project" in harness.llm.prompts[-1]
    finally:
        harness.dispose()


async def test_coding_question_then_complexity_followup(monkeypatch):
    harness = ReplayHarness(monkeypatch=monkeypatch, memory_factory=SqliteSessionMemory)
    try:
        result = await harness.play([
            ReplayEvent(at_ms=0, text="Find two numbers in an array whose sum equals a target."),
            ReplayEvent(at_ms=6000, text="What is the time complexity?"),
        ])

        detected = result.of(EventType.QUESTION_DETECTED)
        assert detected[0].data["classification"]["category"] == "CODING"
        # Narrow follow-up: generic schema, so no instruction to regenerate the
        # full approach/code/complexity/edge-case block.
        from app.llm.prompts import CODING_SCHEMA_HINT
        assert CODING_SCHEMA_HINT in harness.llm.prompts[0]
        assert CODING_SCHEMA_HINT not in harness.llm.prompts[-1]
        assert "sum equals a target" in harness.llm.prompts[-1]
    finally:
        harness.dispose()


# ------------------------------------------------------ interruption outcomes


async def test_interruption_after_partial_content_is_preserved(monkeypatch):
    """Superseded once useful text existed: the partial is kept, the turn is
    INTERRUPTED, and it must not reach conversation memory."""
    harness = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0.02),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    try:
        # Drive it directly rather than via play(): the supersede has to land
        # *after* real streamed text exists, and that is a wall-clock race the
        # scripted `at_ms` timeline (detector time only) cannot express.
        from app.sessions.schemas import TranscriptSource

        await harness.live.on_transcript(
            "Design a URL shortener.", TranscriptSource.LOOPBACK, True, now=0.0
        )
        for _ in range(200):
            await asyncio.sleep(0.01)
            if harness.result.of(EventType.ANSWER_DELTA):
                break
        assert harness.result.of(EventType.ANSWER_DELTA), "no visible text streamed"

        await harness.live.on_transcript(
            "What is caching?", TranscriptSource.LOOPBACK, True, now=5.0
        )
        await harness.settle()
        result = harness.result

        cancelled = result.of(EventType.ANSWER_CANCELLED)
        assert len(cancelled) == 1
        assert cancelled[0].data["interrupted"] is True
        assert (cancelled[0].data["partial_summary"] or "").strip() != ""

        turns = {t.status for t in harness.sessions.get_turns(harness.session_id)}
        assert TurnStatus.INTERRUPTED in turns
        # Only ANSWERED turns are readable as history, so the truncated answer
        # cannot influence anything that follows.
        answered = harness.sessions.get_answered_turns(harness.session_id)
        assert all("URL shortener" not in t.question for t in answered)
    finally:
        harness.dispose()


async def test_interruption_before_any_content_leaves_no_answer(monkeypatch):
    """Superseded before a single visible token: CANCELLED, no partial, and no
    junk answer in history."""
    harness = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=5.0),  # never reaches a first delta
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    try:
        result = await harness.play(
            [
                ReplayEvent(at_ms=0, text="Design a URL shortener."),
                ReplayEvent(at_ms=100, text="What is caching?"),
            ],
            settle_between=False,
        )

        cancelled = result.of(EventType.ANSWER_CANCELLED)
        assert len(cancelled) == 1
        assert cancelled[0].data["interrupted"] is False
        assert not (cancelled[0].data["partial_summary"] or "")

        statuses = {t.status for t in harness.sessions.get_turns(harness.session_id)}
        assert TurnStatus.CANCELLED in statuses
        assert TurnStatus.INTERRUPTED not in statuses
    finally:
        harness.dispose()
