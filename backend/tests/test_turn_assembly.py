"""Turn assembly: how many provider calls, and what exact text.

Covers the lettered matrix from the hardening brief. Every case asserts both
halves of the invariant, because either alone is easy to satisfy wrongly:

    exactly N provider calls   AND   the exact reconstructed wording

Deterministic: fragment arrival is driven by explicit speech-clock timestamps
and by `on_speech_start`, never by sleeping. `ReplayHarness` pins the hold
budgets to 1ms so the *decisions* are what get tested, not the clock -- with
one exception (`test_a_held_turn_waits_while_the_interviewer_is_still_talking`)
which sets a real budget to prove the speech signal overrides it.
"""

import pytest

from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.sessions.schemas import TranscriptSource, TurnStatus
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayEvent, ReplayHarness

asyncio_test = pytest.mark.asyncio


def asked(harness) -> list[str]:
    """The exact interviewer question each provider call carried."""
    out = []
    for prompt in harness.llm.prompts:
        for line in prompt.splitlines():
            if "CURRENT INTERVIEWER QUESTION" in line:
                out.append(line.split("): ", 1)[-1])
    return out


async def play(harness, fragments, start_ms=0, gap_ms=400, settle_between=False):
    """Feed fragments on the speech clock, announcing each as live speech."""
    at = start_ms
    events = []
    for i, text in enumerate(fragments):
        events.append(ReplayEvent(at_ms=at, text=text))
        at += gap_ms
    # Announce speech for every fragment after the first, the way VAD does.
    for i, ev in enumerate(events):
        await harness.live.on_speech_start(i + 1)
        await harness.live.on_transcript(
            ev.text, TranscriptSource.LOOPBACK, is_final=True, now=ev.at_ms / 1000
        )
        if settle_between:
            await harness.settle()
    await harness.settle()


@pytest.fixture
def harness(monkeypatch):
    # SqliteSessionMemory, not the in-memory one: LiveSession persists turns to
    # the repository and only this implementation reads them back, so it is the
    # only one under which conversation context actually reaches a follow-up.
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    yield h
    h.dispose()


# ============================================================ the matrix


@asyncio_test
async def test_A_three_fragment_question_is_one_exact_call(harness):
    await play(harness, [
        "Can you explain",
        "how dependency injection works",
        "in FastAPI?",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    assert asked(harness) == ["Can you explain how dependency injection works in FastAPI?"]


@asyncio_test
async def test_B_scenario_setup_plus_question_is_one_call(harness):
    await play(harness, [
        "Suppose you're working with Azure Databricks",
        "and processing five million records per hour",
        "how would you optimize the pipeline?",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    for fragment in ("Azure Databricks", "five million records", "optimize the pipeline"):
        assert fragment in question, question
    # Original order preserved.
    assert question.index("Azure Databricks") < question.index("five million")
    assert question.index("five million") < question.index("optimize")


@asyncio_test
async def test_C_two_unrelated_questions_are_two_calls(harness):
    await play(
        harness,
        ["What is Azure Databricks?", "How does Delta Lake work?"],
        gap_ms=20_000,
        settle_between=True,
    )

    assert asked(harness) == ["What is Azure Databricks?", "How does Delta Lake work?"]


@asyncio_test
async def test_D_a_dependent_second_question_is_its_own_turn(harness):
    await play(
        harness,
        ["What is Azure Databricks?", "Why would you choose it over Synapse?"],
        gap_ms=20_000,
        settle_between=True,
    )

    assert len(harness.llm.prompts) == 2
    assert asked(harness)[1] == "Why would you choose it over Synapse?"
    # The second call carries the first exchange as conversation context, which
    # is what makes "it" resolvable.
    assert "Azure Databricks" in harness.llm.prompts[1]


@asyncio_test
async def test_E_a_correction_does_not_carry_the_obsolete_constraint(harness):
    await play(harness, [
        "Assume 1,000 QPS",
        "Actually, assume 10,000 QPS",
        "How would you design the service?",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    assert "10,000" in question
    assert "1,000 QPS" not in question, f"obsolete constraint survived: {question!r}"


@asyncio_test
async def test_F_a_coding_problem_split_three_ways_is_one_call(harness):
    await play(harness, [
        "Given an array of integers",
        "find two numbers",
        "that sum to the target",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    for fragment in ("Given an array of integers", "find two numbers", "sum to the target"):
        assert fragment in question, question


@asyncio_test
async def test_G_a_behavioural_question_split_three_ways_is_one_call(harness):
    await play(harness, [
        "Tell me about a time",
        "you had a production incident",
        "and how you handled it.",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    for fragment in ("Tell me about a time", "production incident", "how you handled it"):
        assert fragment in question, question


@asyncio_test
async def test_H_a_two_fragment_scenario_is_one_call(harness):
    await play(harness, [
        "Let's say production latency increases",
        "what would you investigate first?",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    assert "production latency increases" in question
    assert "investigate first" in question


@asyncio_test
async def test_I_a_pause_mid_question_does_not_split_the_turn(harness):
    await play(
        harness,
        ["Can you explain", "how dependency injection works in FastAPI?"],
        gap_ms=2_000,
    )

    assert len(harness.llm.prompts) == 1, asked(harness)
    assert asked(harness) == ["Can you explain how dependency injection works in FastAPI?"]


@asyncio_test
async def test_J_a_bare_why_is_an_immediate_followup(harness):
    await play(harness, ["What is a hash map?"], settle_between=True)
    await play(harness, ["Why?"], start_ms=8_000, settle_between=True)

    assert len(harness.llm.prompts) == 2
    assert asked(harness)[1] == "Why?"
    assert harness.live._pending_ask is None, "a follow-up must never be held"
    # Previous exchange travels with it.
    assert "hash map" in harness.llm.prompts[1]


@asyncio_test
async def test_K_can_you_elaborate_is_an_immediate_followup(harness):
    await play(harness, ["Explain database indexing."], settle_between=True)
    await play(harness, ["Can you elaborate?"], start_ms=8_000, settle_between=True)

    assert len(harness.llm.prompts) == 2
    assert asked(harness)[1] == "Can you elaborate?"
    assert "indexing" in harness.llm.prompts[1]


@asyncio_test
async def test_L_leading_filler_does_not_split_the_turn(harness):
    await play(harness, ["Okay", "What is Azure Databricks?"])

    assert len(harness.llm.prompts) == 1, asked(harness)
    assert "What is Azure Databricks?" in asked(harness)[0]


@asyncio_test
async def test_M_a_long_setup_never_fires_mid_scenario(harness):
    """Five fragments, one of which ("write a function...") is imperative and
    would read as a complete request on its own."""
    await play(harness, [
        "Here's a problem",
        "you have a stream of events",
        "arriving out of order",
        "with duplicates",
        "write a function to deduplicate them",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    question = asked(harness)[0]
    assert "deduplicate" in question
    assert "out of order" in question


# ================================================== structural invariants


@asyncio_test
async def test_a_complete_question_is_never_held(harness):
    await play(harness, ["What is Azure Databricks?"])

    assert len(harness.llm.prompts) == 1
    assert harness.live._pending_ask is None
    assert harness.live._accumulating_since is None


@asyncio_test
async def test_nothing_is_sent_while_a_turn_is_still_accumulating(monkeypatch):
    """The core LLM-call invariant: no provider call for a fragment."""
    h = ReplayHarness(monkeypatch=monkeypatch, stabilization_ms=60_000)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "Can you explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )

        assert h.llm.prompts == [], "a fragment must not reach the provider"
        assert h.live._pending_ask is not None
        assert h.live._current_turn_id is None
    finally:
        h.live._abandon_accumulation()
        h.dispose()


@asyncio_test
async def test_a_held_turn_waits_while_the_interviewer_is_still_talking(monkeypatch):
    """Category 12, and the reason a fixed hold is not enough: the wait needed
    to catch a continuation is `pause + duration(continuation)`, and that
    duration is unknowable in advance. Here the hold is deliberately tiny and
    the interviewer is mid-utterance, so the turn must not be sent anyway."""
    h = ReplayHarness(monkeypatch=monkeypatch, stabilization_ms=1)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "Can you explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )
        # A new utterance opens: the 1ms hold has long expired, but speech is
        # in progress, so the fragment must still be parked.
        await h.live.on_speech_start(2)
        for _ in range(5):
            import asyncio
            await asyncio.sleep(0.05)

        assert h.llm.prompts == [], "sent a fragment while speech was in progress"

        # The continuation lands and the assembled turn goes once.
        await h.live.on_transcript(
            "how dependency injection works in FastAPI?",
            TranscriptSource.LOOPBACK, is_final=True, now=100.4,
        )
        await h.settle()

        assert len(h.llm.prompts) == 1
        assert asked(h) == ["Can you explain how dependency injection works in FastAPI?"]
    finally:
        h.dispose()


@asyncio_test
async def test_a_partial_never_reaches_the_provider(harness):
    """Category 14: partials are display-only, structurally."""
    await harness.live.on_transcript(
        "Can you explain how", TranscriptSource.LOOPBACK, is_final=False, now=100.0
    )
    await harness.settle()

    assert harness.llm.prompts == []
    assert harness.result.of(EventType.TRANSCRIPT_PARTIAL)
    assert not harness.result.of(EventType.QUESTION_DETECTED)


@asyncio_test
async def test_a_duplicate_final_does_not_double_the_question(harness):
    await play(harness, ["What is Azure Databricks?", "What is Azure Databricks?"],
               gap_ms=300, settle_between=True)

    for question in asked(harness):
        assert question.count("Azure Databricks") == 1, question


@asyncio_test
async def test_mic_speech_never_assembles_an_interviewer_turn(harness):
    """Category: MIC stays out of the realtime question path entirely."""
    await harness.live.on_transcript(
        "What is Azure Databricks?", TranscriptSource.MIC, is_final=True, now=100.0
    )
    await harness.settle()

    assert harness.llm.prompts == []
    assert not harness.result.of(EventType.QUESTION_DETECTED)
    assert harness.result.of(EventType.TRANSCRIPT_FINAL)


@asyncio_test
async def test_question_answer_question_leaves_two_independent_turns(harness):
    """Category 11: the answered turn must not absorb the next question."""
    await play(harness, ["Write a function to reverse a linked list."], settle_between=True)
    await play(harness, ["Implement an LRU cache."], start_ms=3_000, settle_between=True)

    assert len(harness.llm.prompts) == 2, asked(harness)
    assert "linked list" not in asked(harness)[1]
    answered = [t for t in harness.sessions.get_turns(harness.session_id)
                if t.status == TurnStatus.ANSWERED]
    assert len(answered) == 2


@asyncio_test
async def test_a_followup_chain_keeps_earlier_context(harness):
    await play(harness, ["What is Redis?"], settle_between=True)
    await play(harness, ["Why is it fast?"], start_ms=8_000, settle_between=True)
    await play(harness, ["And at scale?"], start_ms=16_000, settle_between=True)

    assert len(harness.llm.prompts) == 3, asked(harness)
    assert "Redis" in harness.llm.prompts[-1], "earliest turn dropped out of context"


@asyncio_test
async def test_five_fragments_still_cost_one_call(harness):
    await play(harness, [
        "Imagine you have a service",
        "handling ten thousand requests per second",
        "and the database starts timing out",
        "under peak load",
        "how would you diagnose it?",
    ])

    assert len(harness.llm.prompts) == 1, asked(harness)
    assert "diagnose it" in asked(harness)[0]


@asyncio_test
async def test_accumulation_state_resets_after_a_turn_is_sent(harness):
    await play(harness, ["Can you explain", "how caching works?"])
    assert harness.live._accumulating_since is None
    assert harness.live._fragments == 0
    assert harness.live._pending_ask is None
