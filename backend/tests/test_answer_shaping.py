"""What shape and length of answer a turn asks for, and how a thread evolves.

Two layers meet here and it matters which wins:

* the deterministic classifier's `Category`, from the question's words alone;
* the LLM's `Intent`, from the whole turn, the history and what material was
  provided.

The intent refines the category, except on the relationships whose whole point
is a narrow answer -- "can you optimize it?" reads as CODING but must not be
promoted to the full coding schema, or the model regenerates the approach,
complexity and edge cases the candidate already has.

The progression tests are the interview shape this system exists for:

    "Explain the approach."          -> conceptual, clean
    "Now give me the implementation." -> coding, carries the task
    "What's the complexity?"          -> narrow, carries the code
    "What are the tradeoffs?"         -> narrow
    "How would you optimize it?"      -> narrow

No keyword lists anywhere: every decision below is driven by the classifier's
structured output, which is why the same words can land differently.
"""

import pytest

from app.llm.prompts import (
    BEHAVIORAL_SCHEMA_HINT,
    CODING_SCHEMA_HINT,
    DEBUGGING_SCHEMA_HINT,
    GENERIC_SCHEMA_HINT,
    SQL_SCHEMA_HINT,
    SYSTEM_DESIGN_SCHEMA_HINT,
    build_prompt,
    schema_for,
)
from app.realtime.question_understanding import (
    Intent,
    Relationship,
    Verbosity,
    parse_understanding,
)
from app.schemas.classification import Category
from tests.test_understanding_session import CountingCompleter, harness_with, reply

asyncio_test = pytest.mark.asyncio


async def say(h, text, now):
    from app.sessions.schemas import TranscriptSource

    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    await h.settle()


def prompts(h) -> list[str]:
    return h.llm.prompts


def spoken(text: str) -> str:
    """The turn's words without the terminal punctuation.

    `extract_interview_prompt` always ends its extraction with "?", so a turn
    the interviewer finished with a full stop appears in the prompt with a
    question mark. The wording is preserved exactly; only that final character
    is normalised, so assertions compare the body.
    """
    return text.rstrip(".?!")


# ======================================================== 11-15. answer modes


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (Intent.CODING, CODING_SCHEMA_HINT),
        (Intent.QUERY, SQL_SCHEMA_HINT),
        (Intent.SYSTEM_DESIGN, SYSTEM_DESIGN_SCHEMA_HINT),
        (Intent.TROUBLESHOOTING, DEBUGGING_SCHEMA_HINT),
        (Intent.BEHAVIORAL, BEHAVIORAL_SCHEMA_HINT),
        (Intent.EXPERIENCE, BEHAVIORAL_SCHEMA_HINT),
    ],
)
def test_an_intent_selects_its_answer_shape(intent, expected):
    """Even when the deterministic category disagrees: the intent is the same
    decision made with more evidence."""
    assert schema_for(Category.UNKNOWN, intent, Relationship.NEW_QUESTION) == expected


@pytest.mark.parametrize(
    "intent",
    [
        Intent.CONCEPTUAL,
        Intent.COMPARISON,
        Intent.TRADEOFF,
        Intent.OPTIMIZATION,
        Intent.SCENARIO,
        Intent.CLARIFICATION,
        Intent.OTHER,
    ],
)
def test_a_shapeless_intent_leaves_the_category_in_charge(intent):
    """These read fine as summary/key_points and deliberately have no entry,
    so the deterministic floor still applies."""
    assert schema_for(Category.CODING, intent, Relationship.NEW_QUESTION) == (
        CODING_SCHEMA_HINT
    )
    assert schema_for(Category.UNKNOWN, intent, Relationship.NEW_QUESTION) == (
        GENERIC_SCHEMA_HINT
    )


@pytest.mark.parametrize(
    "relationship",
    [
        Relationship.FOLLOW_UP,
        Relationship.CLARIFICATION,
        Relationship.ACKNOWLEDGEMENT,
        Relationship.DUPLICATE,
    ],
)
def test_a_narrow_follow_up_keeps_the_generic_shape(relationship):
    """The measured behaviour the generic schema exists for: a narrow
    follow-up to a coding question must not be handed the full coding schema,
    or it re-answers the whole problem."""
    assert schema_for(Category.UNKNOWN, Intent.CODING, relationship) == (
        GENERIC_SCHEMA_HINT
    )


def test_a_new_implementation_is_not_narrow():
    """"Now give me the implementation" is the progression step that *should*
    get the full coding shape."""
    for relationship in (
        Relationship.NEW_IMPLEMENTATION,
        Relationship.NEW_METHOD,
        Relationship.CONSTRAINT_CHANGE,
        Relationship.CORRECTION,
        Relationship.CONTINUATION,
    ):
        assert schema_for(Category.UNKNOWN, Intent.CODING, relationship) == (
            CODING_SCHEMA_HINT
        )


def test_no_intent_behaves_exactly_as_before():
    """The deterministic and fallback paths pass no usable intent, so the
    answer shape is whatever it was before this layer existed."""
    for category, expected in [
        (Category.CODING, CODING_SCHEMA_HINT),
        (Category.SQL, SQL_SCHEMA_HINT),
        (Category.UNKNOWN, GENERIC_SCHEMA_HINT),
    ]:
        assert schema_for(category, None, None) == expected
        assert schema_for(category, Intent.OTHER, Relationship.NEW_QUESTION) == expected


# ==================================================== 16. concise vs detailed


def test_the_default_length_adds_no_instruction():
    """Nothing is padded to fill a mode nobody asked for."""
    prompt = build_prompt("What is a hash map?", Category.UNKNOWN, [], [])
    assert "LENGTH:" not in prompt


@pytest.mark.parametrize(
    ("verbosity", "marker"),
    [
        (Verbosity.DIRECT, "direct answer"),
        (Verbosity.DETAILED, "walked through"),
        (Verbosity.STEP_BY_STEP, "step by step"),
        (Verbosity.CODE_FIRST, "see the code"),
    ],
)
def test_a_requested_length_reaches_the_prompt(verbosity, marker):
    prompt = build_prompt(
        "What is a hash map?", Category.UNKNOWN, [], [], verbosity=verbosity
    )
    assert "LENGTH:" in prompt
    assert marker in prompt


def test_the_requested_length_is_read_after_the_schema():
    """Ordering is the mechanism: the interviewer's request is the last
    constraint the model reads, so it wins over what the schema implies."""
    prompt = build_prompt(
        "Write a function to reverse a list.",
        Category.CODING,
        [],
        [],
        verbosity=Verbosity.DIRECT,
    )
    assert prompt.index("JSON shape") < prompt.index("LENGTH:")
    assert prompt.index("LENGTH:") < prompt.index("CURRENT INTERVIEWER QUESTION")


def test_an_unusable_verbosity_degrades_instead_of_failing():
    """Length is the least important field, so a bad value must not cost the
    relationship reading that came with it."""
    understanding = parse_understanding(
        reply(relationship="follow_up", verbosity="extremely_long_please"),
        "Why?",
    )
    assert understanding.verbosity is Verbosity.DEFAULT
    assert understanding.relationship is Relationship.FOLLOW_UP


def test_a_missing_verbosity_is_the_default():
    """An older or terser classifier response omits the field entirely."""
    assert '"verbosity"' not in reply(), "the fake now sends one; revise this test"
    assert parse_understanding(reply(), "What is X?").verbosity is Verbosity.DEFAULT
    # And an explicit null is the same as absent.
    assert (
        parse_understanding(reply(verbosity=None), "What is X?").verbosity
        is Verbosity.DEFAULT
    )


# ============================================ 8/9/10. thread and progression


@asyncio_test
async def test_a_progression_carries_the_task_without_the_whole_interview(
    monkeypatch,
):
    """The five-step sequence this system is for. Each step gets what it needs
    and no more -- the assertion that matters is the *upper* bound, since
    sending everything would pass any "has context" check."""
    completer = CountingCompleter(
        reply(intent="conceptual", relationship="new_question"),
        reply(intent="coding", relationship="new_implementation",
              needs_previous_context=True),
        reply(intent="conceptual", relationship="follow_up",
              needs_previous_code=True),
        reply(intent="tradeoff", relationship="follow_up",
              needs_previous_context=True),
        reply(intent="optimization", relationship="follow_up",
              needs_previous_code=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        steps = [
            "Explain the approach for finding duplicate numbers.",
            "Now give me the implementation.",
            "What's the complexity?",
            "What are the tradeoffs?",
            "How would you optimize it?",
        ]
        for i, text in enumerate(steps):
            await say(h, text, now=100.0 + i * 10)

        assert len(prompts(h)) == 5
        assert completer.calls == 5, "one understanding call per turn, no more"

        # Every step keeps its own exact words.
        for text, prompt in zip(steps, prompts(h)):
            assert spoken(text) in prompt

        # The opening question starts clean.
        assert "Previous Q&A" not in prompts(h)[0]
        # Every later step carries the thread.
        for prompt in prompts(h)[1:]:
            assert "Previous Q&A" in prompt
        # And the window stays bounded rather than growing per turn: two Q&A
        # pairs plus at most the anchor.
        assert prompts(h)[-1].count("\nQ: ") <= 3, "history grew with the thread"
    finally:
        h.dispose()


@asyncio_test
async def test_the_implementation_step_gets_the_coding_shape(monkeypatch):
    completer = CountingCompleter(
        reply(intent="conceptual", relationship="new_question"),
        reply(intent="coding", relationship="new_implementation"),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Explain how you would find duplicates.", now=100.0)
        await say(h, "Now give me the implementation.", now=120.0)

        assert "edge_cases" in prompts(h)[1], "not asked for the coding shape"
        assert spoken("Now give me the implementation.") in prompts(h)[1]
    finally:
        h.dispose()


@asyncio_test
async def test_an_unrelated_question_inherits_nothing(monkeypatch):
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="conceptual", relationship="new_question"),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Write a function to reverse a linked list.", now=100.0)
        await say(h, "What is Azure Databricks?", now=200.0)

        second = prompts(h)[1]
        assert "What is Azure Databricks?" in second
        assert "linked list" not in second, "an unrelated topic leaked forward"
        assert "Previous Q&A" not in second
    finally:
        h.dispose()


@asyncio_test
async def test_a_repeated_question_is_still_answerable(monkeypatch):
    """A duplicate gets one pair of context -- enough to answer differently
    than last time, not enough to re-anchor on the previous wording."""
    completer = CountingCompleter(
        reply(relationship="new_question"),
        reply(relationship="duplicate", needs_previous_answer=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)
        await say(h, "What is a covering index?", now=200.0)

        assert len(prompts(h)) == 2, "the repeat was dropped instead of answered"
        assert "What is a covering index?" in prompts(h)[1]
    finally:
        h.dispose()


# ================================================== 7. correction state


@asyncio_test
async def test_a_correction_puts_the_new_constraint_last(monkeypatch):
    """Both turns survive in the transcript; the answer model reads the
    revised constraint last, which is what makes the latest one win.

    No phrase rules: "Actually, use Kafka instead" is a correction because the
    classifier said so, not because of the word "actually".
    """
    completer = CountingCompleter(
        reply(relationship="new_question", constraints=["Redis"]),
        reply(relationship="correction", constraints=["Kafka"],
              needs_previous_context=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Design a rate limiter using Redis.", now=100.0)
        await say(h, "Actually, use Kafka instead.", now=120.0)

        final = prompts(h)[1]
        # The exact words of both turns are preserved, neither rewritten.
        assert "Actually, use Kafka instead." in final
        assert spoken("Design a rate limiter using Redis.") in final
        # Ordering carries the override: the superseded constraint is
        # background, the correction is the question being answered.
        assert final.index(spoken("Design a rate limiter using Redis.")) < final.index(
            "CURRENT INTERVIEWER QUESTION"
        )
        assert "Kafka" in final[final.index("CURRENT INTERVIEWER QUESTION"):]
    finally:
        h.dispose()


@asyncio_test
async def test_a_numeric_constraint_change_keeps_the_original_task(monkeypatch):
    completer = CountingCompleter(
        reply(relationship="new_question", constraints=["10 million rows"]),
        reply(relationship="constraint_change", constraints=["100 million rows"],
              needs_previous_context=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Design a pipeline. Assume the data is 10 million rows.",
                  now=100.0)
        await say(h, "No, make that 100 million.", now=120.0)

        final = prompts(h)[1]
        assert "No, make that 100 million." in final
        assert "10 million rows" in final, "the original task was dropped"
        # The revised figure is in the question; the old one only as history.
        current = final[final.index("CURRENT INTERVIEWER QUESTION"):]
        assert "100 million" in current
        assert "10 million" not in current
    finally:
        h.dispose()


@asyncio_test
async def test_a_correction_does_not_move_the_task_anchor(monkeypatch):
    """A correction refines the active task rather than opening a new one, so
    the thread it belongs to stays reachable."""
    completer = CountingCompleter(
        reply(relationship="new_question"),
        reply(relationship="correction"),
        reply(relationship="constraint_change"),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Design a rate limiter using Redis.", now=100.0)
        anchor = h.live._thread_anchor
        assert anchor is not None

        await say(h, "Actually, use Kafka instead.", now=120.0)
        assert h.live._thread_anchor == anchor, "a correction opened a new thread"
        await say(h, "And assume 100 million rows.", now=140.0)
        assert h.live._thread_anchor == anchor
    finally:
        h.dispose()
