"""Context selection: which prior evidence each relationship pulls in.

Pure functions over what `SessionMemory.bounded_context()` already returns --
no new memory system, and nothing here reaches further back than the memory
layer's own token-bounded window.

The two failure modes being guarded against pull in opposite directions:
sending the whole conversation every turn (context pollution), and sending
none of it when the turn genuinely depends on it (a follow-up that forgot the
question it follows).
"""

import pytest

from app.realtime.question_understanding import (
    ContextPlan,
    Relationship,
    Understanding,
    UnderstandingSource,
    plan_for,
    select_context,
)

SUMMARY = "[Earlier in this session] Discussed Kafka partitioning."


def window(*questions: str) -> list[str]:
    """A bounded_context-shaped window: alternating Q/A lines, oldest first."""
    out: list[str] = []
    for q in questions:
        out.append(f"Q: {q}")
        out.append(f"A: answer to {q}")
    return out


def understanding(
    relationship: Relationship,
    source: UnderstandingSource = UnderstandingSource.LLM,
    **flags,
) -> Understanding:
    return Understanding(
        exact_question="current question",
        relationship=relationship,
        source=source,
        **flags,
    )


# ------------------------------------------------------- no inheritance


@pytest.mark.parametrize(
    "relationship", [Relationship.NEW_QUESTION, Relationship.OTHER]
)
def test_a_new_question_inherits_nothing(relationship):
    """H3: a fresh subject after a long unrelated conversation starts clean."""
    history = [SUMMARY] + window(
        "What is Kafka?", "How does partitioning work?",
        "Explain Spark Structured Streaming.", "What about checkpointing?",
    )
    selected = select_context(history, understanding(relationship))

    assert selected == []


def test_a_new_question_after_a_long_conversation_is_not_polluted():
    history = [SUMMARY] + window(*[f"kafka question {i}" for i in range(15)])
    selected = select_context(
        history, understanding(Relationship.NEW_QUESTION)
    )
    assert selected == []
    assert not any("kafka" in line for line in selected)


# ---------------------------------------------------- immediate context


@pytest.mark.parametrize(
    "relationship",
    [Relationship.FOLLOW_UP, Relationship.CLARIFICATION, Relationship.CONTINUATION],
)
def test_conversational_turns_get_the_immediate_context(relationship):
    history = window("first", "second", "third", "fourth")
    selected = select_context(history, understanding(relationship))

    # The most recent pairs, and not the whole conversation.
    assert "Q: fourth" in selected
    assert "Q: third" in selected
    assert "Q: first" not in selected, "sent the entire conversation"


def test_a_follow_up_chain_keeps_seeing_the_preceding_answer():
    """H3 second example: a design thread's follow-ups keep the design."""
    history = window(
        "Design a streaming pipeline.", "What if throughput doubles?",
        "How would you handle failures?",
    )
    selected = select_context(history, understanding(Relationship.FOLLOW_UP))

    assert "A: answer to How would you handle failures?" in selected


def test_a_duplicate_gets_context_but_less_of_it():
    """A re-explanation should differ from the first answer without
    re-anchoring on the whole thread."""
    history = window("first", "second", "third")
    plan = plan_for(understanding(Relationship.DUPLICATE))
    selected = select_context(history, understanding(Relationship.DUPLICATE))

    assert plan.recent_pairs == 1
    assert "Q: third" in selected
    assert "Q: second" not in selected
    assert selected != [], "a duplicate must still be answerable with context"


# ------------------------------------------------- the task-thread anchor


@pytest.mark.parametrize(
    "relationship",
    [
        Relationship.NEW_METHOD,
        Relationship.NEW_IMPLEMENTATION,
        Relationship.CONSTRAINT_CHANGE,
        Relationship.CORRECTION,
    ],
)
def test_task_continuations_keep_the_underlying_problem(relationship):
    """H4: five turns later, "now do it in Java" still needs the original
    problem -- which a recent-window-only selection would have dropped."""
    anchor = "Write a Python function to find duplicate numbers."
    history = window(
        anchor,
        "Show me another way using a set.",
        "Can you do it without extra space?",
        "What's the complexity?",
    )

    selected = select_context(history, understanding(relationship), anchor)

    assert f"Q: {anchor}" in selected, "lost the underlying task"
    # And the recent refinements are still there.
    assert "Q: What's the complexity?" in selected
    # The original task reads first, before the refinements to it.
    assert selected.index(f"Q: {anchor}") < selected.index("Q: What's the complexity?")


def test_the_anchor_is_not_duplicated_when_it_is_already_recent():
    anchor = "Write a function to find duplicates."
    history = window(anchor, "Show me another way.")

    selected = select_context(
        history, understanding(Relationship.NEW_METHOD), anchor
    )

    assert selected.count(f"Q: {anchor}") == 1


def test_a_missing_anchor_is_simply_absent_not_an_error():
    """It may have fallen out of the memory layer's window entirely, in which
    case the compressed summary is what carries it."""
    history = [SUMMARY] + window("recent one")

    selected = select_context(
        history, understanding(Relationship.NEW_METHOD), "a question long gone"
    )

    assert SUMMARY in selected
    assert "Q: recent one" in selected


def test_a_follow_up_does_not_drag_in_the_anchor():
    """Only task continuations need the original problem. "Why?" needs what
    was just said, which is cheaper."""
    anchor = "Write a function to find duplicates."
    history = window(anchor, "a", "b", "c")

    selected = select_context(
        history, understanding(Relationship.FOLLOW_UP), anchor
    )

    assert f"Q: {anchor}" not in selected


# ------------------------------------------- classifier flags widen only


def test_explicit_flags_can_widen_a_plan():
    plan = plan_for(
        understanding(Relationship.NEW_QUESTION, needs_previous_context=True)
    )
    assert plan.recent_pairs > 0, "the classifier's own signal was ignored"
    assert plan.include_summary


def test_needs_previous_code_reaches_for_the_anchor():
    plan = plan_for(
        understanding(Relationship.FOLLOW_UP, needs_previous_code=True)
    )
    assert plan.include_anchor, "code lives in the task that introduced it"


def test_flags_never_shrink_a_plan():
    """A model that under-reports must not be able to strip context the
    relationship says is required."""
    quiet = plan_for(
        understanding(
            Relationship.NEW_METHOD,
            needs_previous_context=False,
            needs_previous_answer=False,
            needs_previous_code=False,
        )
    )
    assert quiet.recent_pairs > 0
    assert quiet.include_anchor


# ------------------------------------------------- non-LLM sources


@pytest.mark.parametrize(
    "source", [UnderstandingSource.FALLBACK, UnderstandingSource.DETERMINISTIC]
)
def test_without_a_real_classification_nothing_is_narrowed(source):
    """Narrowing is a capability the classifier provides. On paths where it
    was not consulted the window is passed through untouched, which is what
    keeps the disabled/failed behaviour identical to before this layer."""
    history = [SUMMARY] + window("one", "two", "three")

    selected = select_context(
        history, understanding(Relationship.NEW_QUESTION, source=source)
    )

    assert selected == history
    assert plan_for(
        understanding(Relationship.NEW_QUESTION, source=source)
    ).everything


# --------------------------------------------------------- mechanics


def test_an_empty_window_selects_nothing():
    for relationship in Relationship:
        assert select_context([], understanding(relationship)) == []


def test_selection_preserves_oldest_first_ordering():
    """What makes a correction work: the revised constraint is simply the
    later pair, so the answer model reads it last."""
    history = window("assume 1,000 QPS", "actually assume 10,000 QPS")

    selected = select_context(history, understanding(Relationship.CORRECTION))

    assert selected.index("Q: assume 1,000 QPS") < selected.index(
        "Q: actually assume 10,000 QPS"
    )


def test_pairs_are_never_split():
    """A question without its answer is worse than neither."""
    history = window("one", "two", "three")
    selected = select_context(history, understanding(Relationship.FOLLOW_UP))

    assert len(selected) % 2 == 0
    for i in range(0, len(selected), 2):
        assert selected[i].startswith("Q: ")
        assert selected[i + 1].startswith("A: ")


def test_selection_never_widens_beyond_the_given_window():
    history = window("only one")
    for relationship in Relationship:
        selected = select_context(history, understanding(relationship), "anything")
        assert set(selected).issubset(set(history))


def test_the_summary_only_travels_when_the_plan_asks_for_it():
    history = [SUMMARY] + window("one", "two")

    with_summary = select_context(history, understanding(Relationship.FOLLOW_UP))
    without = select_context(history, understanding(Relationship.CLARIFICATION))

    assert SUMMARY in with_summary
    assert SUMMARY not in without, "clarification needs the words, not the digest"


def test_plan_wants_nothing_is_distinguishable_from_everything():
    assert ContextPlan().wants_nothing
    assert not ContextPlan(everything=True).wants_nothing
    assert not ContextPlan(recent_pairs=1).wants_nothing
    assert not ContextPlan(include_anchor=True).wants_nothing
