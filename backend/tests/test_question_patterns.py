"""Natural interviewer phrasing: which utterances are answerable as-is.

Pure detector-level classification -- no event loop, no timing, no sleeps.
Every case is a real phrasing shape an interviewer uses, grouped by the
category it exercises. The companion file `test_turn_assembly.py` covers what
the session then *does* with these (how many provider calls, what exact text).

The single invariant behind the whole file: an utterance is held only when
deterministic evidence says it is not yet a whole request. Everything else
goes immediately.
"""

import pytest

from app.realtime.question_detector import Finality, QuestionDetector
from app.schemas.classification import Category

# --------------------------------------------------------------- immediate
# Categories 1, 2, 5, 6: complete requests, with and without "?".


IMMEDIATE = [
    # 1. direct questions
    "What is Azure Databricks?",
    "Explain dependency injection.",
    "How does Kafka work?",
    "Why did you choose PostgreSQL?",
    "Tell me about your experience with Spark.",
    # 2. questions with no question mark
    "Explain how you would design this system.",
    "Tell me how you would debug this.",
    "Describe your approach.",
    "Walk me through the architecture.",
    # 5. coding questions that are already whole
    "Write a function to reverse a linked list.",
    "Implement an LRU cache.",
    "How would you optimize this query?",
    "Given an array of integers, find two numbers that sum to a target.",
    # 6. behavioural questions that are already whole
    "Tell me about a time you handled conflict.",
    "Can you walk me through a difficult project?",
    "Describe a production incident you handled.",
]


@pytest.mark.parametrize("text", IMMEDIATE)
def test_a_whole_request_is_never_held(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"
    assert detection.finality is Finality.COMPLETE
    assert detection.hold_ms == 0, "a complete question must not pay any hold"


@pytest.mark.parametrize("text", IMMEDIATE)
def test_a_whole_request_keeps_its_exact_wording(text):
    """Category 13. Detection may classify, but it must not rewrite: the only
    permitted edits are the established trailing-punctuation normalisation."""
    detection = QuestionDetector().inspect(text)
    stripped = text.rstrip(".!?").lower()
    assert detection.text.rstrip("?").lower() == stripped
    assert detection.effective_text.endswith(detection.text)


# ------------------------------------------------------------ accumulating
# Categories 3, 4, 5: fragments that are provably not a request yet.


@pytest.mark.parametrize(
    "text",
    [
        # bare trigger, nothing asked for
        "Can you explain",
        "Tell me about",
        "Could you describe",
        "Walk me through",
        # trailing into a clause that says nothing yet
        "Can you explain how",
        "Can you explain what happens when",
        "How would you scale this if",
        "Tell me about a project you built and",
        # 4. scenario openers with no request in them
        "Given an array of integers",
        "Suppose you're working with Azure Databricks, tell me",
        # placeholder object still waiting for its relative clause
        "Tell me about a time",
        "Describe a situation",
        "Walk me through an example",
    ],
)
def test_an_incomplete_fragment_accumulates(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.finality is Finality.ACCUMULATING
    assert detection.hold_ms > 0


def test_a_quantity_without_its_constraint_is_ambiguous_not_incomplete():
    """"find two numbers" is answerable, just probably not as meant -- so it
    gets the shorter tier, and must be distinguishable from a bare premise."""
    ambiguous = QuestionDetector().inspect("Given an array of integers, find two numbers")
    assert ambiguous.finality is Finality.POTENTIALLY_COMPLETE

    incomplete = QuestionDetector().inspect("Given an array of integers")
    assert incomplete.finality is Finality.ACCUMULATING
    assert ambiguous.hold_ms <= incomplete.hold_ms


def test_a_placeholder_object_outranks_the_ambiguous_tier():
    """"a time" is not answerable at all, unlike "two numbers"."""
    assert (
        QuestionDetector().inspect("Tell me about a time").finality
        is Finality.ACCUMULATING
    )
    # ...but a specific object is complete.
    assert (
        QuestionDetector().inspect("Tell me about your last project.").finality
        is Finality.COMPLETE
    )


# --------------------------------------------------------------- closure
# Category 12 / 15: an explicit "?" is the one thing that ends a turn early.


def test_an_explicit_question_mark_is_reported_as_closure():
    assert QuestionDetector().inspect("What is Azure Databricks?").explicit_closure


def test_a_prompt_without_a_question_mark_is_not_closure():
    """Which is why these still work, but cannot short-circuit a hold."""
    detection = QuestionDetector().inspect("Walk me through the architecture.")
    assert detection.accepted
    assert not detection.explicit_closure


def test_an_explicit_question_mark_beats_a_scenario_opener():
    """"If you had to choose...?" opens like a premise but Whisper closed it."""
    detection = QuestionDetector().inspect("If you had to choose between Redis and Memcached?")
    assert detection.finality is Finality.COMPLETE
    assert detection.explicit_closure


# --------------------------------------------------------------- follow-ups
# Category 7: never held, and never merged into the previous question.


FOLLOWUPS = [
    "Why?", "How?", "What?",
    "Why did you choose that?",
    "How would you change it?",
    "Can you elaborate?",
    "Can you explain that?",
    "And at scale?",
    "What about failure recovery?",
    "What would happen if the node died?",
]


@pytest.mark.parametrize("followup", FOLLOWUPS)
def test_a_followup_is_immediate(followup):
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)

    detection = detector.inspect(followup, now=112.0)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"
    assert detection.finality is Finality.COMPLETE
    assert detection.hold_ms == 0


def test_a_bare_followup_needs_a_recent_question():
    """Without one it is just a stray word, not a question."""
    detection = QuestionDetector().inspect("Why?", now=100.0)
    assert not detection.accepted


# ----------------------------------------------------------------- fillers
# Category 9: harmless filler is not a question and not a boundary.


@pytest.mark.parametrize(
    "filler",
    ["So", "Okay, so", "Right", "Basically", "Let's see", "Um", "Okay", "Yeah"],
)
def test_filler_is_not_a_question(filler):
    assert not QuestionDetector().inspect(filler).accepted


@pytest.mark.parametrize("ack", ["Okay.", "Got it.", "That makes sense.", "Right."])
def test_an_acknowledgement_never_merges_into_the_question_before_it(ack):
    """Merging filler onto a question re-asks it and bills a second answer."""
    detector = QuestionDetector()
    detector.inspect("Find two numbers whose sum equals a target.", now=100.0)

    detection = detector.inspect(ack, now=100.2)
    assert not detection.accepted
    assert "sum equals" not in detection.text


# ------------------------------------------------------------- corrections
# Category 8: a self-revision replaces the premise it revises.


@pytest.mark.parametrize(
    "correction",
    [
        "Actually, assume 10,000 QPS",
        "Sorry, use DynamoDB instead",
        "No, let's use Postgres",
        "Scratch that, assume it is read-heavy",
        "I mean 10,000 QPS",
    ],
)
def test_a_correction_replaces_the_buffered_premise(correction):
    detector = QuestionDetector()
    detector.inspect("Assume 1,000 QPS", now=100.0)
    detector.inspect(correction, now=101.0)

    question = detector.inspect("How would you design the service?", now=102.0)
    assert question.accepted
    assert "1,000" not in question.effective_text, (
        f"obsolete premise survived the correction: {question.effective_text!r}"
    )


def test_a_non_correction_premise_still_accumulates_context():
    """The correction rule must not eat ordinary multi-clause setup."""
    detector = QuestionDetector()
    detector.inspect("Suppose you have a stream of events", now=100.0)
    detector.inspect("arriving out of order", now=101.0)

    question = detector.inspect("How would you deduplicate them?", now=102.0)
    assert "stream of events" in question.effective_text
    assert "out of order" in question.effective_text


# ------------------------------------------------- duplicate / stale finals
# From the "also test" list: the STT path can redeliver or reorder.


def test_a_duplicate_final_is_not_concatenated_onto_itself():
    detector = QuestionDetector()
    first = detector.inspect("What is caching?", now=100.0)
    repeat = detector.inspect("What is caching?", now=100.3)

    assert repeat.accepted
    assert not repeat.supersedes
    assert repeat.text == first.text, "a redelivered final must not double up"


def test_an_out_of_order_final_does_not_merge_backwards():
    """A short utterance overtaking a long one produces a negative speech-clock
    gap, which would pass any naive window test."""
    detector = QuestionDetector()
    detector.inspect("How would you design a URL shortener?", now=105.0)

    stale = detector.inspect("What is a database index?", now=100.0)
    assert stale.accepted
    assert not stale.supersedes
    assert "URL shortener" not in stale.text


def test_an_empty_final_is_rejected_without_state_change():
    detector = QuestionDetector()
    accepted = detector.inspect("What is Azure Databricks?", now=100.0)
    assert accepted.accepted

    for junk in ("", "   ", "?? -- ..."):
        assert not detector.inspect(junk, now=100.5).accepted

    # The real question is still the merge anchor, unpolluted by the junk.
    followup = detector.inspect("Why?", now=101.0)
    assert followup.accepted and followup.detail == "follow_up"


# ------------------------------------------------------- turn independence
# Category 11: an answered turn is no longer something to extend.


def test_closing_a_turn_drops_the_long_imperative_merge_window():
    """Two unrelated coding questions a few seconds apart must not merge just
    because the imperative window is wide."""
    detector = QuestionDetector()
    detector.inspect("Write a function to reverse a linked list.", now=100.0)
    detector.close_turn()

    second = detector.inspect("Implement an LRU cache.", now=103.0)
    assert second.accepted
    assert not second.supersedes
    assert "linked list" not in second.text


def test_closing_a_turn_keeps_followups_working():
    """close_turn must not cost the follow-up bypass, which depends on the
    same recency state."""
    detector = QuestionDetector()
    detector.inspect("What is a hash map?", now=100.0)
    detector.close_turn()

    followup = detector.inspect("Why?", now=103.0)
    assert followup.accepted
    assert followup.detail == "follow_up"


# ------------------------------------------------------------ classification
# Sanity that the categories still route, since the prompts above changed.


@pytest.mark.parametrize(
    "text,category",
    [
        ("Write a function to reverse a linked list.", Category.CODING),
        ("Tell me about a time you had a team conflict.", Category.BEHAVIORAL),
        ("How would you design a URL shortener?", Category.SYSTEM_DESIGN),
        ("Find the second highest salary.", Category.SQL),
    ],
)
def test_categories_still_route(text, category):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.classification.category == category

# ------------------------------------------------- task-continuation verbs
# An interviewer advancing an existing task uses a different verb set from
# one opening a new one. These were rejected outright before the vocabulary
# included them, so the step was silently lost mid-progression.

CONTINUATIONS = [
    "Now handle a stream of integers.",
    "Now optimize it for high throughput.",
    "Now refactor it using a different approach.",
    "Now extend it to support retries.",
    "Now modify it to handle null values.",
    # Both spellings, following the existing summari[sz]e convention.
    "Now optimise it for throughput.",
]


@pytest.mark.parametrize("text", CONTINUATIONS)
def test_a_task_continuation_is_accepted(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"


@pytest.mark.parametrize("text", CONTINUATIONS)
def test_a_task_continuation_is_answerable_immediately(text):
    """These are complete instructions, so they must not pay a hold."""
    detection = QuestionDetector().inspect(text)
    assert detection.finality is Finality.COMPLETE
    assert detection.hold_ms == 0


@pytest.mark.parametrize("text", CONTINUATIONS)
def test_a_task_continuation_reads_as_an_imperative_task(text):
    """The detail matters downstream: it selects the longer merge window and
    is what the understanding layer's fallback reads."""
    detection = QuestionDetector().inspect(text)
    assert detection.detail == "imperative_task"


@pytest.mark.parametrize("text", CONTINUATIONS)
def test_a_task_continuation_keeps_its_exact_wording(text):
    detection = QuestionDetector().inspect(text)
    assert detection.text.rstrip("?").lower() == text.rstrip(".").lower()


@pytest.mark.parametrize(
    "text",
    [
        # The new verbs mid-sentence, which clause-initial anchoring must keep
        # out -- this is the guard that stops the vocabulary from firing on
        # ordinary interviewer speech.
        "I usually handle that with a retry.",
        "We modify the config at deploy time.",
        "That is how I would extend it.",
        "The team refactored it last quarter.",
        "We optimize for readability over cleverness there.",
    ],
)
def test_the_new_verbs_do_not_fire_mid_sentence(text):
    assert not QuestionDetector().inspect(text).accepted


def test_a_continuation_can_follow_a_coding_problem(text=None):
    """The case this vocabulary was added for: a progression step that used to
    be dropped now merges into the thread instead."""
    detector = QuestionDetector()
    first = detector.inspect("Write a function to find duplicate numbers.", now=100.0)
    assert first.accepted

    step = detector.inspect("Now handle a stream of integers.", now=140.0)
    assert step.accepted
    assert step.finality is Finality.COMPLETE
    # Its own words survive; it is a step, not a restatement of the problem.
    assert "stream of integers" in step.text
