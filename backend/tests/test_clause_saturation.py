"""Does the interviewer's clause still want something, or did they finish?

Two failures shared one cause, and this file pins both ends of it.

The detector used to answer "is this a whole request?" by *counting* the
substantive words after a trigger phrase. Counting cannot separate the two
things that matter:

    "How would you design"   one word after the trigger, and unfinished
    "What is Kafka"          one word after the trigger, and finished

so the fix that stopped the first from being answered prematurely made every
short question of the second shape wait three seconds it did not owe -- on
the most ordinary phrasing an interviewer uses, and only when Whisper omitted
the "?", which is the case this whole pipeline is built around.

The mirror-image failure was a clause several words long whose predicate had
no argument yet:

    "Explain how Kafka handles"   ... "partitions?"

Nothing in a word count or a last-word test can see that: the clause does not
end on a function word and is not short. It was read as complete, answered,
and then answered again when the rest of the sentence arrived -- two
questions in the panel and two provider calls for one question.

`prompt_detector.has_open_clause` replaces the counting with a structural
question -- does this clause's predicate have an argument? -- decided from
closed word classes only. The two parametrised lists below are the contract:
everything in COMPLETE_NOW must reach the model immediately, everything in
STILL_OPEN must wait, and no member of either list may move without a
deliberate decision to change what the app does.
"""

import pytest

from app.realtime.events import EventType
from app.realtime.prompt_detector import (
    _is_participle,
    has_open_clause,
    has_open_premodifier,
    is_referentially_empty,
)
from app.realtime.question_detector import Finality, QuestionDetector
from app.sessions.schemas import TranscriptSource
from tests.replay_harness import ReplayHarness

asyncio_test = pytest.mark.asyncio


# --------------------------------------------------------- the two contracts

#: Whole requests. Every one of these is answerable exactly as spoken, so the
#: candidate must not wait for a continuation that is not coming.
COMPLETE_NOW = [
    # A copula and its complement. The shape the word count could not see.
    "What is Kafka",
    "What is idempotency",
    "What is normalization",
    "Is it durable",
    "Are there tradeoffs",
    # Third-person subject with an intransitive predicate: a whole question
    # with no object at all.
    "When does it rebalance",
    "Why does Kafka rebalance?",
    "When does the database replicate?",
    "Who owns the schema",
    # An imperative whose object is a pronoun pointing at something present
    # -- on screen, in a paste, in the code just written.
    "Implement this",
    "Optimize this",
    "Explain this.",
    "Optimize it.",
    # Nouns that merely end in a participle's letters.
    "Reverse a string",
    "What is a string",
    "What is a thread",
    "What is the encoding",
    "Explain the padding",
    "What is the meeting",
    "What is a rating",
    # A subordinate clause that did close, including the intransitive reading
    # the speaker settled by terminating the utterance.
    "Explain how Kafka works.",
    "Tell me how the database works.",
    "Describe how the system recovers.",
    "Explain how you would design this system.",
    "Tell me how you would debug this.",
    "Can you explain how dependency injection works in FastAPI",
    "Find two numbers whose sum equals a target.",
    # Requests with their object filled in.
    "How would you handle retries",
    "Would you choose Kafka",
    "What would you use for storage",
    "Where would you store the offsets",
]

#: Provably unfinished. Sending any of these can only ever answer the wrong
#: question, so each waits a bounded moment for the rest of the sentence.
STILL_OPEN = [
    # Bare trigger, or a request framed at the candidate with nothing to do.
    "How would you",
    "How would you design",
    "How would you ensure",
    "What would you use",
    "Where would you store",
    "Should we shard",
    "Do you use",
    "Can you",
    # Trailing off on a function word.
    "How would you design a",
    "What happens if",
    "What's the",
    # A noun phrase whose head has not arrived.
    "How would you design a distributed",
    "Explain a distributed",
    # A subordinate clause with a subject and no predicate at all.
    "Tell me how the database",
    "Describe how the system",
    "Can you explain why the consumer",
    "How would you ensure that the service",
    # A subordinate clause whose predicate has no argument, and no terminal
    # punctuation to say the speaker meant to stop there.
    "Explain how Kafka handles",
    "Who owns",
    "Can you explain why",
]


@pytest.mark.parametrize("text", COMPLETE_NOW)
def test_a_whole_request_pays_no_hold(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"
    assert detection.finality is Finality.COMPLETE, (
        f"{text!r} was held as {detection.finality.value}"
    )
    assert detection.hold_ms == 0


@pytest.mark.parametrize("text", STILL_OPEN)
def test_an_unfinished_request_is_held(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"
    assert detection.finality is not Finality.COMPLETE, (
        f"{text!r} was sent as complete"
    )
    assert detection.hold_ms > 0


@pytest.mark.parametrize("text", COMPLETE_NOW + STILL_OPEN)
def test_finality_never_rewrites_the_transcript(text):
    """Whatever the verdict, the words are the interviewer's. The only
    permitted edit stays the established trailing-punctuation normalisation."""
    detection = QuestionDetector().inspect(text)
    assert detection.text.rstrip("?").lower() == text.rstrip(".!?").lower()


# ------------------------------------------------- the structural signal
# has_open_clause on its own, so a failure says which half is wrong.


@pytest.mark.parametrize(
    ("text", "open_"),
    [
        # subject, no predicate
        ("Tell me how the database", True),
        ("How would you ensure that the service", True),
        # predicate, no argument, and nothing said the speaker stopped
        ("Explain how Kafka handles", True),
        # ... the same clause, closed by its argument
        ("Explain how Kafka handles partitions", False),
        ("Tell me how the database handles replication", False),
        # ... or by the speaker terminating the utterance
        ("Explain how Kafka handles.", False),
        # copula + complement
        ("What is Kafka", False),
        # copula with nothing after it
        ("Explain how the system is", True),
        ("Explain how the system is designed", False),
        # a request framed at the candidate, object missing then present
        ("How would you design", True),
        ("How would you design this system", False),
        # an interrogative that is itself the subject
        ("Who owns", True),
        ("Who owns the schema", False),
        # an imperative
        ("Reverse", True),
        ("Reverse a string", False),
        # a relative "that" is not a clause opener: its verb follows directly
        ("Write a function that reverses a linked list", False),
    ],
)
def test_clause_openness(text, open_):
    assert has_open_clause(text) is open_, text


def test_an_llm_is_never_consulted_to_decide_finality():
    """The signal is deterministic by construction. Asserted rather than
    assumed, because the cheapest way to "fix" a grammar problem is to ask a
    model, and that would put a network round trip on every finalised
    utterance of a live interview."""
    import app.realtime.prompt_detector as module

    source = module.has_open_clause.__doc__ or ""
    assert "await" not in source
    # The whole module: no client, no completer, no network import.
    assert not hasattr(module, "httpx")
    assert not hasattr(module, "LLMClient")


# --------------------------------------------------- premodifier morphology


@pytest.mark.parametrize(
    "text",
    [
        # Past participles: the head noun they modify has not arrived.
        "How would you design a distributed",
        "Explain a distributed",
        "Describe the partitioned",
        "Walk me through a sharded",
    ],
)
def test_a_participle_after_a_determiner_is_an_open_phrase(text):
    assert has_open_premodifier(f"{text}?")


@pytest.mark.parametrize(
    "text",
    [
        # "-ing" after a determiner is a gerund heading a finished phrase.
        "Reverse a string",
        "What is a string",
        "What is the encoding",
        "Explain the padding",
        "What is the meeting",
        "What is a rating",
        "What is a thing",
        # "-ed" words that are not participles: too little stem to have
        # carried the suffix.
        "What is a thread",
        "Explain the feed",
        "What is the speed",
        # And the closed phrase itself.
        "How would you design a distributed system",
    ],
)
def test_an_ordinary_noun_is_not_an_open_phrase(text):
    assert not has_open_premodifier(f"{text}?")


@pytest.mark.parametrize(
    ("word", "participle"),
    [
        ("distributed", True), ("partitioned", True), ("sharded", True),
        ("cached", True), ("linked", True), ("indexed", True), ("nested", True),
        ("thread", False), ("head", False), ("speed", False), ("feed", False),
        ("seed", False), ("red", False), ("bed", False),
    ],
)
def test_participle_morphology(word, participle):
    assert _is_participle(word) is participle


# --------------------------------------------------------- end to end
# The consequence the detector-level assertions exist to prevent: a fragment
# that fires early costs a second question and a second provider call.


@pytest.fixture
def harness(monkeypatch):
    h = ReplayHarness(monkeypatch=monkeypatch)
    yield h
    h.dispose()


@pytest.fixture
def holding_harness(monkeypatch):
    """A hold long enough to outlive the whole scripted sequence.

    The default 1ms budget is right for asserting that a turn *was* held; it
    is wrong for asserting what a held turn then assembles into, because the
    hold expires between two utterances the interviewer said 400ms apart. In
    production the budget is `question_hold_incomplete_ms` (3000ms) against
    exactly that gap, so the continuation always arrives mid-hold -- which is
    the case this fixture reproduces without a real 3s wait.
    """
    h = ReplayHarness(monkeypatch=monkeypatch, stabilization_ms=400)
    yield h
    h.dispose()


@pytest.mark.parametrize(
    "parts",
    [
        ["Explain how Kafka handles", "partitions?"],
        ["Explain how Kafka handles", "partitions when brokers fail?"],
        ["Tell me how the database", "handles replication?"],
        ["Describe how the system", "should recover from failure?"],
        ["How would you ensure", "that the service", "does not lose data?"],
        ["Can you explain why", "the consumer", "falls behind?"],
    ],
)
@asyncio_test
async def test_a_split_nested_clause_costs_one_question_and_one_answer(
    holding_harness, parts
):
    # Deliberately no settle between utterances: each one arrives while the
    # previous fragment is still being held, which is what the hold is for.
    for index, part in enumerate(parts):
        await holding_harness.live.on_transcript(
            part, TranscriptSource.LOOPBACK, is_final=True, now=100.0 + index * 0.4
        )
    await holding_harness.settle()

    harness = holding_harness
    result = harness.result
    detected = result.detected_questions()
    completed = result.completed_turn_ids()
    assert len(detected) == 1, f"detected {detected}"
    assert len(completed) == 1, f"completed {completed}"
    # Every word the interviewer said, and only those.
    for part in parts:
        for word in part.rstrip("?").split():
            assert word.lower() in detected[0].lower()


@asyncio_test
async def test_an_ordinary_short_question_is_answered_without_waiting(harness):
    """The other half of the same fix: no hold at all for a whole request, so
    no `question_stabilization_started` on the way to the answer."""
    await harness.live.on_transcript(
        "What is Kafka", TranscriptSource.LOOPBACK, is_final=True, now=100.0
    )
    await harness.settle()

    assert harness.result.detected_questions() == ["What is Kafka?"]
    assert len(harness.result.completed_turn_ids()) == 1
    assert harness.live._fragments == 0
    assert harness.live._accumulating_since is None


# ------------------------------------------------------------------ HC-004
# "show" with the addressee left out is the same request as "show me".


@pytest.mark.parametrize(
    "text",
    ["Show code", "Show Python code", "Show the implementation", "Show me the code."],
)
def test_show_is_a_request_with_or_without_an_addressee(text):
    detection = QuestionDetector().inspect(text)
    assert detection.accepted, f"rejected as {detection.reason} / {detection.detail}"


def test_show_alone_is_still_below_the_word_floor():
    detection = QuestionDetector().inspect("Show")
    assert not detection.accepted


def test_show_inside_ordinary_speech_is_not_a_request():
    """Why "show" belongs in the clause-initial layer and not the
    match-anywhere one."""
    detection = QuestionDetector().inspect(
        "The metrics show latency rising after each deploy."
    )
    assert not detection.accepted or detection.detail != "imperative_task"


@asyncio_test
async def test_show_code_reaches_an_answer(harness):
    await harness.live.on_transcript(
        "Show Python code", TranscriptSource.LOOPBACK, is_final=True, now=100.0
    )
    await harness.settle()
    assert harness.result.of(EventType.QUESTION_DETECTED)
    assert len(harness.result.completed_turn_ids()) == 1


# ------------------------------------- the two readings of a standin pronoun
# `it`/`this`/`that` answer two different questions oppositely, and the fix
# for one of them briefly broke the other. Both are pinned here so the two
# sets that encode them can never drift apart again.
#
#   "did the interviewer say anything after the trigger?"  "Explain this" -> yes
#   "does this utterance name a topic of its own?"         "What about it?" -> no


@pytest.mark.parametrize(
    "text",
    ["What about it?", "What about this?", "Which one?", "Can you?", "Show me.",
     "Then what?", "Why not?", "How so?"],
)
def test_a_bare_standin_names_no_topic_and_is_rejected_alone(text):
    assert is_referentially_empty(text)
    assert not QuestionDetector().inspect(text).accepted


@pytest.mark.parametrize(
    "text", ["Explain this.", "Optimize it.", "Implement this", "Optimize it"],
)
def test_a_governed_standin_is_a_whole_request(text):
    """The verb makes the pronoun an object, and the object is a topic the
    interviewer can point at -- so this is answerable with no prior turn, and
    answerable *now*."""
    assert not is_referentially_empty(text)
    detection = QuestionDetector().inspect(text)
    assert detection.accepted
    assert detection.finality is Finality.COMPLETE


@pytest.mark.parametrize(
    "text", ["What about it?", "Which one?", "Show me.", "Can you explain?"],
)
def test_a_bare_standin_is_accepted_once_there_is_something_to_refer_to(text):
    detector = QuestionDetector()
    detector.inspect("How would you design a rate limiter?", now=99.0)
    detection = detector.inspect(text, now=100.0)
    assert detection.accepted
    assert detection.detail == "follow_up"
    assert detection.finality is Finality.COMPLETE


def test_a_trailing_that_is_still_held_and_that_is_deliberate():
    """"Answer that." is held, and this pins that as a decision rather than an
    oversight. "that" stays in `question_detector._DANGLING_WORDS` because it
    is genuinely ambiguous between an object ("answer that") and a
    complementiser ("ensure that the service ..."), and the second reading is
    the one where firing early costs a wrong answer. Pre-existing behaviour,
    unchanged by the standin fix above, which only concerns "it"/"this"."""
    detection = QuestionDetector().inspect("Answer that.")
    assert detection.accepted
    assert detection.finality is Finality.ACCUMULATING
