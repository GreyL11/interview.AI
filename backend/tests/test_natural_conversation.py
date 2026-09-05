"""Interviewer speech as it actually arrives, and what "this" resolves against.

Three things are pinned down here that nothing else covered:

* **Premises.** A real question is rarely one sentence. "We have ten million
  rows in the orders table. How would you speed up this join?" was reaching the
  model as the second sentence alone -- speed up which join, on a table of
  unknown size.
* **Previously generated code.** The conversation window carries answer
  *summaries*, so the code the model itself wrote was unreachable. "Explain the
  code you wrote" had nothing to explain.
* **References.** "Explain this" has no fixed meaning. The adversarial cases at
  the bottom exist to fail if anyone ever resolves these with a word list: the
  same phrase must land differently depending only on what context exists, and
  differently-worded turns with the same meaning must land the same way.

Deterministic throughout: scripted classifier output, explicit timestamps, no
sleeps and no provider.
"""

import pytest

from app.realtime.events import EventType
from app.realtime.prompt_detector import extract_interview_prompt
from app.realtime.question_detector import QuestionDetector
from app.memory.sqlite_memory import SqliteSessionMemory
from app.schemas.answer import Answer
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayHarness
from tests.test_understanding_session import CountingCompleter, reply

asyncio_test = pytest.mark.asyncio

CODE = "def two_sum(nums, target):\n    seen = {}\n    return seen"
PASTED_CODE = "def broken(rows):\n    return sum(r['amount'] for r in rows)"
ERROR_LOG = "TypeError: 'NoneType' object is not subscriptable\n  at broken(rows)"


def harness_with(completer, monkeypatch, answer=None) -> ReplayHarness:
    h = ReplayHarness(
        llm=SlowStreamingLLM(answer=answer, chunk_delay=0),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    if completer is not None:
        from app.realtime.question_understanding import QuestionUnderstander

        h.live._understander = QuestionUnderstander(completer)
    return h


async def say(h, text, now):
    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    await h.settle()


async def paste(h, content, now, kind="code"):
    await h.live.on_context_attached(kind=kind, content=content, now=now)
    await h.settle()


def prompts(h) -> list[str]:
    return h.llm.prompts


def coding_answer() -> Answer:
    return Answer(
        summary="Use a dictionary of seen values.",
        approach=["scan once", "look up the complement"],
        code=CODE,
        complexity={"time": "O(n)", "space": "O(n)"},
    )


# ============================================ pillar 31. multi-sentence premise


@pytest.mark.parametrize(
    ("utterance", "premise_marker", "question_marker"),
    [
        (
            "We have ten million rows in the orders table. "
            "How would you speed up this join?",
            "ten million rows",
            "speed up this join",
        ),
        (
            "Suppose you have a Kafka pipeline processing one million events per "
            "second. How would you make it reliable if one consumer goes down?",
            "one million events per second",
            "one consumer goes down",
        ),
        (
            "The table has no index on customer_id. Why is this query slow?",
            "no index on customer_id",
            "Why is this query slow",
        ),
    ],
)
def test_a_premise_survives_into_what_the_model_sees(
    utterance, premise_marker, question_marker
):
    """The sentence scanner keeps the last prompt-like sentence, which is right
    for dropping an acknowledgement and wrong for a constraint."""
    detection = QuestionDetector().inspect(utterance, now=100.0)

    assert detection.accepted
    assert question_marker in detection.effective_text
    assert premise_marker in detection.effective_text, "the premise was dropped"


def test_the_panel_still_shows_only_the_question():
    """`text` drives the interview panel and the turns table. A premise belongs
    in what the model reads, not in what the interviewer is shown having
    asked."""
    detection = QuestionDetector().inspect(
        "We have ten million rows in the orders table. How would you speed up this join?",
        now=100.0,
    )
    assert detection.text == "How would you speed up this join?"
    assert "ten million" not in detection.text


@pytest.mark.parametrize(
    "utterance",
    [
        "Very good. So tell me what is the difference between a list and a tuple?",
        "Okay, thanks. Now explain Python decorators.",
        "Right, got it. Explain closures.",
    ],
)
def test_pleasantries_are_still_discarded_not_promoted_to_premises(utterance):
    """The behaviour the last-match scan exists for must survive: an
    acknowledgement is not a premise."""
    match = extract_interview_prompt(utterance)
    assert match is not None
    assert match.premise == "", f"filler kept as premise: {match.premise!r}"


def test_a_premise_is_bounded():
    """An interviewer who monologues contributes the part nearest the question,
    not an unbounded prefix."""
    match = extract_interview_prompt(
        "So. " + ("This is a long setup sentence. " * 60) + "How would you scale it?"
    )
    assert match is not None
    assert len(match.premise) <= 400
    # Tail-anchored: the sentence just before the question is the one most
    # likely to carry its constraint.
    assert match.premise.endswith("This is a long setup sentence.")


@asyncio_test
async def test_a_long_question_costs_one_turn_and_one_call(monkeypatch):
    completer = CountingCompleter(reply(intent="system_design"))
    h = harness_with(completer, monkeypatch)
    try:
        await say(
            h,
            "Suppose you have a Kafka pipeline processing one million events per "
            "second. How would you make it reliable if one consumer goes down, "
            "and what tradeoffs would your design introduce?",
            now=100.0,
        )

        assert len(prompts(h)) == 1, "a multi-sentence turn split into two"
        assert completer.calls == 1
        assert "one million events per second" in prompts(h)[0]
        assert "tradeoffs would your design introduce" in prompts(h)[0]
    finally:
        h.dispose()


# ======================================= pillar 4. previously generated code


@asyncio_test
async def test_code_the_model_wrote_can_be_asked_about(monkeypatch):
    """The window carries summaries only, so without this the model's own code
    was unreachable and "explain the code you wrote" had nothing to explain."""
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="conceptual", relationship="follow_up", needs_previous_code=True),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, "Write a function to find two numbers summing to a target.",
                  now=100.0)
        await say(h, "Can you explain the code you wrote?", now=120.0)

        second = prompts(h)[1]
        assert CODE in second, "the generated code never reached the prompt"
        assert "FROM YOUR OWN EARLIER ANSWER" in second, "provenance not labelled"
    finally:
        h.dispose()


@asyncio_test
async def test_generated_code_is_withheld_unless_the_turn_refers_to_it(monkeypatch):
    """Otherwise every later question in the session carries a stale
    implementation."""
    completer = CountingCompleter(
        reply(intent="coding", relationship="new_question"),
        reply(intent="conceptual", relationship="new_question",
              needs_previous_code=False),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, "Write a function to find two numbers summing to a target.",
                  now=100.0)
        await say(h, "What is Azure Databricks?", now=300.0)

        assert CODE not in prompts(h)[1], "stale code leaked into a new question"
    finally:
        h.dispose()


@asyncio_test
async def test_a_broken_classifier_does_not_quote_code_back(monkeypatch):
    """`deterministic_fallback` cannot judge whether *this* question refers to
    earlier code, so the failure path must not act as if it had."""
    completer = CountingCompleter(error=RuntimeError("provider down"))
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, "Write a function to find duplicates.", now=100.0)
        await say(h, "Can you explain the code you wrote?", now=120.0)

        assert CODE not in prompts(h)[1]
        # The turn is still answered.
        assert len(prompts(h)) == 2
    finally:
        h.dispose()


@asyncio_test
async def test_the_newest_implementation_is_the_one_quoted(monkeypatch):
    """A progression refers to the last thing written, not the first."""
    first = Answer(summary="v1", code="def v1(): pass")
    completer = CountingCompleter(
        reply(intent="coding"),
        reply(intent="coding", relationship="new_implementation"),
        reply(relationship="follow_up", needs_previous_code=True),
    )
    h = harness_with(completer, monkeypatch, answer=first)
    try:
        await say(h, "Write a function to find duplicates.", now=100.0)
        h.llm.answer = Answer(summary="v2", code="def v2(): pass")
        await say(h, "Now give me the version without extra space.", now=120.0)
        await say(h, "Why did you write it that way?", now=140.0)

        third = prompts(h)[2]
        assert "def v2(): pass" in third
        assert "def v1(): pass" not in third, "an older implementation was quoted"
    finally:
        h.dispose()


@asyncio_test
async def test_pasted_code_and_generated_code_stay_distinguishable(monkeypatch):
    """Both can be in one prompt, and they are not the same thing: one was
    handed over by the interviewer, one is the candidate's own prior output."""
    completer = CountingCompleter(
        reply(intent="coding"),
        reply(intent="troubleshooting", relationship="follow_up",
              needs_previous_code=True, needs_attachments=True),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, "Write a function to total the amounts.", now=100.0)
        await paste(h, PASTED_CODE, now=110.0)
        await say(h, "How does yours differ from this one?", now=111.0)

        final = prompts(h)[1]
        assert CODE in final and PASTED_CODE in final
        assert final.index("MATERIAL THE INTERVIEWER PROVIDED") < final.index(
            "FROM YOUR OWN EARLIER ANSWER"
        )
    finally:
        h.dispose()


# ============================== pillar 5/29. references, and the adversarial half


@asyncio_test
async def test_the_same_phrase_resolves_differently_by_context(monkeypatch):
    """"Explain this." against a paste, then against generated code, then
    against nothing. Identical words, three outcomes -- impossible for a word
    list to produce, which is the point of this test."""
    completer = CountingCompleter(
        reply(needs_attachments=True),                       # paste present
    )
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED_CODE, now=100.0)
        await say(h, "Explain this.", now=101.0)
        assert PASTED_CODE in prompts(h)[0]
    finally:
        h.dispose()

    completer = CountingCompleter(
        reply(intent="coding"),
        reply(needs_previous_code=True, relationship="follow_up"),
    )
    h = harness_with(completer, monkeypatch, answer=coding_answer())
    try:
        await say(h, "Write a function to find duplicates.", now=100.0)
        await say(h, "Explain this.", now=110.0)
        assert CODE in prompts(h)[1]
        assert PASTED_CODE not in prompts(h)[1]
    finally:
        h.dispose()

    completer = CountingCompleter(reply(relationship="new_question"))
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "Explain this.", now=100.0)
        # Answered, but with no material and no history: the model is entitled
        # to ask what "this" is, and the prompt must not invent a referent.
        assert len(prompts(h)) == 1
        assert "MATERIAL THE INTERVIEWER PROVIDED" not in prompts(h)[0]
        assert "FROM YOUR OWN EARLIER ANSWER" not in prompts(h)[0]
        assert "Previous Q&A" not in prompts(h)[0]
    finally:
        h.dispose()


@pytest.mark.parametrize(
    "phrasing",
    [
        "Explain this.",
        "Answer this.",
        "What about this?",
        "Walk me through this.",
        "Tell me the result for this.",
        "Can you explain the above?",
    ],
)
@asyncio_test
async def test_different_wordings_of_one_reference_behave_alike(monkeypatch, phrasing):
    """The mirror of the test above: same available context, six wordings, one
    outcome. A word list would have to enumerate all of them."""
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED_CODE, now=100.0)
        await say(h, phrasing, now=101.0)

        assert len(prompts(h)) == 1
        assert PASTED_CODE in prompts(h)[0], f"{phrasing!r} did not resolve"
        # And the wording itself is untouched.
        assert phrasing.rstrip(".?") in prompts(h)[0]
    finally:
        h.dispose()


@asyncio_test
async def test_competing_referents_are_both_offered_not_guessed_between(monkeypatch):
    """Two unrelated pastes and a bare reference. The system must not silently
    pick one -- it hands over what it has and lets the model say it is
    ambiguous, which is the honest outcome."""
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED_CODE, now=100.0)
        await paste(h, ERROR_LOG, now=101.0, kind="text")
        await say(h, "Why is this failing?", now=102.0)

        prompt = prompts(h)[0]
        assert PASTED_CODE in prompt and ERROR_LOG in prompt
        # Arrival order preserved, so the model can reason about which is which.
        assert prompt.index(PASTED_CODE) < prompt.index(ERROR_LOG)
    finally:
        h.dispose()


@asyncio_test
async def test_a_reference_with_no_referent_does_not_borrow_an_old_one(monkeypatch):
    """The failure this guards: an unrelated question late in a session
    inheriting whatever material happened to still exist."""
    completer = CountingCompleter(
        reply(needs_attachments=True),
        reply(relationship="new_question", needs_attachments=False,
              needs_previous_code=False),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED_CODE, now=100.0)
        await say(h, "Explain this.", now=101.0)
        assert PASTED_CODE in prompts(h)[0]

        await say(h, "What is Azure Databricks?", now=400.0)
        assert PASTED_CODE not in prompts(h)[1], "material leaked to a new topic"
    finally:
        h.dispose()


# ================================================ pillar 21. retrieval isolation


@asyncio_test
async def test_retrieval_runs_per_question_not_per_session(monkeypatch):
    """Each turn retrieves against its own question, so an unrelated later
    question cannot inherit the earlier topic's chunks merely because they are
    in history."""
    completer = CountingCompleter(
        reply(relationship="new_question"), reply(relationship="new_question")
    )
    h = harness_with(completer, monkeypatch)
    queried: list[str] = []
    original = h.live._retriever.retrieve

    async def spy(question, **kwargs):
        queried.append(question)
        return await original(question, **kwargs)

    h.live._retriever.retrieve = spy
    try:
        # Personal questions, because retrieval deliberately does not run for
        # reasoning-routed ones -- see `_retrieve` / `route_for`.
        await say(h, "What is your experience with Kafka?", now=100.0)
        await say(h, "What is your experience with Spark?", now=200.0)

        assert len(queried) == 2
        assert "Kafka" in queried[0] and "Spark" in queried[1]
        assert "Kafka" not in queried[1], "retrieval reused the earlier question"
    finally:
        h.dispose()


# ================================================== pillar 25. bounded context


@asyncio_test
async def test_prompt_size_does_not_grow_with_interview_length(monkeypatch):
    """Turn 30 must cost about what turn 3 did. Unbounded growth here is a
    latency problem that only shows up late in a real interview, which is
    exactly when it matters most."""
    completer = CountingCompleter(*[reply(relationship="follow_up",
                                          needs_previous_context=True)] * 30)
    h = harness_with(completer, monkeypatch)
    try:
        for i in range(30):
            await say(h, f"Question number {i} about indexing?", now=100.0 + i * 20)

        sizes = [len(p) for p in prompts(h)]
        assert len(sizes) == 30
        early, late = sizes[2], sizes[-1]
        assert late < early * 2, (
            f"prompt grew with the interview: turn 3 {early}, turn 30 {late}"
        )
        # And the selected history stays at the documented bound.
        assert prompts(h)[-1].count("\nQ: ") <= 3
    finally:
        h.dispose()


# ============================================ pillar 32. STT imperfections


@pytest.mark.parametrize(
    ("noisy", "marker"),
    [
        # No terminal punctuation at all -- Whisper routinely gives "." or
        # nothing to a genuine question.
        ("how would you scale this system", "scale this system"),
        # A doubled word.
        ("How would would you scale this?", "scale this"),
        # Lowercased opening, missing apostrophe.
        ("whats the complexity of that", "complexity"),
        # Trailing filler word attached to the question.
        ("How would you scale this, um?", "scale this"),
    ],
)
def test_ordinary_transcription_noise_still_reads_as_a_question(noisy, marker):
    detection = QuestionDetector().inspect(noisy, now=100.0)
    assert detection.accepted, f"rejected: {detection.detail}"
    assert marker in detection.text


@pytest.mark.parametrize("junk", ["", "   ", ".", "?", "-- ...", "\n", "..."])
def test_a_junk_final_is_never_a_question_on_its_own(junk):
    assert not QuestionDetector().inspect(junk, now=100.0).accepted


@pytest.mark.parametrize("junk", [".", "?", "-- ...", "..."])
def test_junk_after_a_question_cannot_change_its_wording(junk):
    """Stray STT punctuation arriving just after a real question is merged
    onto it by design -- a trailing clause is usually real content, and the
    no-words guard only rejects a merge with no words at all. What must never
    happen is the question's own wording changing."""
    detector = QuestionDetector()
    first = detector.inspect("How would you scale this?", now=100.0)
    assert first.accepted

    after = detector.inspect(junk, now=100.5)
    if after.accepted:
        assert "How would you scale this" in after.text
        assert after.text.count("scale this") == 1


def test_a_redelivered_final_is_not_concatenated_onto_itself():
    """Duplicate finals happen. Merging one onto itself would double the
    question and change its wording."""
    detector = QuestionDetector()
    first = detector.inspect("How would you scale this?", now=100.0)
    again = detector.inspect("How would you scale this?", now=100.5)

    assert first.accepted
    assert again.text.count("scale this") == 1, again.text


def test_out_of_order_finals_do_not_merge_backwards():
    """A monotonic-now guard: a final timestamped before the last accepted one
    must not be glued to it in the wrong order."""
    detector = QuestionDetector()
    detector.inspect("How would you scale this?", now=100.0)
    earlier = detector.inspect("and what about failures?", now=99.0)

    assert "scale this" not in earlier.text


@asyncio_test
async def test_a_self_correction_mid_turn_keeps_the_final_wording(monkeypatch):
    """Two fragments of one speech turn where the second revises the first.
    The turn that gets answered has to be the corrected one."""
    completer = CountingCompleter(reply(intent="query"))
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "How would you optimize the query", TranscriptSource.LOOPBACK,
            is_final=True, now=100.0,
        )
        await h.live.on_speech_start(2)
        await h.live.on_transcript(
            "actually, how would you optimize the entire pipeline?",
            TranscriptSource.LOOPBACK, is_final=True, now=100.6,
        )
        await h.settle()

        # "How would you optimize the query" is a complete question by
        # grammar, so it is answered rather than held -- nothing deterministic
        # can know a correction is coming, and the alternative is delaying
        # every complete question. What matters is that the correction
        # supersedes it: one surviving answer, carrying the revised wording
        # and still carrying the clause it revised.
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "both fragments produced a surviving answer"
        assert "entire pipeline" in prompts(h)[-1]
        assert "optimize the query" in prompts(h)[-1], "the first clause was lost"
    finally:
        h.dispose()
