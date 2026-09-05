"""Attachment ownership across a turn: deterministic window, then meaning.

The failure this file pins down came from real use. A candidate pasted a
71-character problem statement, the interviewer read it out and asked "Tell me
the answer for this?", and the answer was a request for clarification -- because
the paste had gone stale by a clock the interviewer knew nothing about, the
material was silently left out of the prompt, and "this" referred to nothing.

Two things are asserted throughout, because either alone is easy to satisfy
wrongly:

    the exact interviewer transcript reaches the prompt, unchanged
    the exact pasted content reaches the *same* prompt

and the phrasing tests exist to fail loudly if anyone ever "fixes" this with a
list of words like "this". Nothing in the production path reads the question
text to make this decision; the classifier's `needs_attachments` does, and the
gate is that flag plus the buffer's lifecycle.

Deterministic: every attachment and speech timestamp is passed explicitly, so
a stale interval is a number rather than a sleep.
"""

import pytest

from app.core.config import settings
from app.realtime.events import EventType
from app.realtime.question_understanding import UnderstandingSource
from app.sessions.schemas import TranscriptSource
from tests.test_understanding_session import CountingCompleter, harness_with, reply

asyncio_test = pytest.mark.asyncio

#: Exactly 71 characters, matching the reported "Text · 71 chars" chip.
PASTED = "Given an array of integers, return the indices of the two summing to K."
assert len(PASTED) == 71, "the reported repro was a 71-character paste"

ASKED = "Tell me the answer for this?"

#: Comfortably past any window this system would sanely be configured with, so
#: these cases can only pass through the semantic path.
STALE = settings.context_attachment_window_ms / 1000 + 30


async def say(h, text, now):
    await h.live.on_speech_start(1)
    await h.live.on_transcript(text, TranscriptSource.LOOPBACK, is_final=True, now=now)
    await h.settle()


async def paste(h, content, now, kind="text"):
    await h.live.on_context_attached(kind=kind, content=content, now=now)
    await h.settle()


def prompts(h) -> list[str]:
    return h.llm.prompts


# =========================================== 1. the exact reported regression


@pytest.mark.parametrize("gap", [1.0, 5.0, 12.0, 30.0, 90.0])
@asyncio_test
async def test_a_pasted_problem_reaches_the_answer_however_long_the_pause(
    monkeypatch, gap
):
    """The reported failure, swept across the pause that caused it.

    12s and 30s used to drop the material outright (the window was 10s); 90s
    is past even the widened window and is recovered by meaning instead. All
    five must look identical from the outside, because to the interviewer they
    are the same act: paste a problem, then ask about it.
    """
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + gap)

        assert len(prompts(h)) == 1, "exactly one answer request per question"
        assert completer.calls == 1, "exactly one understanding call per question"
        prompt = prompts(h)[0]
        assert ASKED in prompt, "the interviewer's exact words must survive"
        assert PASTED in prompt, f"pasted material lost at a {gap}s pause"
    finally:
        h.dispose()


# ================================================= 2. phrasing independence


@pytest.mark.parametrize(
    "text",
    [
        "Tell me the answer for this?",
        "Answer this?",
        "Explain this?",
        "What about this?",
        "Tell me the result for this?",
        "Can you solve this?",
    ],
)
@asyncio_test
async def test_any_referring_phrasing_resolves_against_stale_material(
    monkeypatch, text
):
    """Phrasing must not matter, at an interval where only meaning can save it.

    If this file ever passes while `test_...phrasing_is_never_inspected` below
    fails, someone has added a keyword rule.
    """
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, text, now=100.0 + STALE)

        assert len(prompts(h)) == 1
        assert text in prompts(h)[0], "exact transcript changed"
        assert PASTED in prompts(h)[0], f"{text!r} did not resolve its reference"
    finally:
        h.dispose()


@asyncio_test
async def test_phrasing_is_never_inspected_for_this_decision(monkeypatch):
    """The mirror image: the *same* referring phrasing gets nothing when the
    classifier says the question does not refer to provided material.

    Same words, opposite outcome -- which is only possible if the decision is
    made on the classification and not on the words.
    """
    completer = CountingCompleter(reply(needs_attachments=False))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        assert ASKED in prompts(h)[0]
        assert PASTED not in prompts(h)[0]
    finally:
        h.dispose()


# ================================================== 3/4. the semantic claim


@asyncio_test
async def test_stale_material_is_claimed_when_the_turn_refers_to_it(monkeypatch):
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        assert len(h.live._attachments.pending) == 1
        await say(h, ASKED, now=100.0 + STALE)

        assert PASTED in prompts(h)[0]
        # pending -> bound, exactly once.
        assert h.live._attachments.pending == []
        assert [a.content for a in h.live._attachments.bound] == [PASTED]
    finally:
        h.dispose()


@asyncio_test
async def test_declined_material_stays_pending_for_a_later_eligible_turn(
    monkeypatch,
):
    """Declining is not discarding.

    The interviewer pasted something and then asked about something else; that
    is not a reason to throw the paste away, because the question about it may
    still be coming.
    """
    completer = CountingCompleter(
        reply(needs_attachments=False),
        reply(needs_attachments=True),
    )
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)

        await say(h, "What is a covering index?", now=100.0 + STALE)
        assert PASTED not in prompts(h)[0], "material attached to an unrelated turn"
        assert len(h.live._attachments.pending) == 1, "material discarded, not kept"

        await say(h, ASKED, now=100.0 + STALE + 20)
        assert PASTED in prompts(h)[1], "the eligible turn could not claim it"
        assert h.live._attachments.pending == []
    finally:
        h.dispose()


# ================================= 5. the classifier is not load-bearing


@asyncio_test
async def test_a_broken_classifier_does_not_attach_stale_material(monkeypatch):
    """`deterministic_fallback` sets needs_attachments from "does material
    exist", which is true whenever anything is pending. Honouring that on the
    failure path would attach stale material to every question whenever the
    classifier was down -- so the claim requires an LLM-sourced reading."""
    completer = CountingCompleter(error=RuntimeError("provider down"))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        assert ASKED in prompts(h)[0], "the question must still be answered"
        assert PASTED not in prompts(h)[0]
        assert len(h.live._attachments.pending) == 1, "material must not be consumed"
    finally:
        h.dispose()


@asyncio_test
async def test_a_broken_classifier_keeps_fresh_binding_working(monkeypatch):
    """The deterministic path is unchanged and owes nothing to the classifier."""
    completer = CountingCompleter(error=RuntimeError("provider down"))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=101.0)

        assert PASTED in prompts(h)[0]
        assert h.live._attachments.pending == []
    finally:
        h.dispose()


@asyncio_test
async def test_understanding_disabled_does_not_attach_stale_material(monkeypatch):
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    monkeypatch.setattr(settings, "question_understanding_enabled", False)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        assert completer.calls == 0, "the classifier must not be consulted at all"
        assert PASTED not in prompts(h)[0]
        assert len(h.live._attachments.pending) == 1
    finally:
        h.dispose()


# ============================================ 6. metadata, never content


@asyncio_test
async def test_the_classifier_sees_pending_metadata_and_never_the_content(
    monkeypatch,
):
    """The one thing that makes the semantic claim possible at all: the
    classifier is told material exists that this turn did not bind. It is told
    the kind and the size, and never a byte of the material itself -- a 20,000
    character schema in a second prompt would cost realtime latency for no
    gain, and puts pasted content somewhere it does not need to be."""
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        assert completer.calls == 1
        understanding_prompt = completer.prompts[0]
        assert PASTED not in understanding_prompt, "raw pasted content leaked"
        assert f"text, {len(PASTED)} characters" in understanding_prompt, (
            "no attachment metadata reached the classifier"
        )
        # And the answer prompt is the only place the bytes appear.
        assert PASTED in prompts(h)[0]
    finally:
        h.dispose()


@asyncio_test
async def test_bound_and_pending_metadata_both_reach_the_classifier(monkeypatch):
    """Two pastes, one fresh enough to bind deterministically and one not.

    The classifier must be told about both, or it judges the question against
    half of what the interviewer provided. Intervals are fractions of the
    window rather than constants: the point is which side of it each paste
    falls on, and the second paste has to land early enough that it does not
    expire the first on arrival.
    """
    window = settings.context_attachment_window_ms / 1000
    old, recent = "x" * 40, "y" * 25
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, old, now=100.0)
        await paste(h, recent, now=100.0 + window * 0.9)
        await say(h, ASKED, now=100.0 + window * 1.1)

        understanding_prompt = completer.prompts[0]
        assert "text, 40 characters" in understanding_prompt, "pending material hidden"
        assert "text, 25 characters" in understanding_prompt, "bound material hidden"
        assert old not in understanding_prompt and recent not in understanding_prompt

        # The turn still carries only what it bound deterministically. The
        # claim is a rescue for a turn that got *nothing*, never a widening of
        # one that succeeded -- so a question that already has its material
        # cannot have older material added to it by the classifier.
        prompt = prompts(h)[0]
        assert recent in prompt
        assert old not in prompt, "a successful fresh binding was widened"
        assert [a.content for a in h.live._attachments.pending] == [old]
    finally:
        h.dispose()


# ==================================================== 7. follow-up binding


@asyncio_test
async def test_a_one_word_followup_claims_material_pasted_since(monkeypatch):
    """"Why?" after a paste is a question about the paste.

    This used to take `carry_forward()` *instead of* binding, so a paste made
    between a question and its follow-up was stranded: the follow-up carried
    only what the previous turn had, which was nothing.
    """
    completer = CountingCompleter(reply(), reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is a hash map?", now=100.0)
        await paste(h, PASTED, now=101.0)
        await say(h, "Why?", now=102.0)

        assert len(prompts(h)) == 2
        assert PASTED in prompts(h)[1], "the follow-up did not see the paste"
        assert h.live._attachments.pending == [], "material left unclaimed"
    finally:
        h.dispose()


@asyncio_test
async def test_a_followup_keeps_the_material_the_thread_already_had(monkeypatch):
    """The other half of additive: binding on a follow-up must not clear what
    the thread is already carrying, and must not duplicate it either."""
    completer = CountingCompleter(reply(), reply(relationship="follow_up"))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, "What is wrong with this?", now=101.0)
        assert PASTED in prompts(h)[0]

        await say(h, "Why?", now=102.0)
        assert prompts(h)[1].count(PASTED) == 1, "material duplicated in the prompt"
    finally:
        h.dispose()


# ================================================ 8/9. exactly one owner


@asyncio_test
async def test_material_consumed_by_a_turn_does_not_reach_a_later_one(monkeypatch):
    """Binding is consuming. Unchanged by any of this, and the reason a stale
    paste cannot become a permanent contaminant."""
    completer = CountingCompleter(reply(needs_attachments=True), reply())
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.5)
        assert PASTED in prompts(h)[0]

        await say(h, "Explain database normalisation.", now=140.0)
        assert len(prompts(h)) == 2
        assert PASTED not in prompts(h)[1], "material leaked into a later turn"
    finally:
        h.dispose()


@asyncio_test
async def test_two_eligible_turns_cannot_both_claim_the_same_material(monkeypatch):
    """Both turns say they refer to provided material; only the first can have
    it, because the claim moves it out of pending."""
    completer = CountingCompleter(
        reply(needs_attachments=True), reply(needs_attachments=True)
    )
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)
        await say(h, "And what about this one?", now=100.0 + STALE + 20)

        assert len(prompts(h)) == 2
        assert PASTED in prompts(h)[0]
        assert PASTED not in prompts(h)[1], "the same paste was claimed twice"
    finally:
        h.dispose()


@asyncio_test
async def test_a_superseded_turn_cannot_claim_material(monkeypatch):
    """A turn that is no longer current must not consume material the turn that
    replaced it may need."""
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        # Pretend the turn under construction has already been replaced.
        h.live._current_turn_id = 999
        claimed = h.live._claim_referenced(1, [], _llm_understanding())

        assert claimed == []
        assert len(h.live._attachments.pending) == 1
    finally:
        h.dispose()


def _llm_understanding():
    from app.realtime.question_understanding import Understanding

    return Understanding(
        exact_question=ASKED,
        needs_attachments=True,
        source=UnderstandingSource.LLM,
    )


# ================================================= the exactness invariant


@asyncio_test
async def test_a_claim_does_not_alter_the_question_or_the_material(monkeypatch):
    awkward = "  SELECT  1\n\tFROM\tdual  \n"
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, awkward, now=100.0, kind="sql")
        await say(h, ASKED, now=100.0 + STALE)

        prompt = prompts(h)[0]
        # Only the surrounding newlines go, exactly as `build_attachment` does
        # for a fresh paste. Interior whitespace is content.
        assert awkward.strip("\n\r") in prompt
        assert ASKED in prompt
        detected = h.result.of(EventType.QUESTION_DETECTED)
        assert [e.data["question"] for e in detected] == [ASKED]
    finally:
        h.dispose()


# ==================================================== the missing log line


class MetricSpy:
    """Captures log_metric calls. Patched into the *importing* module, since
    both call sites do `from app.core.metrics import log_metric`."""

    def __init__(self, monkeypatch, *modules: str) -> None:
        self.calls: list[tuple[str, dict]] = []
        for module in modules:
            monkeypatch.setattr(f"{module}.log_metric", self)

    def __call__(self, name: str, **fields) -> None:
        self.calls.append((name, fields))

    def of(self, name: str) -> list[dict]:
        return [fields for called, fields in self.calls if called == name]


@asyncio_test
async def test_material_left_behind_by_the_window_is_reported(monkeypatch):
    """The silence that made the original failure undiagnosable: material
    outside the window is skipped rather than dropped, so nothing expired and
    nothing was logged."""
    spy = MetricSpy(monkeypatch, "app.realtime.attachments", "app.realtime.session")
    completer = CountingCompleter(reply(needs_attachments=False))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        window = [e for e in spy.of("attachment_unbound") if e["reason"] == "outside_window"]
        assert len(window) == 1, spy.of("attachment_unbound")
        assert window[0]["count"] == 1
        assert window[0]["age_ms"] == int(STALE * 1000)
    finally:
        h.dispose()


@asyncio_test
async def test_declining_to_claim_is_reported_with_its_reason(monkeypatch):
    spy = MetricSpy(monkeypatch, "app.realtime.attachments", "app.realtime.session")
    completer = CountingCompleter(reply(needs_attachments=False))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        reasons = [e["reason"] for e in spy.of("attachment_unbound")]
        assert "not_referenced" in reasons, reasons
    finally:
        h.dispose()


@asyncio_test
async def test_an_ordinary_turn_logs_no_attachment_noise(monkeypatch):
    """No material in play must mean no lines at all -- this fires on every
    turn of every session, so it cannot become chatter."""
    spy = MetricSpy(monkeypatch, "app.realtime.attachments", "app.realtime.session")
    completer = CountingCompleter(reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is a covering index?", now=100.0)
        assert spy.of("attachment_unbound") == []
        assert spy.of("attachment_claimed") == []
    finally:
        h.dispose()


@asyncio_test
async def test_a_claim_is_reported_with_the_age_it_rescued(monkeypatch):
    spy = MetricSpy(monkeypatch, "app.realtime.attachments", "app.realtime.session")
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await paste(h, PASTED, now=100.0)
        await say(h, ASKED, now=100.0 + STALE)

        claimed = spy.of("attachment_claimed")
        assert len(claimed) == 1, claimed
        assert claimed[0]["count"] == 1
        assert claimed[0]["age_ms"] >= int(STALE * 1000)
    finally:
        h.dispose()
