"""Interviewer-pasted material joined to the spoken question.

An interviewer pastes a table and asks about it, or asks and then pastes.
Either way it is one interview turn, and the model must receive the spoken
words and the pasted bytes together, in arrival order, unaltered.

Every case here asserts three things, because any one alone is easy to
satisfy wrongly:

    the exact pasted content reaches the prompt
    the exact spoken wording reaches the prompt
    the number of provider calls is what it should be

Deterministic: attachment and speech timestamps are passed explicitly, never
slept on.
"""

import pytest

from app.core.config import settings
from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.attachments import (
    AttachmentBuffer,
    AttachmentError,
    RejectReason,
    build_attachment,
)
from app.realtime.events import EventType
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayHarness

asyncio_test = pytest.mark.asyncio

TABLE = "order_id | amount | status\n1        | 10.00  | paid\n2        | NULL   | pending"
QUERY = "SELECT status, SUM(amount)\nFROM orders\nGROUP BY status;"
SNIPPET = "def total(rows):\n    return sum(r['amount'] for r in rows)"


@pytest.fixture
def harness(monkeypatch):
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    yield h
    h.dispose()


def prompts(harness) -> list[str]:
    return harness.llm.prompts


def only_prompt(harness) -> str:
    assert len(harness.llm.prompts) == 1, [
        p[-200:] for p in harness.llm.prompts
    ]
    return harness.llm.prompts[0]


async def say(harness, text, now, settle=True):
    await harness.live.on_speech_start(1)
    await harness.live.on_transcript(
        text, TranscriptSource.LOOPBACK, is_final=True, now=now
    )
    if settle:
        await harness.settle()


async def paste(harness, kind, content, now, name="", settle=True):
    """`settle=False` leaves a held or streaming turn alone, which is what the
    mid-flight and mid-accumulation cases need -- settling would run the hold
    out and answer the fragment before the next event arrives."""
    await harness.live.on_context_attached(
        kind=kind, content=content, name=name, now=now
    )
    if settle:
        await harness.settle()


# ===================================================== the ordering matrix


@asyncio_test
async def test_paste_before_question_attaches_to_it(harness):
    await paste(harness, "table", TABLE, now=100.0)
    await say(harness, "What is wrong with this data?", now=101.0)

    prompt = only_prompt(harness)
    assert TABLE in prompt, "pasted table did not reach the model verbatim"
    assert "What is wrong with this data?" in prompt


@asyncio_test
async def test_question_before_paste_reasks_once_with_the_material(monkeypatch):
    """Voice first, paste second, while the answer is still streaming. The
    turn must end up carrying the material, and the question must not be
    answered as two independent turns."""
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0.02),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    try:
        await say(h, "What is wrong with this data?", now=100.0, settle=False)
        await paste(h, "table", TABLE, now=100.5, settle=False)
        await h.settle()

        # The surviving answer carries both halves of the turn.
        assert TABLE in prompts(h)[-1], "re-ask did not carry the material"
        assert "What is wrong with this data?" in prompts(h)[-1]
        # Exactly one answer survives -- the re-ask superseded the in-flight
        # one rather than opening a second turn for the same question.
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "only the attachment-carrying answer may survive"
        # At most one wasted call. Whether the first answer reached the
        # provider at all before being superseded is a genuine race (it
        # depends how far into retrieval it got), so this bounds it rather
        # than pinning it.
        assert len(prompts(h)) <= 2, prompts(h)
    finally:
        h.dispose()


@asyncio_test
async def test_voice_fragment_then_paste_then_continuation_is_one_call(harness):
    """Paste during accumulation: the fragments keep assembling and the
    material rides along, with no extra provider call for the paste."""
    # "Can you explain" is provably incomplete, so the turn is still
    # accumulating when the paste lands.
    await harness.live.on_speech_start(1)
    await harness.live.on_transcript(
        "Can you explain", TranscriptSource.LOOPBACK, is_final=True, now=100.0,
    )
    await paste(harness, "sql", QUERY, now=100.3, settle=False)
    assert prompts(harness) == [], "a paste must not answer a half-spoken question"

    await harness.live.on_speech_start(2)
    await harness.live.on_transcript(
        "why this query is slow?", TranscriptSource.LOOPBACK,
        is_final=True, now=100.6,
    )
    await harness.settle()

    prompt = only_prompt(harness)
    assert QUERY in prompt
    assert "Can you explain why this query is slow?" in prompt


@asyncio_test
async def test_multiple_pastes_all_attach_in_arrival_order(harness):
    await paste(harness, "table", TABLE, now=100.0)
    await paste(harness, "sql", QUERY, now=100.5)
    await paste(harness, "code", SNIPPET, now=101.0)
    await say(harness, "Which of these is the bug?", now=101.5)

    prompt = only_prompt(harness)
    for content in (TABLE, QUERY, SNIPPET):
        assert content in prompt, f"missing attachment: {content[:30]!r}"
    # Arrival order preserved: schema, then query, then code.
    assert prompt.index(TABLE) < prompt.index(QUERY) < prompt.index(SNIPPET)


@asyncio_test
async def test_an_unrelated_old_paste_does_not_attach(harness):
    """Outside the window the material belongs to nothing. A stale paste that
    kept attaching would contaminate every later question in the session."""
    await paste(harness, "table", TABLE, now=100.0)
    stale_by = settings.context_attachment_window_ms / 1000 + 5
    await say(harness, "What is a covering index?", now=100.0 + stale_by)

    prompt = only_prompt(harness)
    assert TABLE not in prompt
    assert "What is a covering index?" in prompt


@asyncio_test
async def test_a_followup_still_sees_the_attached_material(harness):
    await paste(harness, "table", TABLE, now=100.0)
    await say(harness, "What is wrong with this data?", now=100.5)
    assert TABLE in only_prompt(harness)

    await say(harness, "Why?", now=104.0)

    assert len(prompts(harness)) == 2
    assert TABLE in prompts(harness)[-1], "follow-up lost the table it refers to"
    assert "Why?" in prompts(harness)[-1]


@asyncio_test
async def test_a_correction_after_a_paste_does_not_duplicate_the_material(monkeypatch):
    """The corrected question is asked with the material once, not twice."""
    h = ReplayHarness(
        llm=SlowStreamingLLM(chunk_delay=0.02),
        memory_factory=SqliteSessionMemory,
        monkeypatch=monkeypatch,
    )
    try:
        await paste(h, "sql", QUERY, now=100.0, settle=False)
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            "How would you optimise this for a thousand rows?",
            TranscriptSource.LOOPBACK, is_final=True, now=100.4,
        )
        await h.live.on_speech_start(2)
        await h.live.on_transcript(
            "Actually, assume ten million rows.",
            TranscriptSource.LOOPBACK, is_final=True, now=100.9,
        )
        await h.settle()

        final = prompts(h)[-1]
        assert final.count(QUERY) == 1, "attachment duplicated into the prompt"
        assert "ten million rows" in final
        completed = h.result.of(EventType.ANSWER_COMPLETED)
        assert len(completed) == 1, "the corrected turn must be the only survivor"
    finally:
        h.dispose()


@asyncio_test
async def test_a_paste_after_the_answer_completed_does_not_reask(harness):
    """The interviewer already has the answer. Re-asking there would answer
    the same question twice; the material waits for the next question."""
    await say(harness, "What is a covering index?", now=100.0)
    assert len(prompts(harness)) == 1

    await paste(harness, "table", TABLE, now=100.5)
    assert len(prompts(harness)) == 1, "a post-answer paste must not re-ask"

    await say(harness, "What does this tell you?", now=101.0)
    assert len(prompts(harness)) == 2
    assert TABLE in prompts(harness)[-1], "pending material lost for the next question"


@asyncio_test
async def test_material_does_not_leak_onto_a_later_unrelated_question(harness):
    """Binding is consuming: once a turn has taken the material, the next
    question starts clean."""
    await paste(harness, "table", TABLE, now=100.0)
    await say(harness, "What is wrong with this data?", now=100.5)
    assert TABLE in prompts(harness)[0]

    await say(harness, "Explain database normalisation.", now=140.0)

    assert len(prompts(harness)) == 2
    assert TABLE not in prompts(harness)[-1], "stale material leaked into a new turn"


# ================================================ validation and exactness


def test_pasted_content_is_preserved_byte_for_byte():
    """Interior whitespace is content: a table's alignment and a snippet's
    indentation are load-bearing."""
    attachment = build_attachment("code", SNIPPET, now=100.0)
    assert attachment.content == SNIPPET
    assert SNIPPET in attachment.as_context()


def test_an_oversized_attachment_is_rejected_not_truncated():
    """A partial table looks complete to the model, which is worse than none."""
    huge = "x" * (settings.context_attachment_max_chars + 1)
    with pytest.raises(AttachmentError) as caught:
        build_attachment("text", huge, now=100.0)
    assert caught.value.reason is RejectReason.TOO_LARGE


def test_an_empty_attachment_is_rejected():
    for blank in ("", "   ", "\n\n"):
        with pytest.raises(AttachmentError) as caught:
            build_attachment("text", blank, now=100.0)
        assert caught.value.reason is RejectReason.EMPTY


def test_an_unknown_kind_degrades_to_text_rather_than_failing():
    attachment = build_attachment("spreadsheet", TABLE, now=100.0)
    assert attachment.content == TABLE


def test_sql_is_fenced_as_sql_so_the_model_knows_what_it_is():
    assert "```sql" in build_attachment("sql", QUERY, now=100.0).as_context()


def test_the_item_cap_evicts_oldest_first():
    buffer = AttachmentBuffer()
    for i in range(settings.context_attachment_max_items + 2):
        buffer.add(build_attachment("text", f"item {i}", now=100.0 + i))

    kept = [a.content for a in buffer.pending]
    assert len(kept) == settings.context_attachment_max_items
    assert "item 0" not in kept, "oldest should be evicted first"
    assert f"item {settings.context_attachment_max_items + 1}" in kept


def test_expiring_drops_only_stale_pending_material():
    buffer = AttachmentBuffer()
    buffer.add(build_attachment("text", "old", now=100.0))
    buffer.add(build_attachment("text", "fresh", now=140.0))

    dropped = buffer.expire(now=141.0)
    assert dropped == 1
    assert [a.content for a in buffer.pending] == ["fresh"]


@asyncio_test
async def test_a_rejected_attachment_reports_a_reason_and_changes_nothing(harness):
    await harness.live.on_context_attached(
        kind="text", content="x" * (settings.context_attachment_max_chars + 1),
        now=100.0,
    )
    await harness.settle()

    rejected = harness.result.of(EventType.CONTEXT_REJECTED)
    assert len(rejected) == 1
    assert rejected[0].data["reason"] == RejectReason.TOO_LARGE.value
    assert not harness.live._attachments.has_pending

    await say(harness, "What is a covering index?", now=100.5)
    assert len(prompts(harness)) == 1


@asyncio_test
async def test_an_accepted_attachment_is_acknowledged_without_echoing_content(harness):
    """The client already has the bytes; the event carries metadata only, so
    the content is not duplicated onto the socket or into the replay buffer."""
    await paste(harness, "table", TABLE, now=100.0, name="orders.csv")

    attached = harness.result.of(EventType.CONTEXT_ATTACHED)
    assert len(attached) == 1
    assert attached[0].data["kind"] == "table"
    assert attached[0].data["name"] == "orders.csv"
    assert attached[0].data["chars"] == len(TABLE)
    assert TABLE not in str(attached[0].data)


@asyncio_test
async def test_a_paste_alone_never_asks_anything(harness):
    """Material with no question is not a question."""
    await paste(harness, "table", TABLE, now=100.0)

    assert prompts(harness) == []
    assert not harness.result.of(EventType.QUESTION_DETECTED)
