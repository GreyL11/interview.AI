"""Production default: no history and no material -> no classifier round trip."""

import pytest

from app.core.config import settings
from tests.test_understanding_session import (
    CountingCompleter, harness_with, prompts, reply, say,
)

asyncio_test = pytest.mark.asyncio


@asyncio_test
async def test_first_question_skips_understanding_then_follow_up_uses_it(monkeypatch):
    monkeypatch.setattr(settings, "question_understanding_skip_without_context", True)
    completer = CountingCompleter(reply(), reply())
    h = harness_with(completer, monkeypatch)
    try:
        await say(h, "What is Azure Databricks?", now=100.0)
        assert completer.calls == 0, "nothing to relate the first question to"
        assert len(prompts(h)) == 1, "still answered"

        await say(h, "How does it compare to Synapse?", now=130.0)
        assert completer.calls == 1, "with history the classifier is consulted"
        assert len(prompts(h)) == 2
    finally:
        h.dispose()


@asyncio_test
async def test_pasted_material_still_uses_understanding(monkeypatch):
    monkeypatch.setattr(settings, "question_understanding_skip_without_context", True)
    completer = CountingCompleter(reply(needs_attachments=True))
    h = harness_with(completer, monkeypatch)
    try:
        await h.live.on_context_attached(kind="text", content="SELECT 1;", now=100.0)
        await say(h, "What does this query return?", now=101.0)
        assert completer.calls == 1
    finally:
        h.dispose()
