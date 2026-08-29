"""Provider routing, failover, cooldown and streaming-commit safety.

All providers here are fakes -- no API keys, no network. These assert the
router's *behavior*; they say nothing about how fast Gemini or Groq actually
are in production.
"""

import asyncio
import time

import pytest

from app.core.config import settings
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.router import RoutingLLMClient
from app.schemas.answer import Answer

pytestmark = pytest.mark.asyncio


def payload(summary: str = "A cache stores hot data.") -> str:
    return Answer(summary=summary, key_points=["a"], detailed_answer="d").model_dump_json()


class FakeProvider(LLMClient):
    """Scriptable provider. `fail_with` raises before any output;
    `fail_after_chars` raises once that much text has been yielded."""

    def __init__(
        self,
        name: str,
        text: str | None = None,
        fail_with: LLMError | None = None,
        fail_after_chars: int | None = None,
        chunk_size: int = 8,
        delay: float = 0.0,
    ) -> None:
        self.provider_name = name
        self._text = text if text is not None else payload()
        self._fail_with = fail_with
        self._fail_after_chars = fail_after_chars
        self._chunk_size = chunk_size
        self._delay = delay
        self.calls = 0
        self.cancelled = 0

    @property
    def model_name(self) -> str:
        return f"{self.provider_name}-model"

    async def generate_answer(self, prompt: str) -> Answer:
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        return Answer.model_validate_json(self._text)

    async def stream_answer(self, prompt: str):
        self.calls += 1
        if self._fail_with is not None and self._fail_after_chars is None:
            raise self._fail_with
        emitted = 0
        try:
            for i in range(0, len(self._text), self._chunk_size):
                if self._delay:
                    await asyncio.sleep(self._delay)
                chunk = self._text[i : i + self._chunk_size]
                if (
                    self._fail_after_chars is not None
                    and emitted >= self._fail_after_chars
                    and self._fail_with is not None
                ):
                    raise self._fail_with
                emitted += len(chunk)
                yield chunk
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


async def collect(client: LLMClient, prompt: str = "q") -> str:
    return "".join([chunk async for chunk in client.stream_answer(prompt)])


def router(*providers: FakeProvider, priority: str | None = None, monkeypatch=None):
    mapping = {p.provider_name: p for p in providers}
    if priority is not None and monkeypatch is not None:
        monkeypatch.setattr(settings, "llm_provider_priority", priority)
    return RoutingLLMClient(mapping)


@pytest.fixture(autouse=True)
def _stable_routing(monkeypatch):
    """Default to priority-only routing so selection tests aren't perturbed by
    the latency heuristic; the latency tests opt back in explicitly."""
    monkeypatch.setattr(settings, "llm_provider_priority", "groq,gemini")
    monkeypatch.setattr(settings, "llm_latency_aware_routing", False)


# ------------------------------------------------------------- selection


async def test_preferred_provider_is_used_and_fallback_never_called():
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    out = await collect(router(groq, gemini))

    assert "A cache stores hot data." in out
    assert groq.calls == 1
    assert gemini.calls == 0


async def test_configured_priority_is_respected(monkeypatch):
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    monkeypatch.setattr(settings, "llm_provider_priority", "gemini,groq")

    await collect(router(groq, gemini))

    assert gemini.calls == 1
    assert groq.calls == 0


async def test_a_provider_missing_from_priority_is_still_reachable(monkeypatch):
    """A configured provider must never be unreachable just because it was
    left out of the priority string."""
    groq = FakeProvider("groq", fail_with=LLMError("down", kind=LLMErrorKind.TRANSIENT))
    gemini = FakeProvider("gemini")
    monkeypatch.setattr(settings, "llm_provider_priority", "groq")

    out = await collect(router(groq, gemini))

    assert gemini.calls == 1
    assert "A cache stores hot data." in out


# ------------------------------------------------------------ 429 failover


async def test_rate_limit_before_first_token_falls_over_immediately():
    groq = FakeProvider(
        "groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT)
    )
    gemini = FakeProvider("gemini")
    client = router(groq, gemini)

    out = await collect(client)

    assert groq.calls == 1, "must not retry the rate-limited provider"
    assert gemini.calls == 1
    assert "A cache stores hot data." in out


async def test_rate_limit_puts_the_provider_in_cooldown_and_skips_it_next_time():
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini")
    client = router(groq, gemini)

    await collect(client)
    assert client._state["groq"].in_cooldown(time.monotonic())

    await collect(client)  # second question, still inside the cooldown

    assert groq.calls == 1, "cooled-down provider must be skipped entirely"
    assert gemini.calls == 2


async def test_cooldown_expiry_makes_the_provider_eligible_again(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_cooldown_seconds", 0.05)
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini")
    client = router(groq, gemini)

    await collect(client)
    groq._fail_with = None  # provider recovers
    await asyncio.sleep(0.06)
    await collect(client)

    assert groq.calls == 2, "should be retried once the cooldown lapsed"


async def test_retry_after_header_drives_the_cooldown_length():
    groq = FakeProvider(
        "groq",
        fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT, retry_after_seconds=45.0),
    )
    client = router(groq, FakeProvider("gemini"))

    await collect(client)

    remaining = client._state["groq"].cooldown_until - time.monotonic()
    assert 40 < remaining <= 45


async def test_retry_after_is_capped(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_max_cooldown_seconds", 10.0)
    groq = FakeProvider(
        "groq",
        fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT, retry_after_seconds=9999.0),
    )
    client = router(groq, FakeProvider("gemini"))

    await collect(client)

    assert client._state["groq"].cooldown_until - time.monotonic() <= 10.0


# --------------------------------------------------- other failure kinds


@pytest.mark.parametrize(
    "kind", [LLMErrorKind.TRANSIENT, LLMErrorKind.TIMEOUT, LLMErrorKind.UNKNOWN]
)
async def test_any_failure_before_first_token_falls_over(kind):
    groq = FakeProvider("groq", fail_with=LLMError("boom", kind=kind))
    gemini = FakeProvider("gemini")

    out = await collect(router(groq, gemini))

    assert gemini.calls == 1
    assert "A cache stores hot data." in out


async def test_auth_failure_cools_the_provider_down_hard(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_auth_cooldown_seconds", 300.0)
    groq = FakeProvider("groq", fail_with=LLMError("bad key", kind=LLMErrorKind.AUTH))
    client = router(groq, FakeProvider("gemini"))

    await collect(client)

    remaining = client._state["groq"].cooldown_until - time.monotonic()
    assert remaining > 200, "a bad key should not be retried every question"


async def test_transient_failures_cool_down_only_after_the_threshold(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_failure_threshold", 3)
    groq = FakeProvider("groq", fail_with=LLMError("5xx", kind=LLMErrorKind.TRANSIENT))
    client = router(groq, FakeProvider("gemini"))
    now = time.monotonic()

    await collect(client)
    assert not client._state["groq"].in_cooldown(now)
    await collect(client)
    assert not client._state["groq"].in_cooldown(now)
    await collect(client)

    assert client._state["groq"].in_cooldown(time.monotonic())


async def test_success_resets_the_failure_streak():
    groq = FakeProvider("groq", fail_with=LLMError("5xx", kind=LLMErrorKind.TRANSIENT))
    client = router(groq, FakeProvider("gemini"))

    await collect(client)
    assert client._state["groq"].consecutive_failures == 1
    groq._fail_with = None
    await collect(client)

    assert client._state["groq"].consecutive_failures == 0


async def test_both_providers_failing_raises_the_last_error():
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini", fail_with=LLMError("5xx", kind=LLMErrorKind.TRANSIENT))

    with pytest.raises(LLMError):
        await collect(router(groq, gemini))


# ------------------------------------------------- streaming commit safety


async def test_failure_after_first_visible_token_never_switches_providers():
    """The whole point of the withhold-then-commit design: once the user can
    see text, a second provider must not restart the answer."""
    groq = FakeProvider(
        "groq",
        fail_with=LLMError("died mid-stream", kind=LLMErrorKind.TRANSIENT),
        fail_after_chars=40,
    )
    gemini = FakeProvider("gemini")

    with pytest.raises(LLMError):
        await collect(router(groq, gemini))

    assert gemini.calls == 0, "must not restart the answer on another provider"


async def test_no_output_is_emitted_before_the_commit_point():
    """Nothing may reach the consumer while failover is still possible, or a
    failover would splice two answers into one corrupt buffer."""
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini", text=payload("Gemini answer."))

    out = await collect(router(groq, gemini))

    assert out == payload("Gemini answer."), "only the winning provider's bytes"


async def test_output_is_byte_identical_to_the_provider_stream():
    """Withholding must not corrupt or reorder the payload -- the existing
    parser has to see exactly what the provider sent."""
    text = payload("Indexes trade write speed for read speed.")
    out = await collect(router(FakeProvider("groq", text=text, chunk_size=3)))

    assert out == text


async def test_a_response_with_no_visible_summary_still_reaches_the_parser():
    """A malformed/empty payload must pass through so the existing validator
    raises its normal error rather than being silently swallowed."""
    groq = FakeProvider("groq", text='{"nope": 1}')

    out = await collect(router(groq, FakeProvider("gemini")))

    assert out == '{"nope": 1}'


# ------------------------------------------------------------ cancellation


async def test_cancellation_propagates_and_does_not_trigger_failover():
    groq = FakeProvider("groq", chunk_size=2, delay=0.02)
    gemini = FakeProvider("gemini")
    client = router(groq, gemini)

    async def consume():
        async for _ in client.stream_answer("q"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert groq.cancelled == 1
    assert gemini.calls == 0, "cancellation is not a provider failure"


async def test_cancellation_during_fallback_does_not_leak_output():
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini", chunk_size=2, delay=0.02)
    client = router(groq, gemini)
    seen = []

    async def consume():
        async for chunk in client.stream_answer("q"):
            seen.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert gemini.cancelled == 1


# -------------------------------------------------------- latency routing


async def test_latency_routing_needs_enough_samples_before_it_acts(monkeypatch):
    monkeypatch.setattr(settings, "llm_latency_aware_routing", True)
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    client = router(groq, gemini)

    await collect(client)

    assert groq.calls == 1, "priority wins until there is evidence"


async def test_a_decisively_faster_provider_can_outrank_priority(monkeypatch):
    monkeypatch.setattr(settings, "llm_latency_aware_routing", True)
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    client = router(groq, gemini)

    # Fill both windows: groq slow, gemini fast.
    for _ in range(5):
        client._state["groq"].first_token_ms.append(900)
        client._state["gemini"].first_token_ms.append(200)

    await collect(client)

    assert gemini.calls == 1, "20%+ faster on a full window should win"


async def test_a_marginally_faster_provider_does_not_displace_priority(monkeypatch):
    monkeypatch.setattr(settings, "llm_latency_aware_routing", True)
    groq, gemini = FakeProvider("groq"), FakeProvider("gemini")
    client = router(groq, gemini)

    for _ in range(5):
        client._state["groq"].first_token_ms.append(500)
        client._state["gemini"].first_token_ms.append(480)  # only 4% faster

    await collect(client)

    assert groq.calls == 1, "routing must not oscillate on noise"


async def test_first_token_latency_is_recorded_on_success():
    client = router(FakeProvider("groq"))

    await collect(client)

    assert len(client._state["groq"].first_token_ms) == 1


# ----------------------------------------------------------- observability


async def test_selection_and_fallback_are_observable(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.llm.router.log_metric", lambda e, **f: events.append((e, f))
    )
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini")

    await collect(router(groq, gemini))

    names = [e for e, _ in events]
    assert "llm_provider_selected" in names
    assert "llm_provider_fallback" in names
    assert "llm_provider_cooldown_started" in names
    assert "llm_provider_request_failed" in names
    assert "llm_provider_first_token" in names
    assert "llm_provider_stream_completed" in names

    fallback = next(f for e, f in events if e == "llm_provider_fallback")
    assert fallback["from_provider"] == "groq"
    assert fallback["to_provider"] == "gemini"
    assert fallback["reason"] == "rate_limit"

    cooldown = next(f for e, f in events if e == "llm_provider_cooldown_started")
    assert cooldown["provider"] == "groq"
    assert cooldown["reason"] == "rate_limit"

    selected = next(f for e, f in events if e == "llm_provider_selected")
    assert selected["provider"] == "groq"
    assert selected["model"] == "groq-model"


async def test_no_api_key_ever_reaches_the_metrics(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.llm.router.log_metric", lambda e, **f: events.append((e, f))
    )
    monkeypatch.setattr(settings, "groq_api_key", "sk-secret-groq-key")
    monkeypatch.setattr(settings, "gemini_api_key", "AIza-secret-gemini-key")

    await collect(router(FakeProvider("groq"), FakeProvider("gemini")))

    rendered = repr(events)
    assert "sk-secret-groq-key" not in rendered
    assert "AIza-secret-gemini-key" not in rendered


# ------------------------------------------------------ provider wiring


async def test_single_provider_configuration_works():
    """Groq-only and Gemini-only are both valid deployments."""
    for name in ("groq", "gemini"):
        only = FakeProvider(name)
        out = await collect(RoutingLLMClient({name: only}))
        assert only.calls == 1
        assert "A cache stores hot data." in out


async def test_generate_answer_also_falls_over():
    """The non-streaming POST /question path shares the routing rules."""
    groq = FakeProvider("groq", fail_with=LLMError("429", kind=LLMErrorKind.RATE_LIMIT))
    gemini = FakeProvider("gemini")

    answer = await router(groq, gemini).generate_answer("q")

    assert answer.summary == "A cache stores hot data."
    assert gemini.calls == 1


async def test_build_router_with_no_keys_still_builds(monkeypatch):
    """The app must start and support typed practice with no keys at all."""
    from app.llm.router import build_router

    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "")

    client = build_router()

    assert isinstance(client, RoutingLLMClient)
    assert "gemini" in client._providers, "keeps the existing not-configured message"


async def test_build_router_includes_only_configured_providers(monkeypatch):
    from app.llm.router import build_router

    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "groq_api_key", "k")

    client = build_router()

    assert set(client._providers) == {"groq"}


async def test_disabled_provider_is_not_wired(monkeypatch):
    from app.llm.router import build_router

    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "groq_enabled", False)

    client = build_router()

    assert set(client._providers) == {"gemini"}
