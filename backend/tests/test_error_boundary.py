"""What a failed turn is allowed to tell the client, and what Settings dumps.

Two hardening findings, both about a boundary that was open rather than about
a leak anyone had observed.

**The WebSocket error event forwarded `str(exc)` verbatim.** Every real
provider client builds its message from a template in `app.llm.messages` and
discards the provider's own string, so there was no reachable leak -- but
nothing enforced that. A provider whose error text echoed the request back,
key included, or an unanticipated internal exception's repr, would have gone
straight to the client. The boundary now works from a whitelist: a sentence
this app built is shown, anything else is replaced with the correct generic
sentence for its classified kind.

**`settings.model_dump()` carried API keys in plain text.** `__repr_args__`
closed repr/str, which was the leak someone was actually likely to cause;
serialisation was the other half and was open, so any future diagnostics
endpoint, support bundle or config echo would have carried every key.

The probe secret below is deliberately unmistakable: if it ever appears in a
failure message, the assertion that failed names exactly which surface let it
through.
"""

import json

import pytest

from app.core.config import settings
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.messages import message_for
from app.realtime.events import EventType
from app.realtime.session import _client_safe_message
from app.sessions.schemas import TranscriptSource
from tests.replay_harness import ReplayHarness

asyncio_test = pytest.mark.asyncio

SECRET = "SUPER_SECRET_TEST_KEY_123456789"
QUESTION = "How would you design a rate limiter?"


class EchoingLLM(LLMClient):
    """A provider that puts the request back in its error message, key and
    all. Not how any client in this app behaves -- which is the point: the
    boundary must not depend on every client behaving."""

    provider_name = "echoing"

    def __init__(self, kind: LLMErrorKind = LLMErrorKind.AUTH) -> None:
        self.kind = kind

    @property
    def model_name(self) -> str:
        return "echoing-model"

    async def generate_answer(self, prompt: str):
        raise LLMError(f"401 from https://api.example/v1 key={SECRET}", kind=self.kind)

    async def stream_answer(self, prompt: str):
        raise LLMError(f"401 from https://api.example/v1 key={SECRET}", kind=self.kind)
        yield ""


class ExplodingLLM(LLMClient):
    """An exception shape the pipeline never anticipated."""

    provider_name = "exploding"

    async def generate_answer(self, prompt: str):
        raise RuntimeError(f"internal state dump: api_key={SECRET}")

    async def stream_answer(self, prompt: str):
        raise RuntimeError(f"internal state dump: api_key={SECRET}")
        yield ""


# ------------------------------------------------------ the sanitiser itself


def test_an_app_built_sentence_is_shown_as_written():
    original = message_for(LLMErrorKind.RATE_LIMIT, "Groq")
    assert _client_safe_message(LLMError(original, kind=LLMErrorKind.RATE_LIMIT)) == original


def test_a_model_name_survives_in_the_sentence():
    """MODEL_UNAVAILABLE interpolates configuration, so it cannot be matched
    by equality -- it still has to pass."""
    original = message_for(LLMErrorKind.MODEL_UNAVAILABLE, "OmniRouter", "gpt-5.6-luna")
    assert _client_safe_message(
        LLMError(original, kind=LLMErrorKind.MODEL_UNAVAILABLE)
    ) == original


@pytest.mark.parametrize("kind", list(LLMErrorKind))
def test_an_unknown_sentence_is_replaced_by_its_own_kind(kind):
    """Replaced, not scrubbed: the reader still learns what went wrong and
    what to do, from the classification the provider already made."""
    safe = _client_safe_message(LLMError(f"raw provider text {SECRET}", kind=kind))
    assert SECRET not in safe
    assert safe == message_for(kind, "the AI provider")


def test_an_unexpected_exception_says_nothing_about_itself():
    safe = _client_safe_message(RuntimeError(f"stack with {SECRET}"))
    assert SECRET not in safe
    assert "went wrong" in safe


def test_a_validation_failure_gets_its_own_sentence():
    from app.intelligence.answer_validator import AnswerValidationError

    safe = _client_safe_message(AnswerValidationError("Answer summary is empty"))
    assert SECRET not in safe
    assert "unusable" in safe


# --------------------------------------------------------------- end to end


@asyncio_test
async def test_a_provider_that_echoes_the_key_does_not_reach_the_client(monkeypatch):
    harness = ReplayHarness(llm=EchoingLLM(), stabilization_ms=1, monkeypatch=monkeypatch)
    try:
        await harness.live.on_transcript(
            QUESTION, TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )
        await harness.settle()

        errors = harness.result.of(EventType.ANSWER_ERROR)
        assert errors, "the turn still has to reach a terminal state"
        wire = json.dumps([e.data for e in harness.result.events], default=str)
        assert SECRET not in wire
        assert "api.example" not in wire, "no URL from the provider either"
        # And the reader is still told something actionable.
        assert errors[0].data["message"] == message_for(
            LLMErrorKind.AUTH, "the AI provider"
        )
    finally:
        await harness.close()


@asyncio_test
async def test_an_internal_exception_does_not_reach_the_client(monkeypatch):
    harness = ReplayHarness(llm=ExplodingLLM(), stabilization_ms=1, monkeypatch=monkeypatch)
    try:
        await harness.live.on_transcript(
            QUESTION, TranscriptSource.LOOPBACK, is_final=True, now=100.0
        )
        await harness.settle()

        errors = harness.result.of(EventType.ANSWER_ERROR)
        assert errors
        assert errors[0].data["code"] == "InternalError"
        wire = json.dumps([e.data for e in harness.result.events], default=str)
        assert SECRET not in wire
        assert "internal state dump" not in wire
    finally:
        await harness.close()


@asyncio_test
async def test_the_session_survives_a_sanitised_failure(monkeypatch):
    """Sanitising must not swallow the failure: the next question still works
    and the broken turn is terminal."""
    harness = ReplayHarness(llm=EchoingLLM(), stabilization_ms=1, monkeypatch=monkeypatch)
    try:
        for now, text in ((100.0, QUESTION), (130.0, "What is a semaphore?")):
            await harness.live.on_transcript(
                text, TranscriptSource.LOOPBACK, is_final=True, now=now
            )
            await harness.settle()
        assert len(harness.result.of(EventType.QUESTION_DETECTED)) == 2
        assert len(harness.result.of(EventType.ANSWER_ERROR)) == 2
    finally:
        await harness.close()


# ------------------------------------------------------- settings serialisation


@pytest.fixture
def planted_secret():
    """Every `*_api_key` field set to the probe secret, restored afterwards."""
    fields = [name for name in type(settings).model_fields if name.endswith("_api_key")]
    original = {name: getattr(settings, name) for name in fields}
    for name in fields:
        setattr(settings, name, SECRET)
    yield fields
    for name, value in original.items():
        setattr(settings, name, value)


def test_repr_and_str_stay_redacted(planted_secret):
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)


def test_model_dump_is_redacted(planted_secret):
    dumped = settings.model_dump()
    assert SECRET not in json.dumps(dumped, default=str)
    for name in planted_secret:
        assert dumped[name] == "***", name


def test_model_dump_json_is_redacted(planted_secret):
    assert SECRET not in settings.model_dump_json()
    assert SECRET not in json.dumps(settings.model_dump(mode="json"), default=str)


def test_a_redacted_dump_still_shows_that_a_key_is_set(planted_secret):
    """An empty string would be indistinguishable from "no key configured",
    which is a different thing and a misleading thing to report."""
    settings.groq_api_key = ""
    dumped = settings.model_dump()
    assert dumped["groq_api_key"] == ""
    assert dumped["openai_api_key"] == "***"


def test_attribute_access_still_returns_the_real_key(planted_secret):
    """Reading a key and publishing one are different acts, and only the
    second is guarded. Everything that has to authenticate reads attributes."""
    from app.core.deps import api_key_for

    assert settings.groq_api_key == SECRET
    assert api_key_for("groq") == SECRET
    assert api_key_for("omnirouter") == SECRET


def test_the_credential_store_round_trip_is_unaffected(planted_secret):
    """`app.core.secrets` discovers fields by name and reads them by
    attribute; redaction must not break saving or loading a key.

    Deliberately does not install this store process-wide: conftest owns that
    for the whole run, and swapping it here leaked into a later test.
    """
    from app.core.secrets import InMemorySecretStore, secret_field_names

    assert "groq_api_key" in secret_field_names()
    assert "omnirouter_api_key" in secret_field_names()

    store = InMemorySecretStore()
    store.set("groq_api_key", SECRET)
    assert store.get("groq_api_key") == SECRET


def test_non_secret_settings_survive_serialisation(planted_secret):
    dumped = settings.model_dump()
    assert dumped["llm_provider"] == settings.llm_provider
    assert dumped["llm_model"] == settings.llm_model
    assert dumped["question_min_words"] == settings.question_min_words
