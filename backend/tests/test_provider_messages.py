"""The sentences a provider failure shows the person reading Settings.

Two things went wrong here, and both are about composition rather than about
any one string.

**A name that did not fit its slot.** Every template was written around a
proper noun ("The {name} request failed.") and then handed a name that
already carried its own article, producing "The the configured endpoint
request failed." and "No the configured endpoint API key is configured."
Fixing the one string would have left the next provider without a name of
its own to hit the same wall, so `app.llm.messages` now offers the three
grammatical forms English needs and each template asks for the one that fits.

**A classification thrown away.** `generate_answer` called `_ensure_client`
inside its own `except Exception`, so the NOT_CONFIGURED error that function
raises -- the one failure with a clear remedy -- was caught by the handler
and re-classified as UNKNOWN. "No API key is configured. Add one in Settings"
became "The request failed."

The parametrisation over every kind and every provider is the point: a
template is only correct if it is correct for all of them.
"""

import asyncio

import httpx
import pytest

from app.llm.base import LLMError, LLMErrorKind
from app.llm.messages import (
    GENERIC_PROVIDER_NAME,
    is_known_message,
    message_for,
    missing_endpoint_message,
    name_forms,
)
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.registry import PROVIDER_DISPLAY_NAMES, ProviderConfig, display_name

PROVIDERS = sorted(PROVIDER_DISPLAY_NAMES)

_ANSWER_JSON = (
    '{"summary": "s", "key_points": ["a"], "detailed_answer": "d", '
    '"approach": null, "code": null, "complexity": null, "edge_cases": null, '
    '"sections": null, "warnings": []}'
)


# --------------------------------------------------------------- composition


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("kind", list(LLMErrorKind))
def test_every_sentence_is_grammatical(provider, kind):
    """No doubled article, no doubled word, one sentence-initial capital.

    Checked structurally rather than against a stored copy of each string, so
    this keeps holding for a provider added later.
    """
    message = message_for(kind, display_name(provider), "gpt-5.6-luna")
    lowered = f" {message.lower()} "

    assert " the the " not in lowered, message
    assert " a a " not in lowered and " an an " not in lowered, message
    assert " no the " not in lowered, message
    assert " configured configured " not in lowered, message
    assert message[0].isupper(), message
    assert message.rstrip().endswith("."), message
    words = message.lower().replace(".", " ").replace(",", " ").split()
    for first, second in zip(words, words[1:]):
        assert first != second, f"doubled {first!r} in {message!r}"


@pytest.mark.parametrize("kind", list(LLMErrorKind))
def test_a_proper_noun_provider_keeps_its_exact_wording(kind):
    """The composition change must not have reworded the providers that
    already read correctly, so these are pinned literally."""
    expected = {
        LLMErrorKind.AUTH:
            "The Groq API key was rejected. Check it in Settings and save it again.",
        LLMErrorKind.NOT_CONFIGURED:
            "No Groq API key is configured. Add one in Settings to enable answers.",
        LLMErrorKind.MODEL_UNAVAILABLE:
            "The configured Groq model is not available to this account. "
            "Choose a different model in Settings.",
        LLMErrorKind.RATE_LIMIT:
            "Groq's rate limit was reached. Wait a moment and ask again.",
        LLMErrorKind.TIMEOUT: "Groq did not respond in time. Ask again.",
        LLMErrorKind.NETWORK:
            "Could not reach Groq. Check this machine's internet connection.",
        LLMErrorKind.SERVER: "Groq reported a server error. Try again shortly.",
        LLMErrorKind.MALFORMED: "Groq returned a response this app could not read.",
        LLMErrorKind.UNKNOWN: "The Groq request failed.",
    }
    assert message_for(kind, "Groq") == expected[kind]


def test_a_definite_provider_name_reads_correctly():
    """The case that was broken, spelled out."""
    name = display_name("openai_compatible")
    assert name.startswith("the "), "the fixture for this test is the name itself"
    assert message_for(LLMErrorKind.UNKNOWN, name) == (
        "The OpenAI-compatible endpoint request failed."
    )
    assert message_for(LLMErrorKind.NOT_CONFIGURED, name) == (
        "No OpenAI-compatible endpoint API key is configured. "
        "Add one in Settings to enable answers."
    )
    assert message_for(LLMErrorKind.RATE_LIMIT, name) == (
        "The OpenAI-compatible endpoint's rate limit was reached. "
        "Wait a moment and ask again."
    )
    assert message_for(LLMErrorKind.NETWORK, name) == (
        "Could not reach the OpenAI-compatible endpoint. "
        "Check this machine's internet connection."
    )


@pytest.mark.parametrize(
    ("name", "bare", "mid", "initial"),
    [
        ("Groq", "Groq", "Groq", "Groq"),
        ("OmniRouter", "OmniRouter", "OmniRouter", "OmniRouter"),
        ("the gateway", "gateway", "the gateway", "The gateway"),
    ],
)
def test_name_forms(name, bare, mid, initial):
    forms = name_forms(name)
    assert forms == {"bare": bare, "name": mid, "Name": initial}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_model_is_named_when_it_is_known(provider):
    message = message_for(
        LLMErrorKind.MODEL_UNAVAILABLE, display_name(provider), "gpt-5.6-terra"
    )
    assert "'gpt-5.6-terra'" in message
    assert "configured" not in message, "the model is named, so it is not 'the configured'"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("kind", list(LLMErrorKind))
def test_every_sentence_is_on_the_safe_whitelist(provider, kind):
    """`app.realtime.session` shows a message only if it was built here. A
    template this module can produce but the whitelist does not recognise
    would silently degrade a real error to a generic one."""
    assert is_known_message(message_for(kind, display_name(provider)))
    assert is_known_message(
        message_for(kind, display_name(provider), "some-model-name")
    )


def test_the_whitelist_rejects_anything_else():
    assert not is_known_message("provider echoed SUPER_SECRET_TEST_KEY_123456789")
    assert not is_known_message("")
    assert not is_known_message("KeyError: 'choices'")
    assert is_known_message(message_for(LLMErrorKind.AUTH, GENERIC_PROVIDER_NAME))
    assert is_known_message(missing_endpoint_message(display_name("omnirouter")))


# ---------------------------------------------------- NOT_CONFIGURED survives


def _unusable(**overrides) -> OpenAICompatibleClient:
    config = ProviderConfig(**{
        "provider": "openai_compatible",
        "model": "some-model",
        "api_key": "TEST_KEY_DO_NOT_LEAK",
        "base_url": "https://gateway.example.invalid/v1",
        **overrides,
    })
    return OpenAICompatibleClient(config)


@pytest.mark.asyncio
async def test_generate_answer_without_a_key_reports_not_configured():
    with pytest.raises(LLMError) as raised:
        await _unusable(api_key="").generate_answer("prompt")
    assert raised.value.kind is LLMErrorKind.NOT_CONFIGURED
    assert "API key is configured" in str(raised.value)


@pytest.mark.asyncio
async def test_stream_answer_without_a_key_reports_not_configured():
    client = _unusable(api_key="")
    with pytest.raises(LLMError) as raised:
        async for _ in client.stream_answer("prompt"):
            pass
    assert raised.value.kind is LLMErrorKind.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_complete_json_without_a_key_reports_not_configured():
    with pytest.raises(LLMError) as raised:
        await _unusable(api_key="").complete_json("prompt")
    assert raised.value.kind is LLMErrorKind.NOT_CONFIGURED


@pytest.mark.asyncio
async def test_generate_answer_without_a_base_url_reports_not_configured():
    """A missing endpoint and a missing key are both NOT_CONFIGURED, and the
    sentence has to send the reader to the right field."""
    with pytest.raises(LLMError) as raised:
        await _unusable(base_url="").generate_answer("prompt")
    assert raised.value.kind is LLMErrorKind.NOT_CONFIGURED
    assert "endpoint is configured" in str(raised.value)
    assert "base URL" in str(raised.value)


@pytest.mark.asyncio
async def test_a_transport_failure_is_still_classified_normally():
    """The fix moves one call out of a try block; it must not stop the block
    from doing its job."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"},
                              headers={"retry-after": "5"})

    client = _unusable()
    client._client = httpx.AsyncClient(
        base_url="https://gateway.example.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError) as raised:
        await client.generate_answer("prompt")
    assert raised.value.kind is LLMErrorKind.RATE_LIMIT
    assert raised.value.retry_after_seconds == 5.0


@pytest.mark.asyncio
async def test_an_already_classified_error_is_never_re_classified():
    """The general form of the bug: this layer must not run its own errors
    back through `classify`, which knows nothing about them and answers
    UNKNOWN."""
    client = _unusable()
    original = LLMError("already normalised", kind=LLMErrorKind.MODEL_UNAVAILABLE)
    assert client._as_llm_error(original) is original
    assert client.last_error_kind is LLMErrorKind.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_cancellation_is_still_cancellation():
    """Moved code must not have turned a supersede into a provider failure."""
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": _ANSWER_JSON}}]})

    client = _unusable()
    client._client = httpx.AsyncClient(
        base_url="https://gateway.example.invalid/v1",
        transport=httpx.MockTransport(slow),
    )
    task = asyncio.create_task(client.generate_answer("prompt"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
