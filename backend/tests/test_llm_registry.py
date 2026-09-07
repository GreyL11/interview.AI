"""Provider registry: config -> LLMClient, and nothing else.

This is the seam the whole "provider-neutral" architecture hangs off: given a
`ProviderConfig`, `build_client` must pick the right implementation and reject
anything unusable *before* a request is sent, without ever branching on
anything but the explicit `provider` name.
"""

import pytest

from app.llm.groq_client import GroqClient
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.registry import (
    ProviderConfig,
    ProviderConfigError,
    build_client,
    display_name,
    known_providers,
    validate_base_url,
    validate_model,
)

# ------------------------------------------------------------- routing table


def test_groq_routes_to_the_groq_client():
    client = build_client(ProviderConfig(provider="groq", model="openai/gpt-oss-120b"))
    assert isinstance(client, GroqClient)


def test_openai_routes_to_the_openai_compatible_transport():
    client = build_client(ProviderConfig(provider="openai", model="gpt-4o-mini", api_key="x"))
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider_name == "openai"


def test_openai_compatible_routes_to_the_same_transport():
    client = build_client(
        ProviderConfig(
            provider="openai_compatible", model="local-model", api_key="x",
            base_url="http://localhost:8000/v1",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)


def test_omnirouter_routes_to_the_same_transport_as_openai():
    """OmniRouter is OpenAI-compatible (see docs/llm-providers.md), so it must
    not duplicate the HTTP client -- it is exactly this branch."""
    client = build_client(
        ProviderConfig(
            provider="omnirouter", model="some-supported-model", api_key="x",
            base_url="http://localhost:20128/v1",
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider_name == "omnirouter"


@pytest.mark.parametrize("provider", ["gemini", "anthropic", "", "GROQ", "bedrock"])
def test_an_unknown_provider_is_rejected_not_guessed(provider):
    """Provider selection is always explicit. A name this build has no
    registry entry for is a configuration error, never silently routed
    anywhere -- including a near-miss like the wrong case."""
    with pytest.raises(ProviderConfigError):
        build_client(ProviderConfig(provider=provider, model="m", api_key="x"))


def test_known_providers_lists_every_routable_name():
    assert known_providers() == {"groq", "openai", "openai_compatible", "omnirouter"}


# ---------------------------------------------------------------- new models


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna", "some-brand-new-model-2027"])
def test_an_arbitrary_model_identifier_is_accepted_as_plain_configuration(model):
    """GPT-5.6 Terra, GPT-5.6 Luna, and anything shipped after them are just
    strings here -- there is no per-model branch anywhere in this module."""
    client = build_client(
        ProviderConfig(
            provider="openai_compatible", model=model, api_key="x",
            base_url="http://localhost:8000/v1",
        )
    )
    assert client.model_name == model


# --------------------------------------------------------------- validation


def test_a_missing_model_is_rejected_for_every_provider():
    for provider in known_providers():
        with pytest.raises(ProviderConfigError):
            build_client(ProviderConfig(provider=provider, model="", api_key="x", base_url="http://x"))


@pytest.mark.parametrize("bad", ["", "   ", "two words", "has\ttab"])
def test_an_unusable_model_is_rejected_at_configuration_time(bad):
    with pytest.raises(ProviderConfigError):
        validate_model(bad)


def test_a_usable_model_is_trimmed():
    assert validate_model("  gpt-5.6-terra  ") == "gpt-5.6-terra"


def test_openai_compatible_without_a_base_url_is_rejected():
    with pytest.raises(ProviderConfigError):
        build_client(ProviderConfig(provider="openai_compatible", model="m", api_key="x"))


def test_omnirouter_without_a_base_url_is_rejected():
    with pytest.raises(ProviderConfigError):
        build_client(ProviderConfig(provider="omnirouter", model="m", api_key="x"))


def test_groq_needs_no_base_url():
    """Groq has a fixed endpoint of its own; the registry must not demand one."""
    build_client(ProviderConfig(provider="groq", model="m", api_key="x"))  # must not raise


def test_openai_has_a_default_endpoint_so_a_base_url_is_optional():
    client = build_client(ProviderConfig(provider="openai", model="m", api_key="x"))
    assert client._base_url == "https://api.openai.com/v1"


def test_a_base_url_without_a_scheme_is_rejected():
    with pytest.raises(ProviderConfigError):
        validate_base_url("openai_compatible", "localhost:8000/v1")


def test_a_well_formed_base_url_is_accepted():
    assert validate_base_url("openai_compatible", "http://localhost:8000/v1") == (
        "http://localhost:8000/v1"
    )


def test_base_url_is_not_required_for_a_provider_that_does_not_need_one():
    assert validate_base_url("groq", "") == ""


# ------------------------------------------------------------------ secrecy


def test_provider_config_repr_never_contains_the_key():
    config = ProviderConfig(provider="openai", model="m", api_key="TEST_SECRET_DO_NOT_LEAK")
    assert "TEST_SECRET_DO_NOT_LEAK" not in repr(config)
    assert "TEST_SECRET_DO_NOT_LEAK" not in str(config)
    assert "set" in repr(config)


def test_provider_config_repr_says_unset_when_there_is_no_key():
    assert "unset" in repr(ProviderConfig(provider="groq", model="m"))


# --------------------------------------------------------------- display


def test_known_providers_have_a_human_readable_name():
    assert display_name("groq") == "Groq"
    assert display_name("openai") == "OpenAI"
    assert display_name("omnirouter") == "OmniRouter"


def test_an_unlisted_provider_falls_back_to_a_capitalised_name():
    assert display_name("some_future_gateway") == "Some_future_gateway"
