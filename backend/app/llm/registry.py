"""Provider registry: configuration in, an `LLMClient` out.

The interview pipeline -- Question Understanding, Answer Generation, the
session summariser -- each ask `app.core.deps` for "an LLMClient" and never
know which provider they got; see `app.llm.base.LLMClient`. This module is
the one place that maps a provider *name* (data, ultimately typed into
Settings) onto the implementation that speaks its protocol. Adding a provider
means adding one entry to `_OPENAI_COMPATIBLE_PROVIDERS` (if it speaks the
OpenAI protocol) or one branch in `build_client` (if it does not) -- never a
new branch in the pipeline itself.

Provider selection is always explicit, never inferred. In particular this
module never looks at the *shape* of an API key (a prefix like `sk-` or
`gsk_`) to guess a provider -- a gateway can issue keys with any prefix it
likes, and doing so would silently misroute a request the moment that
assumption stopped holding.
"""

from dataclasses import dataclass

from app.llm.base import LLMClient

#: Display name shown in user-facing error sentences (see `app.llm.messages`).
#: Falls back to `provider.capitalize()` for anything not listed here -- a
#: self-hosted gateway registered only as "openai_compatible", for instance.
#:
#: A name may be a proper noun ("Groq") or a definite noun phrase written
#: with its article ("the OpenAI-compatible endpoint"). `app.llm.messages`
#: reads the leading "the " and renders the sentence accordingly, so a
#: provider with no name of its own still composes grammatically; do not
#: strip the article to "make it look like the others".
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "groq": "Groq",
    "openai": "OpenAI",
    "openai_compatible": "the OpenAI-compatible endpoint",
    "omnirouter": "OmniRouter",
}

#: Providers reached through the generic OpenAI-Compatible transport
#: (`app.llm.openai_compatible`) rather than a native SDK. OmniRouter is here
#: because it publishes an OpenAI-compatible `/v1` endpoint -- see
#: docs/llm-providers.md for what was verified and what a new deployment
#: should still check for itself before trusting it in production.
_OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openai_compatible", "omnirouter"})

#: Providers with no endpoint of their own, so Settings must supply one.
_REQUIRES_BASE_URL = frozenset({"openai_compatible", "omnirouter"})


def display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())


def known_providers() -> frozenset[str]:
    return frozenset({"groq"}) | _OPENAI_COMPATIBLE_PROVIDERS


def requires_base_url(provider: str) -> bool:
    return provider in _REQUIRES_BASE_URL


class ProviderConfigError(ValueError):
    """A configuration mistake caught before any request is sent: a blank or
    malformed model, an unknown provider, or a base_url a provider needs but
    was not given. Raised at construction/validation time rather than
    surfacing as a confusing provider-side failure on the first question."""


@dataclass(frozen=True)
class ProviderConfig:
    """Everything one `LLMClient` needs, resolved from Settings.

    A plain value object with no behaviour -- `build_client` reads it and
    nothing else does. `api_key` never appears in `repr()`/`str()`: a stray
    log line, an unhandled traceback, or a debugger's repr of a local variable
    must not be how a key leaks.
    """

    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={'set' if self.api_key else 'unset'}, "
            f"base_url={self.base_url!r}, timeout_seconds={self.timeout_seconds!r}, "
            f"max_retries={self.max_retries!r})"
        )


def validate_model(name: str) -> str:
    """Normalise and sanity-check a model identifier.

    Deliberately not an allow-list of known models: that list changes on each
    provider's own schedule, not this app's, and a stale copy would reject a
    model that works -- including a brand new one (GPT-5.6 Terra, GPT-5.6
    Luna, or whatever ships after them). What *can* be checked locally is
    checked here; whether the account can actually use the model is answered
    by the provider itself, and surfaces as MODEL_UNAVAILABLE with the name in
    the message.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise ProviderConfigError("The model is empty; set a model identifier.")
    if any(character.isspace() for character in cleaned):
        raise ProviderConfigError(
            f"Model {cleaned!r} contains whitespace; it must be a single identifier."
        )
    return cleaned


def validate_base_url(provider: str, base_url: str) -> str:
    """Sanity-check a base_url without pretending to know it is reachable.

    A provider that needs one (no fixed endpoint of its own) must have a
    non-empty value; anything supplied, for any provider, must at least look
    like a URL. Whether it actually answers is discovered on the first
    request and surfaces as NETWORK or SERVER, not here.
    """
    cleaned = (base_url or "").strip()
    if requires_base_url(provider) and not cleaned:
        raise ProviderConfigError(
            f"'{provider}' needs a base_url (an OpenAI-compatible endpoint); "
            "none was configured."
        )
    if cleaned and not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        raise ProviderConfigError(f"base_url {cleaned!r} must start with http:// or https://.")
    return cleaned


def build_client(config: ProviderConfig) -> LLMClient:
    """Composition root for one provider client.

    `config.provider` is untrusted data -- ultimately from the Settings API --
    and is never used to *infer* anything about the key or the wire protocol;
    it only selects which branch below runs. An unknown provider, a blank
    model, or a missing base_url is a configuration failure raised here, not a
    crash: the app must still start and serve everything that is not an
    answer (see how callers of `app.core.deps.get_llm_client` already handle
    this for the single-provider case).
    """
    validate_model(config.model)

    if config.provider == "groq":
        from app.llm.groq_client import GroqClient

        return GroqClient(config)

    if config.provider in _OPENAI_COMPATIBLE_PROVIDERS:
        from app.llm.openai_compatible import OpenAICompatibleClient

        validate_base_url(config.provider, config.base_url)
        return OpenAICompatibleClient(config)

    raise ProviderConfigError(
        f"Unknown LLM provider '{config.provider}'. Known providers: "
        f"{', '.join(sorted(known_providers()))}."
    )
