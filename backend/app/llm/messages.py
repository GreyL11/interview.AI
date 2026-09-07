"""User-facing sentences for each `LLMErrorKind`, shared by every provider.

One copy of the wording, so two providers hitting the identical failure
explain it identically to whoever reads Settings -- only the provider's own
display name changes (see `app.llm.registry.display_name`). A provider module
supplies nothing here beyond that name and, for MODEL_UNAVAILABLE, the model.

Composition
-----------
Provider names come in two grammatical shapes, and a template that assumes
one produces nonsense for the other:

    "Groq"                            a proper noun
    "the OpenAI-compatible endpoint"  a definite noun phrase

An earlier version wrote every template around the proper-noun shape ("The
{name} request failed.", "No {name} API key is configured.") and then
substituted a name that already carried its own article, producing "The the
configured endpoint request failed." and "No the configured endpoint API key
is configured."

So the name is offered to the templates in the three forms English actually
needs, and each template asks for the one that fits its slot:

    {bare}  Groq            | OpenAI-compatible endpoint      after an article
    {name}  Groq            | the OpenAI-compatible endpoint   mid-sentence
    {Name}  Groq            | The OpenAI-compatible endpoint   sentence-initial

Which shape a display name has is read off the name itself -- a leading
"the " -- so registering a provider stays a one-line change with no extra
flag to remember. For a proper noun all three forms are the same string,
which is why every existing sentence for Groq, OpenAI and OmniRouter is
byte-for-byte what it was.
"""

from app.llm.base import LLMErrorKind

_TEMPLATES: dict[LLMErrorKind, str] = {
    LLMErrorKind.AUTH: (
        "The {bare} API key was rejected. Check it in Settings and save it again."
    ),
    LLMErrorKind.NOT_CONFIGURED: (
        "No {bare} API key is configured. Add one in Settings to enable answers."
    ),
    LLMErrorKind.MODEL_UNAVAILABLE: (
        "The configured {bare} model is not available to this account. "
        "Choose a different model in Settings."
    ),
    LLMErrorKind.RATE_LIMIT: "{Name}'s rate limit was reached. Wait a moment and ask again.",
    LLMErrorKind.TIMEOUT: "{Name} did not respond in time. Ask again.",
    LLMErrorKind.NETWORK: "Could not reach {name}. Check this machine's internet connection.",
    LLMErrorKind.SERVER: "{Name} reported a server error. Try again shortly.",
    LLMErrorKind.MALFORMED: "{Name} returned a response this app could not read.",
    LLMErrorKind.UNKNOWN: "The {bare} request failed.",
}

#: MODEL_UNAVAILABLE names the model when one is known. A separate template
#: rather than string surgery on the one above: substituting into rendered
#: prose broke the moment the rendered prose stopped starting with "The
#: {name} model".
_MODEL_UNAVAILABLE_WITH_MODEL = (
    "The {bare} model '{model}' is not available to this account. "
    "Choose a different model in Settings."
)

#: Used when the failure is known but the provider is not -- see
#: `app.realtime.session`, which sanitises an error it did not build.
GENERIC_PROVIDER_NAME = "the AI provider"

_DEFINITE_PREFIX = "the "


def name_forms(name: str) -> dict[str, str]:
    """The three grammatical forms of a provider's display name.

    A name that already carries its article is definite; anything else is
    treated as a proper noun, for which all three forms coincide.
    """
    if name.lower().startswith(_DEFINITE_PREFIX):
        bare = name[len(_DEFINITE_PREFIX):]
        return {
            "bare": bare,
            "name": f"{_DEFINITE_PREFIX}{bare}",
            "Name": f"{_DEFINITE_PREFIX.capitalize()}{bare}",
        }
    return {"bare": name, "name": name, "Name": name}


def message_for(kind: LLMErrorKind, name: str, model: str | None = None) -> str:
    """The sentence Settings and the answer pane show verbatim for `kind`.

    `name` is the provider's *display* name, never the raw provider key and
    never anything derived from the request -- a provider's own error string
    can echo the request back, key included, so nothing from the causing
    exception ever reaches this string.
    """
    forms = name_forms(name)
    if kind is LLMErrorKind.MODEL_UNAVAILABLE and model:
        return _MODEL_UNAVAILABLE_WITH_MODEL.format(model=model, **forms)
    template = _TEMPLATES.get(kind, _TEMPLATES[LLMErrorKind.UNKNOWN])
    return template.format(**forms)


def missing_endpoint_message(name: str) -> str:
    """A provider that needs an endpoint of its own and was not given one.

    Distinct from NOT_CONFIGURED's sentence, which is about a key: telling
    someone to add an API key when what is missing is the base URL sends
    them to the wrong field.
    """
    return (
        f"No endpoint is configured for {name_forms(name)['name']}. "
        "Set its base URL in Settings."
    )


def is_known_message(text: str) -> bool:
    """Was this sentence built here?

    The whitelist behind `app.realtime.session`'s error sanitising: a message
    this module produced is safe to show, because every one of them is a
    fixed template filled only with a provider display name and a model
    identifier. Anything else -- a provider's own error string, an internal
    exception's text -- is not, and must not reach a client.

    Model identifiers are configuration rather than a closed set, so
    MODEL_UNAVAILABLE is matched on its fixed head and tail instead of by
    equality.
    """
    return text in _known_messages() or _is_model_unavailable_message(text)


def _known_messages() -> frozenset[str]:
    global _CACHED_KNOWN
    if _CACHED_KNOWN is None:
        from app.llm.registry import PROVIDER_DISPLAY_NAMES

        names = {*PROVIDER_DISPLAY_NAMES.values(), GENERIC_PROVIDER_NAME}
        _CACHED_KNOWN = frozenset(
            [message_for(kind, name) for kind in LLMErrorKind for name in names]
            + [missing_endpoint_message(name) for name in names]
        )
    return _CACHED_KNOWN


_CACHED_KNOWN: frozenset[str] | None = None

_MODEL_HEAD = "The "
_MODEL_TAIL = " is not available to this account. Choose a different model in Settings."


def _is_model_unavailable_message(text: str) -> bool:
    return (
        text.startswith(_MODEL_HEAD)
        and text.endswith(_MODEL_TAIL)
        and " model '" in text
    )
