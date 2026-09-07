# LLM providers

The interview pipeline (Question Understanding, Answer Generation, session
summarisation) depends only on `app.llm.base.LLMClient`. It never imports
`GroqClient`, `OpenAICompatibleClient`, or any other concrete provider —
those live behind `app.llm.registry.build_client`, which turns a
`ProviderConfig` (provider name, model, key, base URL) into the client that
speaks that provider's protocol.

```
Interview pipeline  →  LLMClient  →  Provider Registry  →  Groq SDK
                                                        →  OpenAI-compatible transport
                                                             (OpenAI, OmniRouter, self-hosted, ...)
```

Model and provider are **configuration, not code**. Switching from Groq to
OmniRouter, or from one model to another, never touches the pipeline —
see `app/core/deps.py` (`_answer_config`, `_understanding_config`).

## Configuring a provider

Two independent stages can be configured, each with its own provider, model,
and (where relevant) base URL:

| Setting | Purpose | Empty means |
|---|---|---|
| `llm_provider` | Answer generation's provider | (required — defaults to `groq`) |
| `llm_model` | Answer generation's model | (required) |
| `llm_base_url` | Only for providers with no fixed endpoint | not applicable |
| `question_understanding_provider` | Question Understanding's provider | same as `llm_provider` |
| `question_understanding_model` | Question Understanding's model | same as `llm_model` |
| `question_understanding_base_url` | Question Understanding's base URL | same as `llm_base_url` |

Each provider's API key lives in its own Settings field, named
`<provider>_api_key` (`groq_api_key`, `openai_api_key`,
`openai_compatible_api_key`, `omnirouter_api_key`). This is the same
convention `app.core.secrets` already used for Groq — any `*_api_key` field
is automatically discovered, loaded from, and saved to the OS credential
store (Windows Credential Manager / macOS Keychain / Secret Service) with no
further wiring. Keys are **never** persisted in a plaintext settings file,
never returned by any API response, and redacted from `repr(settings)` /
`str(settings)` (see `Settings.__repr_args__`).

Set them through the Settings API:

```
PUT /settings
{"llm_provider": "openai", "llm_model": "gpt-4o-mini"}

PUT /providers/openai/key
{"api_key": "sk-..."}
```

A provider name that `app.llm.registry` does not know, a blank model, or a
missing/malformed `base_url` for a provider that needs one is rejected with
`400` at `PUT /settings` — before it can take the provider down on the next
question.

## Supported providers

- **`groq`** — reached through the official `groq` SDK (`app/llm/groq_client.py`).
  Needs no `base_url`.
- **`openai`** — reached through the generic OpenAI-compatible transport
  (`app/llm/openai_compatible.py`), defaulting to `https://api.openai.com/v1`
  when no `base_url` is given.
- **`openai_compatible`** — the same transport, pointed at any endpoint that
  speaks the OpenAI Chat Completions protocol (a self-hosted gateway, vLLM,
  LM Studio, Ollama's OpenAI-compatible shim, ...). Requires `base_url`.
- **`omnirouter`** — the same transport again. OmniRouter publishes an
  OpenAI-compatible `/v1` endpoint (bearer-token auth, standard chat-completions
  request/response shape), so it needs no dedicated HTTP client — only its
  own provider name in the registry. **Requires `base_url`**: OmniRouter has
  no single fixed public endpoint (it is typically self-hosted, commonly at
  `http://localhost:<port>/v1`), so point it at whichever instance issued
  your key. Before depending on this in production, verify against *your*
  OmniRouter deployment's own docs — the compatibility contract above was
  checked against public documentation at the time this was written, not
  against a live instance.

Adding a future OpenAI-compatible provider is a one-line addition to
`_OPENAI_COMPATIBLE_PROVIDERS` in `app/llm/registry.py`, never a new HTTP
client. A provider with a genuinely different protocol gets its own module
next to `groq_client.py`, behind the same `LLMClient` interface.

## Model identifiers

There is no allow-list of model names anywhere in this codebase. `llm_model`
and `question_understanding_model` are plain strings, validated only for
"non-empty, no whitespace" (`app.llm.registry.validate_model`) — whether the
model actually exists is answered by the provider itself on the first
request, and surfaces as a `MODEL_UNAVAILABLE` error naming the model.

This is what lets a model like **GPT-5.6 Terra** or **GPT-5.6 Luna** work as
an ordinary configuration value: set `llm_model` (or
`question_understanding_model`) to the identifier your provider expects, on
whichever provider actually serves it (e.g. `openai_compatible` or
`omnirouter` pointed at a gateway that exposes it) — nothing in this
application special-cases the name. These specific identifiers were not
independently verified against a live provider at the time this was written;
verify the exact model name your provider expects before relying on it.

## Error handling

Every provider client normalises its failures onto one taxonomy
(`app.llm.base.LLMErrorKind`): `not_configured`, `auth`, `model_unavailable`,
`rate_limit`, `timeout`, `network`, `server`, `malformed`, `unknown`. The rest
of the app reacts to the *kind*, never to a provider-specific exception
class, and the user-facing sentence for each kind is shared across every
provider (`app.llm.messages`) — only the provider's display name changes.

Nothing in this layer retries automatically. A provider's own SDK may retry
what it judges safe (Groq's SDK retries connection errors, 408, 429, and
5xx); this app never adds a second retry loop on top, because a duplicated
retry on the interview's realtime path risks sending two answers to one
question.

Cancellation (`asyncio.CancelledError`, raised when a newer question
supersedes an in-flight one) is never caught and reclassified as a provider
error — every provider client re-raises it unchanged.

## Adding a new provider

1. If it speaks the OpenAI Chat Completions protocol, add its name to
   `_OPENAI_COMPATIBLE_PROVIDERS` (and, if it needs a `base_url`, to
   `_REQUIRES_BASE_URL`) in `app/llm/registry.py`. Done — no new code.
2. Otherwise, write a new module next to `app/llm/groq_client.py`
   implementing `LLMClient` (`generate_answer`, `stream_answer`, and
   `complete_json` if it should serve Question Understanding too), mapping
   that provider's own exceptions onto `LLMErrorKind`. Add one branch to
   `build_client` in `app/llm/registry.py`.
3. Add a `<name>_api_key` field to `Settings` if the provider needs a key —
   `app.core.secrets` picks it up automatically.
4. Add its display name to `PROVIDER_DISPLAY_NAMES` in `app/llm/registry.py`
   (optional — falls back to a capitalised provider name).

Never infer a provider from an API key's shape (a prefix like `sk-` or
`gsk_`). Provider selection is always the explicit `provider` field; a
gateway can issue keys with any prefix it likes.
