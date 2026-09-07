"""Transport for any endpoint that speaks the OpenAI Chat Completions
protocol: OpenAI itself, a self-hosted gateway (vLLM, LM Studio, Ollama's
OpenAI-compatible shim, ...), and any other compatible provider -- OmniRouter
included, since it publishes an OpenAI-compatible `/v1` endpoint. See
docs/llm-providers.md for exactly what was verified about OmniRouter and what
a given deployment should still check for itself (self-hosted gateways vary
in base_url and in exactly which optional fields they accept).

One request/response shape, one error mapping, for every provider that speaks
it. Adding a new compatible endpoint is a Settings change -- provider name,
base_url, model, key -- never a new HTTP client; see `app.llm.registry` for
where a provider name gets routed here.

Built on `httpx`, already a dependency of this app, rather than the `openai`
SDK: the whole surface this needs -- one POST, one JSON body, one bearer
header, one SSE stream to read -- is a few dozen lines, and a second SDK for
the wire protocol `groq_client.py` already exercises would be a dependency for
something a few lines already do.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.messages import message_for, missing_endpoint_message
from app.llm.registry import ProviderConfig, display_name
from app.llm.streaming import parse_answer_payload
from app.schemas.answer import Answer

logger = get_logger(__name__)

#: Fixed endpoints for providers that have one. A provider absent here (an
#: openai_compatible gateway, OmniRouter, a future addition) must supply its
#: own base_url -- see `app.llm.registry.requires_base_url`.
_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}

#: Exceptions raised by response-shape problems rather than transport
#: problems -- a body that is not JSON, or JSON missing the field this app
#: reads. Classified as MALFORMED rather than UNKNOWN: the request reached the
#: provider and got an answer, just not one shaped the way this app expects.
_MALFORMED_EXCEPTIONS = (KeyError, IndexError, ValueError)


class OpenAICompatibleClient(LLMClient):
    """One provider configuration is the whole difference between OpenAI,
    OmniRouter and a self-hosted gateway -- see the module docstring.

    `provider_name` is the *configured* provider, not a fixed string, so
    metrics and the Settings screen tell them apart even though every one of
    them runs through this one implementation.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.provider_name = config.provider
        self._config = config
        self._name = display_name(config.provider)
        base_url = config.base_url.strip() or _DEFAULT_BASE_URLS.get(config.provider, "")
        self._base_url = base_url.rstrip("/")
        #: Last classified failure, for the Settings screen. Never a message
        #: from the provider and never anything key-derived -- just the kind.
        self.last_error_kind: LLMErrorKind | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def configured(self) -> bool:
        return bool(self._config.api_key)

    # ------------------------------------------------------------- lifecycle

    def _ensure_client(self) -> httpx.AsyncClient:
        """One client per instance, so the connection pool is reused. A
        per-request client would discard it and put TLS setup on every
        answer."""
        if self._client is not None:
            return self._client
        if not self._config.api_key:
            raise LLMError(
                message_for(LLMErrorKind.NOT_CONFIGURED, self._name),
                kind=LLMErrorKind.NOT_CONFIGURED,
            )
        if not self._base_url:
            # Caught earlier by `validate_base_url` for every path that goes
            # through the registry; kept here too so constructing this class
            # directly (as tests do) fails the same honest way. Worded through
            # `app.llm.messages` like every other user-facing sentence, so it
            # is on the whitelist the realtime layer checks before showing an
            # error to a client -- see `messages.is_known_message`.
            raise LLMError(
                missing_endpoint_message(self._name),
                kind=LLMErrorKind.NOT_CONFIGURED,
            )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._config.timeout_seconds),
        )
        logger.info(
            "openai_compatible_client_initialised provider=%s model=%s base_url=%s",
            self.provider_name, self.model_name, self._base_url,
        )
        return self._client

    def reset(self) -> None:
        """Drop the cached client so the next call picks up a new key."""
        self._client = None
        self.last_error_kind = None

    def warmup(self) -> None:
        try:
            self._ensure_client()
        except LLMError:
            pass  # no key: still reported per-answer, which is the right place

    # -------------------------------------------------------- classification

    def classify(self, exc: Exception) -> LLMErrorKind:
        """Map an httpx failure, or an HTTP error status, onto the shared
        taxonomy.

        Every OpenAI-compatible provider uses the same status-code
        conventions, so -- unlike Groq's SDK-specific exception classes --
        this is the whole mapping for every provider this transport serves.
        """
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            return LLMErrorKind.TIMEOUT
        if isinstance(exc, httpx.ConnectError):
            return LLMErrorKind.NETWORK
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return LLMErrorKind.AUTH
            if status == 404:
                return LLMErrorKind.MODEL_UNAVAILABLE
            if status == 429:
                return LLMErrorKind.RATE_LIMIT
            if 500 <= status < 600:
                return LLMErrorKind.SERVER
            if status == 400 and "model" in _safe_body_text(exc.response).lower():
                # Some gateways' other way of saying "no such model".
                return LLMErrorKind.MODEL_UNAVAILABLE
            return LLMErrorKind.UNKNOWN
        if isinstance(exc, httpx.HTTPError):
            return LLMErrorKind.NETWORK
        if isinstance(exc, _MALFORMED_EXCEPTIONS):
            return LLMErrorKind.MALFORMED
        return LLMErrorKind.UNKNOWN

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _as_llm_error(self, exc: Exception) -> LLMError:
        # An LLMError raised by this layer is already normalised: its `kind`
        # came from `classify` and its sentence from `app.llm.messages`.
        # Re-classifying it can only ever lose information -- that is how a
        # NOT_CONFIGURED "add a key in Settings" became an UNKNOWN "the
        # request failed" -- so it is returned as it stands.
        if isinstance(exc, LLMError):
            self.last_error_kind = exc.kind
            return exc
        kind = self.classify(exc)
        self.last_error_kind = kind
        return LLMError(
            message_for(kind, self._name, self.model_name),
            kind=kind,
            retry_after_seconds=self._retry_after_seconds(exc),
        )

    def _fail(self, exc: Exception, started: float, phase: str) -> LLMError:
        """Classify, log, and return the error to raise. The log line carries
        provider, model, duration and classification -- and the exception
        *type*, never its text, because a provider's error string can echo
        the request back."""
        error = self._as_llm_error(exc)
        log_metric(
            "llm_request_failed",
            provider=self.provider_name,
            model=self.model_name,
            phase=phase,
            failure=error.kind.value,
            exception=exc.__class__.__name__,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        return error

    # ------------------------------------------------------------- requests

    def _body(self, prompt: str, *, model: str, stream: bool, json_mode: bool, **extra) -> dict:
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(extra)
        return body

    async def _post(self, client: httpx.AsyncClient, body: dict, timeout: float) -> dict:
        response = await asyncio.wait_for(
            client.post("/chat/completions", json=body), timeout=timeout
        )
        response.raise_for_status()
        return _extract_message(response.json())

    async def generate_answer(self, prompt: str) -> Answer:
        # Outside the try, exactly as in `stream_answer` and in
        # `GroqClient.generate_answer`. `_ensure_client` raises an LLMError
        # that is already classified -- NOT_CONFIGURED for a missing key or
        # base_url -- and inside the try that classification was thrown away
        # and replaced with UNKNOWN by `_fail`, so the one failure with a
        # clear remedy ("Add a key in Settings") was reported as "The request
        # failed." Nothing here may re-classify an error this layer raised.
        client = self._ensure_client()
        started = time.monotonic()
        log_metric(
            "llm_request_started", provider=self.provider_name, model=self.model_name,
            streaming=False,
        )
        try:
            content = await self._post(
                client,
                self._body(prompt, model=self.model_name, stream=False, json_mode=True),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._fail(exc, started, phase="request") from exc

        self.last_error_kind = None
        log_metric(
            "llm_request_completed", provider=self.provider_name, model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        return self._to_answer(content)

    async def complete_json(
        self, prompt: str, *, model: str = "", timeout_seconds: float = 5.0
    ) -> str:
        """One short, non-streaming JSON completion, for the question-
        understanding layer (`StructuredCompleter`). Mirrors
        `GroqClient.complete_json` -- see that docstring for why this exists
        as its own method rather than reusing `generate_answer`."""
        client = self._ensure_client()
        chosen = model or self.model_name
        started = time.monotonic()
        body = self._body(
            prompt, model=chosen, stream=False, json_mode=True,
            max_tokens=512, temperature=0,
        )
        try:
            return await self._post(client, body, timeout=timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Classified and logged like any other provider failure, but *not*
            # recorded in `last_error_kind`: the Settings screen reports the
            # health of answering, and a classifier blip is recovered from
            # silently rather than shown to the user as a broken key.
            log_metric(
                "llm_request_failed",
                provider=self.provider_name,
                model=chosen,
                phase="understanding",
                failure=self.classify(exc).value,
                exception=exc.__class__.__name__,
                duration_ms=elapsed_ms(started, time.monotonic()),
            )
            raise

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        client = self._ensure_client()
        started = time.monotonic()
        log_metric(
            "llm_request_started", provider=self.provider_name, model=self.model_name,
            streaming=True,
        )
        # No response_format={"type": "json_object"} here on purpose, mirroring
        # groq_client.stream_answer: JSON mode on a streaming request buffers
        # the whole object into one chunk on some providers, which would kill
        # the partial-summary rendering the UI depends on. The prompt already
        # demands raw JSON, and parse_answer_payload tolerates a stray
        # markdown fence.
        body = self._body(prompt, model=self.model_name, stream=True, json_mode=False)

        first_text_seen = False
        characters = 0
        try:
            async with client.stream("POST", "/chat/completions", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    text = _sse_delta(line)
                    if not text:
                        continue
                    characters += len(text)
                    if not first_text_seen:
                        first_text_seen = True
                        log_metric(
                            "llm_first_token", provider=self.provider_name,
                            model=self.model_name,
                            duration_ms=elapsed_ms(started, time.monotonic()),
                        )
                    yield text
        except asyncio.CancelledError:
            # Normal control flow when a newer question supersedes this one.
            logger.info("llm_stream_cancelled provider=%s", self.provider_name)
            raise
        except Exception as exc:
            raise self._fail(exc, started, phase="stream") from exc

        self.last_error_kind = None
        log_metric(
            "llm_request_completed", provider=self.provider_name, model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()), chars=characters,
        )

    # -------------------------------------------------------------- parsing

    def _to_answer(self, text: str) -> Answer:
        text = (text or "").strip()
        if not text:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, self._name), kind=LLMErrorKind.MALFORMED
            )
        try:
            data = parse_answer_payload(text)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, self._name), kind=LLMErrorKind.MALFORMED
            ) from exc
        try:
            return Answer.model_validate(data)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, self._name), kind=LLMErrorKind.MALFORMED
            ) from exc


def _safe_body_text(response: httpx.Response) -> str:
    """`.text` on a response whose body was never read (a streaming request
    that failed before the first chunk) raises; that is not itself a failure
    worth surfacing over the classification already in hand."""
    try:
        return response.text
    except Exception:
        return ""


def _extract_message(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"] or ""


def _sse_delta(line: str) -> str:
    """Pull the text delta out of one Server-Sent-Events line of a streamed
    chat-completions response, tolerating the `[DONE]` sentinel and the
    empty/role-only chunks providers send around the actual content."""
    if not line.startswith("data:"):
        return ""
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return ""
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    try:
        return chunk["choices"][0]["delta"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
