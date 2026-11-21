"""Groq behind the shared `LLMClient` interface.

Reached through the official SDK, so this module is genuinely Groq-specific
in a way the OpenAI-compatible transport (`app.llm.openai_compatible`) is
not: Groq's SDK raises its own exception classes rather than plain HTTP
errors, so the mapping from those classes onto `LLMErrorKind` has to live
somewhere, and here is the only place that imports the SDK at all.

What lives here, and nowhere else:

  * the client (one per instance, so its connection pool is reused across
    requests -- see `_ensure_client`)
  * the mapping from Groq's exception classes onto `LLMErrorKind`

Everything else -- the wording for each failure kind, answer parsing and
validation, provider selection -- is shared (`app.llm.messages`,
`app.llm.streaming`, `app.llm.registry`, `app.intelligence.answer_validator`),
because none of it is actually Groq-specific.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.llm.base import (
    LLMClient, LLMError, LLMErrorKind, answer_max_tokens, json_response_format, keepalive_limits,
)
from app.llm.messages import message_for
from app.llm.registry import ProviderConfig, validate_model
from app.llm.streaming import parse_answer_payload
from app.schemas.answer import Answer

logger = get_logger(__name__)

_NAME = "Groq"


class GroqClient(LLMClient):
    """Groq behind the shared LLMClient interface.

    Construction never raises on a missing key: the app must still start and
    still serve documents, history and transcription without one, so "no key"
    surfaces per-answer as a NOT_CONFIGURED LLMError. A malformed *model*
    still raises at construction (via `validate_model`), because that is a
    configuration mistake rather than a runtime failure worth rediscovering on
    every question.
    """

    provider_name = "groq"

    #: Groq SDK exception class names, mapped to our taxonomy. Matched by name
    #: rather than by isinstance so this module never has to import the SDK at
    #: definition time -- importing groq costs ~1.8s and must stay in warmup().
    _BY_EXCEPTION_NAME: dict[str, LLMErrorKind] = {
        "AuthenticationError": LLMErrorKind.AUTH,
        "PermissionDeniedError": LLMErrorKind.AUTH,
        "NotFoundError": LLMErrorKind.MODEL_UNAVAILABLE,
        "RateLimitError": LLMErrorKind.RATE_LIMIT,
        "APITimeoutError": LLMErrorKind.TIMEOUT,
        "APIConnectionError": LLMErrorKind.NETWORK,
        "InternalServerError": LLMErrorKind.SERVER,
        "APIResponseValidationError": LLMErrorKind.MALFORMED,
        # httpx/anyio shapes, in case one escapes the SDK's own wrapping.
        "ConnectError": LLMErrorKind.NETWORK,
        "ConnectTimeout": LLMErrorKind.TIMEOUT,
        "ReadTimeout": LLMErrorKind.TIMEOUT,
        "TimeoutException": LLMErrorKind.TIMEOUT,
        "RemoteProtocolError": LLMErrorKind.NETWORK,
    }

    _BY_STATUS: dict[int, LLMErrorKind] = {
        401: LLMErrorKind.AUTH,
        403: LLMErrorKind.AUTH,
        404: LLMErrorKind.MODEL_UNAVAILABLE,
        408: LLMErrorKind.TIMEOUT,
        429: LLMErrorKind.RATE_LIMIT,
        500: LLMErrorKind.SERVER,
        502: LLMErrorKind.SERVER,
        503: LLMErrorKind.SERVER,
        504: LLMErrorKind.SERVER,
    }

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = None
        #: Last classified failure, for the Settings screen. Never a message
        #: from the provider and never anything key-derived -- just the kind.
        self.last_error_kind: LLMErrorKind | None = None
        #: Set once this provider has rejected a schema-constrained request.
        #: Strict mode is model-gated and the model is user-configured, so an
        #: unsupported one must cost a single degraded call, not every call.
        self._schema_rejected = False
        # Fails fast and loudly on a misconfigured model rather than sending it
        # to Groq once per question.
        validate_model(config.model)

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def configured(self) -> bool:
        return bool(self._config.api_key)

    # ------------------------------------------------------------- lifecycle

    def _ensure_client(self):
        """One client per instance, so the connection pool is reused. A
        per-request client would discard it and put TLS setup on every
        answer."""
        if self._client is not None:
            return self._client
        if not self._config.api_key:
            raise LLMError(
                message_for(LLMErrorKind.NOT_CONFIGURED, _NAME),
                kind=LLMErrorKind.NOT_CONFIGURED,
            )
        try:
            from groq import AsyncGroq, DefaultAsyncHttpxClient
        except ImportError as exc:  # pragma: no cover - packaging guard
            # A packaging failure, not a user error: say so plainly rather than
            # telling the user to install a Python package into a desktop app.
            logger.error("groq_sdk_missing error=%s", exc)
            raise LLMError(
                "This installation is missing its Groq support files. "
                "Reinstall Call Assistant.",
                kind=LLMErrorKind.NOT_CONFIGURED,
            ) from exc

        self._client = AsyncGroq(
            api_key=self._config.api_key,
            # The SDK already retries exactly the failures worth retrying
            # (connection errors, 408, 429, 5xx) with backoff, and never retries
            # a deterministic 400/401/404. Set explicitly so that contract is
            # visible here rather than inherited from an SDK default.
            max_retries=self._config.max_retries,
            # SDK default client, plus a long keep-alive (see base.py).
            http_client=DefaultAsyncHttpxClient(limits=keepalive_limits()),
        )
        logger.info("groq_client_initialised model=%s", self.model_name)
        return self._client

    def reset(self) -> None:
        """Drop the cached client so the next call picks up a new key."""
        self._client = None
        self.last_error_kind = None

    def warmup(self) -> None:
        # Measured on this machine: `import groq` ~1.8s, AsyncGroq() ~0.6s.
        # Lazily that lands on the first question of a session -- the worst
        # possible moment for an interview copilot.
        try:
            self._ensure_client()
        except LLMError:
            pass  # no key: still reported per-answer, which is the right place

    # -------------------------------------------------------- classification

    def _status_code(self, exc: Exception) -> int | None:
        value = getattr(exc, "status_code", None)
        if isinstance(value, int):
            return value
        response = getattr(exc, "response", None)
        if response is not None:
            value = getattr(response, "status_code", None)
            if isinstance(value, int):
                return value
        return None

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        for source in (exc, getattr(exc, "response", None)):
            headers = getattr(source, "headers", None)
            if not headers:
                continue
            for key in ("retry-after", "Retry-After"):
                try:
                    raw = headers.get(key)
                except AttributeError:
                    continue
                if raw is None:
                    continue
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        return None

    def classify(self, exc: Exception) -> LLMErrorKind:
        """Map an SDK exception onto the shared taxonomy.

        Exception class first, status code second: Groq's own classes are more
        specific than the status alone (a 404 is always a missing model here,
        because the only resource this app addresses by name is the model).
        """
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return LLMErrorKind.TIMEOUT

        by_name = self._BY_EXCEPTION_NAME.get(exc.__class__.__name__)
        if by_name is not None:
            return by_name

        status = self._status_code(exc)
        if status is not None:
            by_status = self._BY_STATUS.get(status)
            if by_status is not None:
                return by_status
            if 500 <= status < 600:
                return LLMErrorKind.SERVER

        # A 400 that names the model is Groq's other way of saying "no such
        # model"; anything else at 400 is a bug in the request this app built.
        if status == 400 and "model" in str(exc).lower():
            return LLMErrorKind.MODEL_UNAVAILABLE
        return LLMErrorKind.UNKNOWN

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
            message_for(kind, _NAME, self.model_name),
            kind=kind,
            retry_after_seconds=self._retry_after_seconds(exc),
        )

    def _fail(self, exc: Exception, started: float, phase: str) -> LLMError:
        """Classify, log, and return the error to raise.

        The log line carries provider, model, duration and classification --
        everything needed to diagnose a failure from a support bundle -- and the
        exception *type*, never its text, because a provider's error string can
        echo the request back.
        """
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

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        # The prompt already carries the system instruction and the schema hint,
        # built once by app.llm.prompts.
        return [{"role": "user", "content": prompt}]

    async def generate_answer(self, prompt: str) -> Answer:
        client = self._ensure_client()
        started = time.monotonic()
        log_metric(
            "llm_request_started",
            provider=self.provider_name,
            model=self.model_name,
            streaming=False,
        )
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model_name,
                    messages=self._messages(prompt),
                    response_format={"type": "json_object"},
                    **answer_max_tokens(),
                ),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._fail(exc, started, phase="request") from exc

        self.last_error_kind = None
        log_metric(
            "llm_request_completed",
            provider=self.provider_name,
            model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        return self._to_answer(response.choices[0].message.content or "")

    async def complete_json(
        self,
        prompt: str,
        *,
        model: str = "",
        timeout_seconds: float = 5.0,
        schema: dict | None = None,
    ) -> str:
        """One short, non-streaming JSON completion.

        Serves the question-understanding layer (`StructuredCompleter`), which
        needs a small structured object rather than a coaching answer, on a
        much tighter budget than the answer-generation timeout -- it sits on
        the realtime path between a finished question and its answer.

        JSON mode is safe to use here, unlike `stream_answer`: nothing streams
        this, so the provider buffering the whole object costs nothing.
        `model` falls back to the answer model when unset, so the setting can
        stay empty until there is a reason to split them.

        A `schema` upgrades JSON mode to constrained decoding -- see
        `app.llm.base.json_response_format` for what that does and does not
        guarantee. A model that cannot do it answers 400, which degrades this
        one call to the caller's fallback and every later one to plain JSON
        mode.
        """
        client = self._ensure_client()
        started = time.monotonic()
        chosen = model or self.model_name
        use_schema = schema is not None and not self._schema_rejected
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=chosen,
                    messages=self._messages(prompt),
                    response_format=json_response_format(schema if use_schema else None),
                    # A classifier that rambles has failed; this also bounds
                    # the worst-case latency contribution.
                    max_tokens=512,
                    temperature=0,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if use_schema and self._status_code(exc) == 400:
                # This model does not do constrained decoding. Stop asking for
                # it: the caller degrades this turn to its fallback, and every
                # later call goes out as plain JSON mode instead.
                self._schema_rejected = True
                log_metric(
                    "llm_schema_unsupported", provider=self.provider_name, model=chosen,
                )
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
        log_metric(
            "llm_understanding_completed",
            provider=self.provider_name,
            model=chosen,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        return response.choices[0].message.content or ""

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        client = self._ensure_client()
        started = time.monotonic()
        log_metric(
            "llm_request_started",
            provider=self.provider_name,
            model=self.model_name,
            streaming=True,
        )
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model_name,
                    messages=self._messages(prompt),
                    # No response_format={"type": "json_object"} here on
                    # purpose: Groq's JSON mode buffers the whole object and
                    # delivers it in a single content chunk, which kills the
                    # partial-summary streaming the UI depends on. The prompt
                    # already demands raw JSON, and parse_answer_payload
                    # tolerates a stray markdown fence. JSON mode is still used
                    # in generate_answer above, where buffering costs nothing.
                    stream=True,
                    **answer_max_tokens(),
                ),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._fail(exc, started, phase="open_stream") from exc

        first_text_seen = False
        characters = 0
        try:
            async for chunk in stream:
                text = _chunk_text(chunk)
                if not text:
                    continue
                characters += len(text)
                if not first_text_seen:
                    first_text_seen = True
                    log_metric(
                        "llm_first_token",
                        provider=self.provider_name,
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
            "llm_request_completed",
            provider=self.provider_name,
            model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
            chars=characters,
        )

    # -------------------------------------------------------------- parsing

    def _to_answer(self, text: str) -> Answer:
        text = (text or "").strip()
        if not text:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, _NAME), kind=LLMErrorKind.MALFORMED
            )
        try:
            data = parse_answer_payload(text)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, _NAME), kind=LLMErrorKind.MALFORMED
            ) from exc
        try:
            return Answer.model_validate(data)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                message_for(LLMErrorKind.MALFORMED, _NAME), kind=LLMErrorKind.MALFORMED
            ) from exc


def _chunk_text(chunk) -> str:
    """Pull the text delta out of a Groq streaming chunk, tolerating the empty
    and role-only chunks the API sends around the actual content."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None) or "" if delta is not None else ""
