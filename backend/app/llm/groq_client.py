"""The application's only cloud provider.

Groq is reached through the official SDK. This module is deliberately the whole
provider layer: there is one cloud model, so a router, a priority list and a
failover table would be machinery with nothing to route between.

What lives here, and nowhere else:

  * the client (one per process, so the connection pool is reused)
  * the mapping from Groq's exception classes onto `LLMErrorKind`
  * the user-facing sentence for each failure kind

Answer parsing and validation stay in the shared path (`app.llm.streaming`,
`app.intelligence.answer_validator`), which is provider-unaware.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.streaming import parse_answer_payload
from app.schemas.answer import Answer

logger = get_logger(__name__)

#: Sentences the Settings screen and the answer pane show verbatim. Each names
#: the thing the user can actually do about it; "The Groq request failed" is
#: only reachable when the SDK produced something we genuinely cannot classify.
_MESSAGES: dict[LLMErrorKind, str] = {
    LLMErrorKind.AUTH: (
        "The Groq API key was rejected. Check it in Settings and save it again."
    ),
    LLMErrorKind.NOT_CONFIGURED: (
        "No Groq API key is configured. Add one in Settings to enable answers."
    ),
    LLMErrorKind.MODEL_UNAVAILABLE: (
        "The configured Groq model is not available to this account. "
        "Choose a different model in Settings."
    ),
    LLMErrorKind.RATE_LIMIT: (
        "Groq's rate limit was reached. Wait a moment and ask again."
    ),
    LLMErrorKind.TIMEOUT: "Groq did not respond in time. Ask again.",
    LLMErrorKind.NETWORK: (
        "Could not reach Groq. Check this machine's internet connection."
    ),
    LLMErrorKind.SERVER: "Groq reported a server error. Try again shortly.",
    LLMErrorKind.MALFORMED: "Groq returned a response this app could not read.",
    LLMErrorKind.UNKNOWN: "The Groq request failed.",
}


def _message(kind: LLMErrorKind, model: str | None = None) -> str:
    base = _MESSAGES.get(kind, _MESSAGES[LLMErrorKind.UNKNOWN])
    if kind is LLMErrorKind.MODEL_UNAVAILABLE and model:
        return base.replace("The configured Groq model", f"The Groq model '{model}'")
    return base


class GroqConfigError(ValueError):
    """The configured model name is unusable. Raised at construction, because a
    blank or malformed model is a configuration mistake, not a runtime failure
    that should be rediscovered on every question."""


def validate_model(name: str) -> str:
    """Normalise and sanity-check a model identifier.

    Deliberately not an allow-list of known Groq models: that list changes on
    Groq's schedule, not this app's, and a stale copy would reject a model that
    works. What *can* be checked locally is checked here; whether the account
    can actually use the model is answered by Groq, and surfaces as
    MODEL_UNAVAILABLE with the name in the message.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise GroqConfigError("GROQ_MODEL is empty; set a Groq model identifier.")
    if any(character.isspace() for character in cleaned):
        raise GroqConfigError(
            f"GROQ_MODEL {cleaned!r} contains whitespace; it must be a single identifier."
        )
    return cleaned


class GroqClient(LLMClient):
    """Groq behind the shared LLMClient interface.

    Construction never raises on a missing key: the app must still start and
    still serve documents, history and transcription without one, so "no key"
    surfaces per-answer as a NOT_CONFIGURED LLMError.
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

    def __init__(self) -> None:
        self._client = None
        #: Last classified failure, for the Settings screen. Never a message
        #: from the provider and never anything key-derived -- just the kind.
        self.last_error_kind: LLMErrorKind | None = None
        # Fails fast and loudly on a misconfigured model rather than sending it
        # to Groq once per question.
        validate_model(settings.groq_model)

    @property
    def model_name(self) -> str:
        # Read through, not cached: PUT /settings can change the model at
        # runtime and the next request must use the new one.
        return settings.groq_model

    @property
    def configured(self) -> bool:
        return bool(settings.groq_api_key)

    # ------------------------------------------------------------- lifecycle

    def _ensure_client(self):
        """One client per process. A per-request client would discard the
        connection pool and put TLS setup on every answer."""
        if self._client is not None:
            return self._client
        if not settings.groq_api_key:
            raise LLMError(
                _message(LLMErrorKind.NOT_CONFIGURED),
                kind=LLMErrorKind.NOT_CONFIGURED,
            )
        try:
            from groq import AsyncGroq
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
            api_key=settings.groq_api_key,
            # The SDK already retries exactly the failures worth retrying
            # (connection errors, 408, 429, 5xx) with backoff, and never retries
            # a deterministic 400/401/404. Set explicitly so that contract is
            # visible here rather than inherited from an SDK default.
            max_retries=settings.groq_max_retries,
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
        kind = self.classify(exc)
        self.last_error_kind = kind
        return LLMError(
            _message(kind, self.model_name),
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
                ),
                timeout=settings.groq_timeout_seconds,
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
        self, prompt: str, *, model: str = "", timeout_seconds: float = 5.0
    ) -> str:
        """One short, non-streaming JSON completion.

        Serves the question-understanding layer (`StructuredCompleter`), which
        needs a small structured object rather than a coaching answer, on a
        much tighter budget than `groq_timeout_seconds` -- it sits on the
        realtime path between a finished question and its answer.

        JSON mode is safe to use here, unlike `stream_answer`: nothing streams
        this, so the provider buffering the whole object costs nothing.
        `model` falls back to the answer model when unset, so the setting can
        stay empty until there is a reason to split them.
        """
        client = self._ensure_client()
        started = time.monotonic()
        chosen = model or self.model_name
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=chosen,
                    messages=self._messages(prompt),
                    response_format={"type": "json_object"},
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
                ),
                timeout=settings.groq_timeout_seconds,
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
                "Groq returned an empty response.", kind=LLMErrorKind.MALFORMED
            )
        try:
            data = parse_answer_payload(text)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                _message(LLMErrorKind.MALFORMED), kind=LLMErrorKind.MALFORMED
            ) from exc
        try:
            return Answer.model_validate(data)
        except Exception as exc:
            self.last_error_kind = LLMErrorKind.MALFORMED
            raise LLMError(
                _message(LLMErrorKind.MALFORMED), kind=LLMErrorKind.MALFORMED
            ) from exc


def _chunk_text(chunk) -> str:
    """Pull the text delta out of a Groq streaming chunk, tolerating the empty
    and role-only chunks the API sends around the actual content."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None) or "" if delta is not None else ""


def build_llm_client() -> LLMClient:
    """Composition root for the LLM layer.

    One provider, so this is a constructor rather than a router. It stays a
    function because `app.core.deps` caches it and the settings API clears that
    cache to pick up a newly saved key without a restart.
    """
    client = GroqClient()
    logger.info(
        "llm_initialised provider=groq model=%s configured=%s",
        client.model_name,
        client.configured,
    )
    return client
