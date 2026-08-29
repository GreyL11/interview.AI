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


class GroqClient(LLMClient):
    """Groq behind the same LLMClient interface as Gemini.

    Deliberately thin. Retry, fallback, cooldown and provider choice all live
    in the router, so this class only does what is genuinely Groq-specific:
    open a client, stream JSON text, and translate Groq's exceptions into the
    shared LLMErrorKind taxonomy. Answer parsing and validation stay in the
    existing shared path -- this yields raw text exactly like Gemini does, so
    `extract_partial_summary` / `parse_answer_payload` need no provider
    awareness at all.

    Like GeminiClient, construction never raises: a missing key surfaces
    per-answer rather than taking the session down at startup.
    """

    provider_name = "groq"

    _TRANSIENT_STATUS_CODES = {408, 500, 502, 503, 504}
    _TRANSIENT_ERROR_NAMES = {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "TimeoutException",
    }
    _AUTH_ERROR_NAMES = {"AuthenticationError", "PermissionDeniedError"}

    def __init__(self) -> None:
        self._client = None

    @property
    def model_name(self) -> str:
        return settings.groq_model

    def _ensure_client(self):
        """One client, reused across requests -- a per-request client would
        throw away the connection pool and add TLS setup to every answer."""
        if self._client is not None:
            return self._client
        if not settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY is not configured. Add it in Setup to enable answers.",
                kind=LLMErrorKind.AUTH,
            )
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise LLMError(
                f"The groq package is not installed: {exc}. "
                "Install it to enable the Groq provider.",
                kind=LLMErrorKind.AUTH,
            ) from exc

        self._client = AsyncGroq(api_key=settings.groq_api_key)
        return self._client

    def warmup(self) -> None:
        # Measured on this machine: `import groq` ~1.8s, AsyncGroq() ~0.6s.
        # Lazily that lands on the first question of a session -- the worst
        # possible moment for an interview copilot.
        try:
            self._ensure_client()
        except LLMError:
            pass  # no key / package: still reported per-answer, as before

    # ------------------------------------------------------- error handling

    def _status_code(self, exc: Exception) -> int | None:
        for attr in ("status_code", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        if response is not None:
            for attr in ("status_code", "status"):
                value = getattr(response, attr, None)
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

    def _classify(self, exc: Exception) -> LLMErrorKind:
        name = exc.__class__.__name__
        status = self._status_code(exc)
        if status == 429 or name == "RateLimitError":
            return LLMErrorKind.RATE_LIMIT
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or name == "APITimeoutError":
            return LLMErrorKind.TIMEOUT
        if status in (401, 403) or name in self._AUTH_ERROR_NAMES:
            return LLMErrorKind.AUTH
        if status in self._TRANSIENT_STATUS_CODES or name in self._TRANSIENT_ERROR_NAMES:
            return LLMErrorKind.TRANSIENT
        return LLMErrorKind.UNKNOWN

    def _as_llm_error(self, exc: Exception) -> LLMError:
        kind = self._classify(exc)
        message = (
            "AI service is temporarily unavailable. Please try again."
            if kind in (LLMErrorKind.RATE_LIMIT, LLMErrorKind.TRANSIENT, LLMErrorKind.TIMEOUT)
            else "AI service request failed."
        )
        return LLMError(message, kind=kind, retry_after_seconds=self._retry_after_seconds(exc))

    # ------------------------------------------------------------ requests

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        # The prompt already carries the system instruction and schema hint,
        # built once by app.llm.prompts for every provider.
        return [{"role": "user", "content": prompt}]

    async def generate_answer(self, prompt: str) -> Answer:
        client = self._ensure_client()
        started = time.monotonic()
        log_metric("llm_api_call_started", provider=self.provider_name, model=self.model_name)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.groq_model,
                    messages=self._messages(prompt),
                    response_format={"type": "json_object"},
                ),
                timeout=settings.groq_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._as_llm_error(exc) from exc

        log_metric(
            "llm_http_response_received",
            provider=self.provider_name,
            model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        return _to_answer(response.choices[0].message.content or "")

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        client = self._ensure_client()
        started = time.monotonic()
        log_metric(
            "llm_stream_started", provider=self.provider_name, model=self.model_name
        )
        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.groq_model,
                    messages=self._messages(prompt),
                    # No response_format={"type": "json_object"} here on
                    # purpose: Groq's JSON mode buffers the whole object and
                    # delivers it in a single content chunk, which kills the
                    # partial-summary streaming the UI depends on. The prompt
                    # already demands raw JSON, and parse_answer_payload
                    # tolerates a stray markdown fence. JSON mode is still used
                    # in the non-streaming generate_answer above, where
                    # buffering costs nothing.
                    stream=True,
                ),
                timeout=settings.groq_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._as_llm_error(exc) from exc

        first_text_seen = False
        try:
            async for chunk in stream:
                text = _chunk_text(chunk)
                if not text:
                    continue
                if not first_text_seen:
                    first_text_seen = True
                    log_metric(
                        "llm_first_text_token_received",
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
            raise self._as_llm_error(exc) from exc

        log_metric(
            "llm_stream_completed",
            provider=self.provider_name,
            model=self.model_name,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )


def _chunk_text(chunk) -> str:
    """Pull the text delta out of a Groq streaming chunk, tolerating the
    empty/rolechunks the API sends around the actual content."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None) or "" if delta is not None else ""


def _to_answer(text: str) -> Answer:
    text = (text or "").strip()
    if not text:
        raise LLMError("Groq returned an empty response")
    try:
        data = parse_answer_payload(text)
    except Exception as exc:
        raise LLMError(f"Groq returned non-JSON output: {exc}") from exc
    try:
        return Answer.model_validate(data)
    except Exception as exc:
        raise LLMError(f"Groq response did not match the answer schema: {exc}") from exc
