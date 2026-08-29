import asyncio
import random
import time
from collections.abc import AsyncIterator

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import elapsed_ms, log_metric
from app.llm.base import LLMClient, LLMError, LLMErrorKind
from app.llm.streaming import parse_answer_payload
from app.schemas.answer import Answer

logger = get_logger(__name__)


class GeminiClient(LLMClient):
    """Gemini behind the LLMClient interface.

    Construction is deliberately side-effect free and never raises: a live
    session must still start, transcribe, and record without a configured key.
    A missing key surfaces per-answer as answer.error, which the UI can show
    against that one turn, instead of refusing the WebSocket handshake and
    taking the whole session down.
    """

    def __init__(self) -> None:
        self._client = None

    _TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
    _TRANSIENT_ERROR_NAMES = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServerError",
        "TimeoutError",
        "TimeoutException",
    }

    def warmup(self) -> None:
        try:
            self._ensure_client()
        except LLMError:
            pass  # no key: still reported per-answer, as before

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is not configured. Add it in Setup to enable answers."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _model_candidates(self) -> list[str]:
        models = [settings.gemini_model]
        fallback_models = [
            model.strip()
            for model in settings.gemini_fallback_models.split(",")
            if model.strip()
        ]
        if not fallback_models:
            fallback = settings.gemini_fallback_model.strip()
            if fallback:
                fallback_models.append(fallback)
        for model in fallback_models:
            if model not in models:
                models.append(model)
        return models

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

    def _error_type(self, exc: Exception) -> str:
        return exc.__class__.__name__

    def _attempt_fields(
        self,
        *,
        model: str,
        attempt: int,
        fallback_index: int,
        exc: Exception | None = None,
        retry_delay_ms: int | None = None,
        before_first_text_token: bool | None = None,
    ) -> dict[str, object]:
        fields: dict[str, object] = {
            "model": model,
            "attempt": attempt,
            "fallback_index": fallback_index,
        }
        if exc is not None:
            fields["error_type"] = self._error_type(exc)
            fields["status_code"] = self._status_code(exc)
        if retry_delay_ms is not None:
            fields["retry_delay_ms"] = retry_delay_ms
        if before_first_text_token is not None:
            fields["before_first_text_token"] = before_first_text_token
        return fields

    def _log_attempt_started(self, event: str, *, model: str, attempt: int, fallback_index: int) -> None:
        log_metric(event, **self._attempt_fields(model=model, attempt=attempt, fallback_index=fallback_index))

    def _log_attempt_failed(
        self,
        event: str,
        *,
        model: str,
        attempt: int,
        fallback_index: int,
        exc: Exception,
        retry_delay_ms: int | None = None,
        before_first_text_token: bool | None = None,
    ) -> None:
        log_metric(
            event,
            **self._attempt_fields(
                model=model,
                attempt=attempt,
                fallback_index=fallback_index,
                exc=exc,
                retry_delay_ms=retry_delay_ms,
                before_first_text_token=before_first_text_token,
            ),
        )

    def _retry_delay_ms(self, retry_index: int) -> int:
        return int(self._retry_delay_seconds(retry_index) * 1000)

    def _request_failure_message(self, exc: Exception, *, streamed_text: bool) -> str:
        if self._is_transient_error(exc):
            return "AI service is temporarily unavailable. Please try again."
        if streamed_text:
            return "AI service request failed during streaming."
        return "AI service request failed."

    def _prompt_stats(self, prompt: str) -> dict[str, int]:
        return {
            "prompt_chars": len(prompt),
            "prompt_lines": prompt.count("\n") + 1 if prompt else 0,
        }

    def _log_prep_metrics(
        self, label: str, started: float, prompt: str, *, model: str, attempt: int
    ) -> dict[str, int]:
        stats = self._prompt_stats(prompt)
        log_metric(
            f"{label}_request_prep_completed",
            model=model,
            attempt=attempt,
            duration_ms=elapsed_ms(started, time.monotonic()),
            **stats,
        )
        return stats

    def _status_code_from_error(self, exc: Exception) -> int | None:
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

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        status = self._status_code_from_error(exc)
        if status in self._TRANSIENT_STATUS_CODES:
            return True
        return exc.__class__.__name__ in self._TRANSIENT_ERROR_NAMES

    def _is_rate_limit(self, exc: Exception) -> bool:
        return self._status_code_from_error(exc) == 429

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        """Honour the provider's own Retry-After hint when it sends one."""
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
        """Map an SDK exception onto the router's shared taxonomy. Rate limit
        is checked before transient: 429 is in _TRANSIENT_STATUS_CODES, but
        the router must treat it as "cool this provider down", not "retry"."""
        if self._is_rate_limit(exc):
            return LLMErrorKind.RATE_LIMIT
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return LLMErrorKind.TIMEOUT
        if self._status_code_from_error(exc) in (401, 403):
            return LLMErrorKind.AUTH
        if self._is_transient_error(exc):
            return LLMErrorKind.TRANSIENT
        return LLMErrorKind.UNKNOWN

    def _as_llm_error(self, exc: Exception, message: str) -> LLMError:
        return LLMError(
            message,
            kind=self._classify(exc),
            retry_after_seconds=self._retry_after_seconds(exc),
        )

    def _should_retry_locally(self, exc: Exception) -> bool:
        """Retry inside this provider only for failures a retry can plausibly
        fix. A 429 cannot be: burning the local retry budget on it just delays
        the failover that will actually produce an answer."""
        return self._is_transient_error(exc) and not self._is_rate_limit(exc)

    def _retry_delay_seconds(self, retry_index: int) -> float:
        base = settings.gemini_retry_initial_delay_seconds * (2**retry_index)
        capped = min(base, settings.gemini_retry_max_delay_seconds)
        return capped * random.uniform(0.8, 1.2)

    async def _sleep_before_retry(self, *, label: str, model: str, attempt: int, exc: Exception) -> None:
        if attempt >= max(1, settings.gemini_retry_max_attempts):
            return
        delay_seconds = self._retry_delay_seconds(attempt - 1)
        logger.warning(
            "%s_retrying model=%s attempt=%d delay_ms=%d error=%s",
            label,
            model,
            attempt + 1,
            int(delay_seconds * 1000),
            exc,
        )
        await asyncio.sleep(delay_seconds)

    async def _stream_chunks(
        self,
        prompt: str,
        *,
        label: str,
        timings: dict[str, int] | None = None,
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        max_attempts = max(1, settings.gemini_retry_max_attempts)
        models = self._model_candidates()
        last_exc: Exception | None = None

        for fallback_index, model_name in enumerate(models):
            if fallback_index > 0:
                log_metric(
                    "llm_fallback_started",
                    **self._attempt_fields(
                        model=model_name, attempt=1, fallback_index=fallback_index
                    ),
                )
            model_failed: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                self._log_attempt_started(
                    "llm_model_attempt_started",
                    model=model_name,
                    attempt=attempt,
                    fallback_index=fallback_index,
                )
                prep_started = time.monotonic()
                log_metric(
                    f"{label}_request_prep_started",
                    model=model_name,
                    attempt=attempt,
                )
                config = self._config()
                stats = self._log_prep_metrics(
                    label, prep_started, prompt, model=model_name, attempt=attempt
                )
                if timings is not None and fallback_index == 0 and attempt == 1:
                    timings["request_prep_ms"] = elapsed_ms(
                        prep_started, time.monotonic()
                    )
                    timings.update(stats)

                api_started = time.monotonic()
                try:
                    stream = await asyncio.wait_for(
                        client.aio.models.generate_content_stream(
                            model=model_name, contents=prompt, config=config
                        ),
                        timeout=settings.gemini_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    self._log_attempt_failed(
                        "llm_model_attempt_failed",
                        model=model_name,
                        attempt=attempt,
                        fallback_index=fallback_index,
                        exc=exc,
                        before_first_text_token=True,
                    )
                    if not self._should_retry_locally(exc):
                        log_metric(
                            "llm_request_failed",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                before_first_text_token=True,
                            ),
                        )
                        raise self._as_llm_error(
                            exc, self._request_failure_message(exc, streamed_text=False)
                        ) from exc
                    if attempt < max_attempts:
                        delay_seconds = self._retry_delay_seconds(attempt - 1)
                        log_metric(
                            "llm_retry_scheduled",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                retry_delay_ms=int(delay_seconds * 1000),
                                before_first_text_token=True,
                            ),
                        )
                        await asyncio.sleep(delay_seconds)
                        continue
                    model_failed = exc
                    break

                stream_created_ms = elapsed_ms(api_started, time.monotonic())
                log_metric(
                    f"{label}_stream_created",
                    model=model_name,
                    attempt=attempt,
                    duration_ms=stream_created_ms,
                )
                if timings is not None and "stream_created_ms" not in timings:
                    timings["stream_created_ms"] = stream_created_ms

                network_started = time.monotonic()
                log_metric(
                    f"{label}_network_request_started",
                    model=model_name,
                    attempt=attempt,
                )

                first_response_seen = False
                first_text_seen = False
                yielded_any = False
                try:
                    async for chunk in stream:
                        if not first_response_seen:
                            first_response_seen = True
                            first_response_ms = elapsed_ms(
                                network_started, time.monotonic()
                            )
                            log_metric(
                                f"{label}_first_response_received",
                                model=model_name,
                                attempt=attempt,
                                duration_ms=first_response_ms,
                            )
                            log_metric(
                                f"{label}_first_chunk_received",
                                model=model_name,
                                attempt=attempt,
                                duration_ms=first_response_ms,
                            )
                            if timings is not None and "first_response_ms" not in timings:
                                timings["first_response_ms"] = first_response_ms
                            if timings is not None and "first_chunk_ms" not in timings:
                                timings["first_chunk_ms"] = first_response_ms
                        if chunk.text and not first_text_seen:
                            first_text_seen = True
                            first_text_ms = elapsed_ms(
                                network_started, time.monotonic()
                            )
                            log_metric(
                                f"{label}_first_text_token_received",
                                model=model_name,
                                attempt=attempt,
                                duration_ms=first_text_ms,
                            )
                            if timings is not None and "first_text_token_ms" not in timings:
                                timings["first_text_token_ms"] = first_text_ms
                        if chunk.text:
                            yielded_any = True
                            yield chunk.text
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    before_first_text = not first_text_seen
                    self._log_attempt_failed(
                        "llm_model_attempt_failed",
                        model=model_name,
                        attempt=attempt,
                        fallback_index=fallback_index,
                        exc=exc,
                        before_first_text_token=before_first_text,
                    )
                    if yielded_any or not self._should_retry_locally(exc):
                        log_metric(
                            "llm_request_failed",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                before_first_text_token=before_first_text,
                            ),
                        )
                        raise self._as_llm_error(
                            exc,
                            self._request_failure_message(
                                exc, streamed_text=yielded_any or first_text_seen
                            ),
                        ) from exc
                    if attempt < max_attempts:
                        delay_seconds = self._retry_delay_seconds(attempt - 1)
                        log_metric(
                            "llm_retry_scheduled",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                retry_delay_ms=int(delay_seconds * 1000),
                                before_first_text_token=True,
                            ),
                        )
                        await asyncio.sleep(delay_seconds)
                        continue
                    model_failed = exc
                    break
                else:
                    if fallback_index > 0:
                        log_metric(
                            "llm_fallback_succeeded",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                            ),
                        )
                    return

            if model_failed is not None and fallback_index > 0:
                log_metric(
                    "llm_fallback_failed",
                    **self._attempt_fields(
                        model=model_name,
                        attempt=max_attempts,
                        fallback_index=fallback_index,
                        exc=model_failed,
                        before_first_text_token=True,
                    ),
                )
            if fallback_index + 1 < len(models):
                logger.warning(
                    "gemini_fallback_model_selected primary=%s fallback=%s",
                    models[fallback_index],
                    models[fallback_index + 1],
                )

        if last_exc is not None:
            log_metric(
                "llm_request_failed",
                **self._attempt_fields(
                    model=models[-1],
                    attempt=max_attempts,
                    fallback_index=len(models) - 1,
                    exc=last_exc,
                    before_first_text_token=True,
                ),
            )
            if self._is_transient_error(last_exc):
                raise self._as_llm_error(
                    last_exc, "AI service is temporarily unavailable. Please try again."
                ) from last_exc
            raise self._as_llm_error(last_exc, "AI service request failed.") from last_exc
        raise LLMError("AI service is temporarily unavailable. Please try again.")

    async def benchmark_stream_latency(
        self, app_prompt: str, minimal_prompt: str
    ) -> dict[str, dict[str, int]]:
        """Compare the normal app prompt with a bare prompt against the same model."""
        results: dict[str, dict[str, int]] = {}
        for label, prompt in (("app", app_prompt), ("minimal", minimal_prompt)):
            timings: dict[str, int] = {}
            async for _ in self._stream_chunks(prompt, label=label, timings=timings):
                pass
            results[label] = timings
        return results

    async def generate_answer(self, prompt: str) -> Answer:
        logger.info("llm_request_started model=%s", settings.gemini_model)
        client = self._ensure_client()
        max_attempts = max(1, settings.gemini_retry_max_attempts)
        models = self._model_candidates()
        last_exc: Exception | None = None

        for fallback_index, model_name in enumerate(models):
            if fallback_index > 0:
                log_metric(
                    "llm_fallback_started",
                    **self._attempt_fields(
                        model=model_name, attempt=1, fallback_index=fallback_index
                    ),
                )
            model_failed: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                self._log_attempt_started(
                    "llm_model_attempt_started",
                    model=model_name,
                    attempt=attempt,
                    fallback_index=fallback_index,
                )
                prep_started = time.monotonic()
                log_metric(
                    "llm_request_prep_started",
                    model=model_name,
                    attempt=attempt,
                )
                config = self._config()
                self._log_prep_metrics(
                    "llm", prep_started, prompt, model=model_name, attempt=attempt
                )
                api_started = time.monotonic()
                log_metric("llm_api_call_started", model=model_name, attempt=attempt)
                try:
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=model_name, contents=prompt, config=config
                        ),
                        timeout=settings.gemini_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    last_exc = exc
                    self._log_attempt_failed(
                        "llm_model_attempt_failed",
                        model=model_name,
                        attempt=attempt,
                        fallback_index=fallback_index,
                        exc=exc,
                        before_first_text_token=True,
                    )
                    if attempt < max_attempts:
                        delay_seconds = self._retry_delay_seconds(attempt - 1)
                        log_metric(
                            "llm_retry_scheduled",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                retry_delay_ms=int(delay_seconds * 1000),
                                before_first_text_token=True,
                            ),
                        )
                        await asyncio.sleep(delay_seconds)
                        continue
                    model_failed = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    self._log_attempt_failed(
                        "llm_model_attempt_failed",
                        model=model_name,
                        attempt=attempt,
                        fallback_index=fallback_index,
                        exc=exc,
                        before_first_text_token=True,
                    )
                    if not self._should_retry_locally(exc):
                        log_metric(
                            "llm_request_failed",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                before_first_text_token=True,
                            ),
                        )
                        raise self._as_llm_error(
                            exc, self._request_failure_message(exc, streamed_text=False)
                        ) from exc
                    if attempt < max_attempts:
                        delay_seconds = self._retry_delay_seconds(attempt - 1)
                        log_metric(
                            "llm_retry_scheduled",
                            **self._attempt_fields(
                                model=model_name,
                                attempt=attempt,
                                fallback_index=fallback_index,
                                exc=exc,
                                retry_delay_ms=int(delay_seconds * 1000),
                                before_first_text_token=True,
                            ),
                        )
                        await asyncio.sleep(delay_seconds)
                        continue
                    model_failed = exc
                    break

                duration_ms = elapsed_ms(api_started, time.monotonic())
                log_metric(
                    "llm_http_response_received",
                    model=model_name,
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
                text = response.text or ""
                if text.strip():
                    log_metric(
                        "llm_first_text_token_received",
                        model=model_name,
                        attempt=attempt,
                        duration_ms=elapsed_ms(api_started, time.monotonic()),
                    )
                logger.info("llm_response_received")
                if fallback_index > 0:
                    log_metric(
                        "llm_fallback_succeeded",
                        **self._attempt_fields(
                            model=model_name,
                            attempt=attempt,
                            fallback_index=fallback_index,
                        ),
                    )
                return _to_answer(text)

            if model_failed is not None and fallback_index > 0:
                log_metric(
                    "llm_fallback_failed",
                    **self._attempt_fields(
                        model=model_name,
                        attempt=max_attempts,
                        fallback_index=fallback_index,
                        exc=model_failed,
                        before_first_text_token=True,
                    ),
                )
            if fallback_index + 1 < len(models):
                logger.warning(
                    "gemini_fallback_model_selected primary=%s fallback=%s",
                    models[fallback_index],
                    models[fallback_index + 1],
                )

        if last_exc is not None:
            log_metric(
                "llm_request_failed",
                **self._attempt_fields(
                    model=models[-1],
                    attempt=max_attempts,
                    fallback_index=len(models) - 1,
                    exc=last_exc,
                    before_first_text_token=True,
                ),
            )
            if self._is_transient_error(last_exc):
                raise self._as_llm_error(
                    last_exc, "AI service is temporarily unavailable. Please try again."
                ) from last_exc
            raise self._as_llm_error(last_exc, "AI service request failed.") from last_exc

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        started = time.monotonic()
        logger.info("llm_stream_started model=%s", settings.gemini_model)
        try:
            async for chunk in self._stream_chunks(prompt, label="llm"):
                yield chunk
        except asyncio.CancelledError:
            # A superseded question cancels this task; that is normal control
            # flow during a live session, not an error.
            logger.info("llm_stream_cancelled")
            raise
        log_metric(
            "llm_stream_completed",
            model=settings.gemini_model,
            duration_ms=elapsed_ms(started, time.monotonic()),
        )
        logger.info("llm_stream_completed")


def _to_answer(text: str) -> Answer:
    text = (text or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response")
    try:
        data = parse_answer_payload(text)
    except Exception as exc:
        raise LLMError(f"Gemini returned non-JSON output: {exc}") from exc
    try:
        return Answer.model_validate(data)
    except Exception as exc:
        raise LLMError(f"Gemini response did not match the answer schema: {exc}") from exc
