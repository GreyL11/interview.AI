import asyncio
from collections.abc import AsyncIterator

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMError
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

        return types.GenerateContentConfig(response_mime_type="application/json")

    async def generate_answer(self, prompt: str) -> Answer:
        client = self._ensure_client()
        logger.info("llm_request_started model=%s", settings.gemini_model)
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=settings.gemini_model, contents=prompt, config=self._config()
                ),
                timeout=settings.gemini_timeout_seconds,
            )
        except TimeoutError as exc:
            raise LLMError("Gemini request timed out") from exc
        except Exception as exc:
            raise LLMError(f"Gemini API call failed: {exc}") from exc

        logger.info("llm_response_received")
        return _to_answer(response.text or "")

    async def stream_answer(self, prompt: str) -> AsyncIterator[str]:
        client = self._ensure_client()
        logger.info("llm_stream_started model=%s", settings.gemini_model)
        try:
            stream = await client.aio.models.generate_content_stream(
                model=settings.gemini_model, contents=prompt, config=self._config()
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except asyncio.CancelledError:
            # A superseded question cancels this task; that is normal control
            # flow during a live session, not an error.
            logger.info("llm_stream_cancelled")
            raise
        except Exception as exc:
            raise LLMError(f"Gemini streaming call failed: {exc}") from exc
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
