import asyncio
import json

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMError
from app.schemas.answer import Answer

logger = get_logger(__name__)


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured")
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate_answer(self, prompt: str) -> Answer:
        logger.info("llm_request_started model=%s", settings.gemini_model)
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                ),
                timeout=settings.gemini_timeout_seconds,
            )
        except TimeoutError as exc:
            raise LLMError("Gemini request timed out") from exc
        except Exception as exc:
            raise LLMError(f"Gemini API call failed: {exc}") from exc

        logger.info("llm_response_received")

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Gemini returned non-JSON output: {exc}") from exc

        try:
            return Answer.model_validate(data)
        except Exception as exc:
            raise LLMError(f"Gemini response did not match the answer schema: {exc}") from exc
