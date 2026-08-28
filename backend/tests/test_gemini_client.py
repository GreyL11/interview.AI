import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.core.config import settings
from app.llm.base import LLMError
from app.llm.gemini_client import GeminiClient

pytestmark = pytest.mark.asyncio


class FakeChunk:
    def __init__(self, text: str):
        self.text = text


class FakeStream:
    def __init__(self, chunks: list[FakeChunk]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FakeModels:
    def __init__(self) -> None:
        self.stream_calls: list[tuple[str, str, object]] = []
        self.generate_calls: list[tuple[str, str, object]] = []
        self.stream_results: list[object] = []
        self.generate_results: list[object] = []

    async def generate_content_stream(self, *, model, contents, config):
        self.stream_calls.append((model, contents, config))
        if self.stream_results:
            result = self.stream_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return FakeStream([FakeChunk(""), FakeChunk('{"summary":"ok"}')])

    async def generate_content(self, *, model, contents, config):
        self.generate_calls.append((model, contents, config))
        if self.generate_results:
            result = self.generate_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return SimpleNamespace(text='{"summary":"ok"}')


class FakeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.aio = SimpleNamespace(models=install_fake_google.state.models)


class FakeAutomaticFunctionCallingConfig:
    def __init__(self, disable: bool = False):
        self.disable = disable


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTransientError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"{status_code} transient")
        self.status_code = status_code


def install_fake_google(monkeypatch):
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    types = ModuleType("google.genai.types")
    state = SimpleNamespace(models=FakeModels())

    genai.Client = FakeClient
    types.AutomaticFunctionCallingConfig = FakeAutomaticFunctionCallingConfig
    types.GenerateContentConfig = FakeGenerateContentConfig
    google.genai = genai
    genai.types = types
    genai.state = state
    install_fake_google.state = state

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    return genai


async def test_stream_answer_disables_afc_and_emits_latency_stages(monkeypatch):
    genai = install_fake_google(monkeypatch)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 1.0)

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.llm.gemini_client.log_metric",
        lambda event, **fields: events.append((event, fields)),
    )

    client = GeminiClient()
    chunks = []
    async for chunk in client.stream_answer("Explain indexing"):
        chunks.append(chunk)

    assert chunks == ['{"summary":"ok"}']
    call = client._client.aio.models.stream_calls[0]
    assert call[0] == "gemini-test"
    assert call[1] == "Explain indexing"
    assert call[2].response_mime_type == "application/json"
    assert call[2].automatic_function_calling.disable is True

    event_names = [name for name, _ in events]
    assert "llm_model_attempt_started" in event_names
    assert "llm_request_prep_started" in event_names
    assert "llm_request_prep_completed" in event_names
    assert "llm_stream_created" in event_names
    assert "llm_network_request_started" in event_names
    assert "llm_first_response_received" in event_names
    assert "llm_first_chunk_received" in event_names
    assert "llm_first_text_token_received" in event_names
    assert "llm_stream_completed" in event_names


async def test_stream_answer_retries_transient_errors_before_succeeding(monkeypatch):
    genai = install_fake_google(monkeypatch)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 1.0)
    monkeypatch.setattr(settings, "gemini_retry_max_attempts", 2)

    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("app.llm.gemini_client.asyncio.sleep", fake_sleep)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.llm.gemini_client.log_metric",
        lambda event, **fields: events.append((event, fields)),
    )
    genai.state.models.stream_results = [
        FakeTransientError(503),
        FakeStream([FakeChunk(""), FakeChunk('{"summary":"ok"}')]),
    ]

    client = GeminiClient()
    chunks = []
    async for chunk in client.stream_answer("Explain indexing"):
        chunks.append(chunk)

    assert chunks == ['{"summary":"ok"}']
    assert len(genai.state.models.stream_calls) == 2
    assert len(sleeps) == 1
    event_names = [name for name, _ in events]
    assert "llm_model_attempt_failed" in event_names
    assert "llm_retry_scheduled" in event_names


async def test_benchmark_stream_latency_compares_app_and_minimal_prompts(monkeypatch):
    install_fake_google(monkeypatch)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 1.0)

    client = GeminiClient()
    result = await client.benchmark_stream_latency("APP PROMPT", "MINIMAL PROMPT")

    assert set(result) == {"app", "minimal"}
    assert result["app"]["prompt_chars"] == len("APP PROMPT")
    assert result["minimal"]["prompt_chars"] == len("MINIMAL PROMPT")
    assert "stream_created_ms" in result["app"]
    assert "first_response_ms" in result["app"]
    assert "first_text_token_ms" in result["app"]
    assert "first_text_token_ms" in result["minimal"]


async def test_generate_answer_uses_fallback_model_after_transient_failure(monkeypatch):
    genai = install_fake_google(monkeypatch)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-primary")
    monkeypatch.setattr(settings, "gemini_fallback_models", "gemini-fallback")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 1.0)
    monkeypatch.setattr(settings, "gemini_retry_max_attempts", 1)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.llm.gemini_client.log_metric",
        lambda event, **fields: events.append((event, fields)),
    )
    genai.state.models.generate_results = [
        FakeTransientError(503),
        SimpleNamespace(text='{"summary":"ok"}'),
    ]

    client = GeminiClient()
    answer = await client.generate_answer("Explain indexing")

    assert answer.summary == "ok"
    assert [call[0] for call in genai.state.models.generate_calls] == [
        "gemini-primary",
        "gemini-fallback",
    ]
    event_names = [name for name, _ in events]
    assert "llm_fallback_started" in event_names
    assert "llm_fallback_succeeded" in event_names


async def test_generate_answer_sanitizes_transient_failure_message(monkeypatch):
    genai = install_fake_google(monkeypatch)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 1.0)
    monkeypatch.setattr(settings, "gemini_retry_max_attempts", 1)
    monkeypatch.setattr(
        "app.llm.gemini_client.log_metric",
        lambda *args, **kwargs: None,
    )

    client = GeminiClient()
    genai.state.models.generate_results = [FakeTransientError(503)]

    with pytest.raises(LLMError) as exc_info:
        await client.generate_answer("Explain indexing")

    assert "503" not in str(exc_info.value)
