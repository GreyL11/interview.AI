"""Proves the provider-neutral architecture actually holds: the identical
Question Understanding and Answer Generation business logic, run unmodified
through two completely different provider clients, produces the identical
result. If either pipeline ever grew a branch on `provider_name`, one of
these would start failing.
"""

import json

import httpx
import pytest

from app.intelligence.orchestrator import Orchestrator
from app.llm.base import LLMClient
from app.llm.groq_client import GroqClient
from app.llm.openai_compatible import OpenAICompatibleClient
from app.llm.registry import ProviderConfig
from app.realtime.question_understanding import QuestionUnderstander
from app.retrieval.mock_retriever import MockRetriever

_ANSWER_JSON = (
    '{"summary": "Make it idempotent.", "key_points": ["dedupe on ingest"], '
    '"detailed_answer": "Full explanation.", "approach": null, "code": null, '
    '"complexity": null, "edge_cases": null, "sections": null, "warnings": []}'
)

_UNDERSTANDING_JSON = json.dumps({
    "intent": "coding",
    "relationship": "new_question",
    "topic": "duplicate records",
    "domain": "data engineering",
    "task": "explain a fix",
    "constraints": [],
    "requested_output": [],
    "entities": [],
    "needs_previous_context": False,
    "needs_previous_answer": False,
    "needs_previous_code": False,
    "needs_attachments": False,
    "verbosity": "default",
    "confidence": 0.9,
})


# ---------------------------------------------------------- Groq-side fakes


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs):
        return _Completion(self._content)


class _FakeGroqSdkClient:
    def __init__(self, content: str) -> None:
        self.chat = type("chat", (), {"completions": _FakeCompletions(content)})()


def _groq_client(content: str) -> GroqClient:
    client = GroqClient(ProviderConfig(provider="groq", model="openai/gpt-oss-120b", api_key="x"))
    client._client = _FakeGroqSdkClient(content)
    return client


# ---------------------------------------------------- OpenAI-compatible fakes


def _openai_compatible_client(content: str) -> OpenAICompatibleClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    config = ProviderConfig(
        provider="openai_compatible", model="gpt-5.6-terra", api_key="x",
        base_url="http://localhost:8000/v1",
    )
    client = OpenAICompatibleClient(config)
    client._client = httpx.AsyncClient(
        base_url="http://localhost:8000/v1", transport=httpx.MockTransport(handler)
    )
    return client


PROVIDER_BUILDERS = {
    "groq": lambda content: _groq_client(content),
    "openai_compatible": lambda content: _openai_compatible_client(content),
}


# --------------------------------------------------------- Answer Generation


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(PROVIDER_BUILDERS))
async def test_answer_generation_is_identical_across_providers(provider):
    client: LLMClient = PROVIDER_BUILDERS[provider](_ANSWER_JSON)
    orchestrator = Orchestrator(retriever=MockRetriever(), llm=client)

    response = await orchestrator.handle("How would you handle duplicate records?", "session-1")

    assert response.answer.summary == "Make it idempotent."
    assert response.answer.key_points == ["dedupe on ingest"]


@pytest.mark.asyncio
async def test_the_orchestrator_never_imports_a_specific_provider():
    """Regression guard for the abstraction itself: the module under test must
    depend only on LLMClient, never on GroqClient or OpenAICompatibleClient."""
    import app.intelligence.orchestrator as orchestrator_module

    source = open(orchestrator_module.__file__, encoding="utf-8").read()
    assert "app.llm.groq_client" not in source
    assert "app.llm.openai_compatible" not in source


# ----------------------------------------------------- Question Understanding


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", list(PROVIDER_BUILDERS))
async def test_question_understanding_is_identical_across_providers(provider):
    completer = PROVIDER_BUILDERS[provider](_UNDERSTANDING_JSON)
    understander = QuestionUnderstander(completer)

    result = await understander.understand(
        "Why does this happen with duplicate records?",
        session_id="session-1",
        question_id=1,
    )

    assert result.intent.value == "coding"
    assert result.relationship.value == "new_question"
    assert result.topic == "duplicate records"
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_question_understanding_never_imports_a_specific_provider():
    import app.realtime.question_understanding as qu_module

    source = open(qu_module.__file__, encoding="utf-8").read()
    assert "app.llm.groq_client" not in source
    assert "app.llm.openai_compatible" not in source
