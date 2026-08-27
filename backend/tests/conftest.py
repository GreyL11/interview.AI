import pytest
from fastapi.testclient import TestClient

from app.api.question import get_orchestrator
from app.intelligence.orchestrator import Orchestrator
from app.llm.base import LLMClient
from app.main import app
from app.retrieval.mock_retriever import MockRetriever
from app.schemas.answer import Answer


class FakeLLM(LLMClient):
    async def generate_answer(self, prompt: str) -> Answer:
        return Answer(
            summary="Make the pipeline idempotent.",
            key_points=["Identify the duplicate key", "Deduplicate on ingest"],
            detailed_answer="Full explanation.",
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_orchestrator] = lambda: Orchestrator(
        retriever=MockRetriever(), llm=FakeLLM()
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
