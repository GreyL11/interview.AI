import pytest

from app.documents.schemas import KnowledgeType
from app.intelligence.orchestrator import Orchestrator
from app.schemas.answer import Answer
from tests.fakes import FakeLLM

pytestmark = pytest.mark.asyncio

RESUME = "At Acme I built a Kafka streaming ingestion pipeline and led the batch migration."
TECHNICAL = "A B-tree index keeps keys sorted so range lookups stay fast in a database."


async def seed(service, text, knowledge_type, name="doc"):
    doc = await service.upload(f"{name}.txt", text.encode(), knowledge_type)
    await service.ingest(doc.document_id)
    return doc.document_id


async def test_personal_question_retrieves_context(service, retriever):
    await seed(service, RESUME, KnowledgeType.RESUME)
    llm = FakeLLM()
    orchestrator = Orchestrator(retriever=retriever, llm=llm)

    await orchestrator.handle("Tell me about a time you led a migration", session_id=None)

    assert "[RESUME]" in llm.prompts[0]
    assert "Kafka" in llm.prompts[0]


async def test_technical_question_does_not_retrieve_personal_context(service, retriever):
    await seed(service, RESUME, KnowledgeType.RESUME)
    llm = FakeLLM()
    orchestrator = Orchestrator(retriever=retriever, llm=llm)

    await orchestrator.handle("What is a database index?", session_id=None)

    # A technical answer must not silently acquire first-person resume claims.
    assert "[RESUME]" not in llm.prompts[0]
    assert "Acme" not in llm.prompts[0]


async def test_technical_documents_never_reach_a_personal_question(service, retriever):
    await seed(service, TECHNICAL, KnowledgeType.TECHNICAL, "notes")
    llm = FakeLLM()
    orchestrator = Orchestrator(retriever=retriever, llm=llm)

    await orchestrator.handle("Tell me about a time you tuned a database", session_id=None)

    # Reference material is knowledge, not lived experience.
    assert "[TECHNICAL]" not in llm.prompts[0]


async def test_context_found_suppresses_the_fabrication_warning(service, retriever):
    await seed(service, RESUME, KnowledgeType.RESUME)
    llm = FakeLLM(Answer(summary="I built a Kafka pipeline at Acme."))
    orchestrator = Orchestrator(retriever=retriever, llm=llm)

    response = await orchestrator.handle("Tell me about a project you built", session_id=None)
    assert response.answer.warnings == []


async def test_missing_context_triggers_the_fabrication_warning(retriever):
    """Same question, empty knowledge base — the claim is now unsupported."""
    llm = FakeLLM(Answer(summary="I built a Kafka pipeline at Acme."))
    orchestrator = Orchestrator(retriever=retriever, llm=llm)

    response = await orchestrator.handle("Tell me about a project you built", session_id=None)
    assert response.answer.warnings
    assert "no personal context" in response.answer.warnings[0]


async def test_empty_knowledge_base_still_answers(retriever):
    orchestrator = Orchestrator(retriever=retriever, llm=FakeLLM())
    response = await orchestrator.handle("How would you dedupe records?", session_id=None)
    assert response.answer.summary
