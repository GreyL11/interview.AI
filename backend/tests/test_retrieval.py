import pytest

from app.documents.schemas import KnowledgeType

pytestmark = pytest.mark.asyncio

RESUME = "I built a Kafka streaming ingestion pipeline at Acme handling duplicate records."
TECHNICAL = "A B-tree index keeps keys sorted to speed up range lookups in a database."
DESSERT = "The lemon tart recipe needs blind baking and a citrus curd filling."


async def seed(service, pairs):
    ids = {}
    for name, text, knowledge_type in pairs:
        doc = await service.upload(f"{name}.txt", text.encode(), knowledge_type)
        await service.ingest(doc.document_id)
        ids[name] = doc.document_id
    return ids


async def test_retrieves_relevant_chunk(service, retriever):
    await seed(service, [("resume", RESUME, KnowledgeType.RESUME),
                         ("dessert", DESSERT, KnowledgeType.REFERENCE)])

    hits = await retriever.retrieve("Kafka streaming ingestion pipeline", min_similarity=0.0)
    assert hits
    assert "Kafka" in hits[0].text


async def test_empty_knowledge_base_returns_nothing(retriever):
    assert await retriever.retrieve("anything at all") == []


async def test_knowledge_type_filter_excludes_technical(service, retriever):
    await seed(service, [("resume", RESUME, KnowledgeType.RESUME),
                         ("notes", TECHNICAL, KnowledgeType.TECHNICAL)])

    personal = await retriever.retrieve(
        "database index range lookups",
        knowledge_types=[KnowledgeType.RESUME],
        min_similarity=0.0,
    )
    assert all(h.knowledge_type == KnowledgeType.RESUME for h in personal)


async def test_knowledge_type_filter_can_select_technical(service, retriever):
    await seed(service, [("resume", RESUME, KnowledgeType.RESUME),
                         ("notes", TECHNICAL, KnowledgeType.TECHNICAL)])

    technical = await retriever.retrieve(
        "B-tree index sorted keys",
        knowledge_types=[KnowledgeType.TECHNICAL],
        min_similarity=0.0,
    )
    assert technical
    assert all(h.knowledge_type == KnowledgeType.TECHNICAL for h in technical)


async def test_min_similarity_filters_weak_matches(service, retriever):
    await seed(service, [("dessert", DESSERT, KnowledgeType.REFERENCE)])

    assert await retriever.retrieve("Kubernetes autoscaling", min_similarity=0.99) == []
    assert await retriever.retrieve("lemon tart recipe", min_similarity=0.0)


async def test_top_k_limits_results(service, retriever):
    await seed(service, [
        (f"doc{i}", f"Document {i} about Kafka streaming pipelines and ingestion.", KnowledgeType.PROJECT)
        for i in range(6)
    ])
    hits = await retriever.retrieve("Kafka streaming", top_k=2, min_similarity=0.0)
    assert len(hits) <= 2


async def test_top_k_zero_returns_nothing(service, retriever):
    await seed(service, [("resume", RESUME, KnowledgeType.RESUME)])
    assert await retriever.retrieve("Kafka", top_k=0) == []


async def test_results_are_ordered_by_score(service, retriever):
    await seed(service, [
        ("a", RESUME, KnowledgeType.RESUME),
        ("b", TECHNICAL, KnowledgeType.RESUME),
        ("c", DESSERT, KnowledgeType.RESUME),
    ])
    hits = await retriever.retrieve("Kafka streaming ingestion", min_similarity=0.0)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_retrieved_chunk_carries_provenance(service, retriever):
    ids = await seed(service, [("resume", RESUME, KnowledgeType.RESUME)])
    hit = (await retriever.retrieve("Kafka pipeline", min_similarity=0.0))[0]

    assert hit.document_id == ids["resume"]
    assert hit.knowledge_type == KnowledgeType.RESUME
    assert hit.chunk_id
    assert "[RESUME]" in hit.as_context()
