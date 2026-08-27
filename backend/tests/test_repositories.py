from app.documents.schemas import (
    Chunk,
    Document,
    DocumentStatus,
    FileType,
    KnowledgeType,
    utcnow,
)


def make_document(document_id="doc-1", knowledge_type=KnowledgeType.RESUME, status=DocumentStatus.UPLOADED):
    return Document(
        document_id=document_id,
        filename=f"{document_id}.txt",
        file_type=FileType.TXT,
        knowledge_type=knowledge_type,
        title=document_id,
        source=f"/tmp/{document_id}.txt",
        created_at=utcnow(),
        status=status,
    )


def test_create_and_get(documents_repo):
    documents_repo.create(make_document())
    fetched = documents_repo.get("doc-1")
    assert fetched.document_id == "doc-1"
    assert fetched.knowledge_type == KnowledgeType.RESUME


def test_get_missing_returns_none(documents_repo):
    assert documents_repo.get("nope") is None


def test_list_filters(documents_repo):
    documents_repo.create(make_document("a", KnowledgeType.RESUME, DocumentStatus.READY))
    documents_repo.create(make_document("b", KnowledgeType.TECHNICAL, DocumentStatus.READY))
    documents_repo.create(make_document("c", KnowledgeType.TECHNICAL, DocumentStatus.FAILED))

    assert len(documents_repo.list()) == 3
    assert {d.document_id for d in documents_repo.list(knowledge_type=KnowledgeType.TECHNICAL)} == {"b", "c"}
    assert {d.document_id for d in documents_repo.list(status=DocumentStatus.READY)} == {"a", "b"}
    assert [d.document_id for d in documents_repo.list(
        knowledge_type=KnowledgeType.TECHNICAL, status=DocumentStatus.READY
    )] == ["b"]


def test_update_status_sets_ingested_at(documents_repo):
    documents_repo.create(make_document())
    documents_repo.update_status("doc-1", DocumentStatus.READY, chunk_count=4)
    doc = documents_repo.get("doc-1")
    assert doc.status == DocumentStatus.READY
    assert doc.chunk_count == 4
    assert doc.ingested_at is not None


def test_update_status_records_error(documents_repo):
    documents_repo.create(make_document())
    documents_repo.update_status("doc-1", DocumentStatus.FAILED, error="bad pdf")
    assert documents_repo.get("doc-1").error == "bad pdf"


def test_reset_stuck_processing(documents_repo):
    documents_repo.create(make_document("a", status=DocumentStatus.PROCESSING))
    documents_repo.create(make_document("b", status=DocumentStatus.READY))

    assert documents_repo.reset_stuck_processing() == 1
    assert documents_repo.get("a").status == DocumentStatus.FAILED
    assert documents_repo.get("b").status == DocumentStatus.READY


def test_chunks_crud(documents_repo, chunks_repo):
    documents_repo.create(make_document())
    chunks = [
        Chunk(chunk_id=f"c{i}", document_id="doc-1", chunk_index=i,
              text=f"chunk {i}", vector_id=100 + i, token_count=5,
              metadata={"knowledge_type": "RESUME"})
        for i in range(3)
    ]
    chunks_repo.create_many(chunks)

    assert [c.chunk_index for c in chunks_repo.get_by_document("doc-1")] == [0, 1, 2]
    assert chunks_repo.get_by_document("doc-1")[0].metadata["knowledge_type"] == "RESUME"
    assert sorted(chunks_repo.get_vector_ids("doc-1")) == [100, 101, 102]
    assert {c.chunk_id for c in chunks_repo.get_by_ids(["c0", "c2"])} == {"c0", "c2"}
    assert chunks_repo.get_by_ids([]) == []

    assert chunks_repo.delete_by_document("doc-1") == 3
    assert chunks_repo.get_by_document("doc-1") == []


def test_deleting_document_cascades_to_chunks(documents_repo, chunks_repo):
    documents_repo.create(make_document())
    chunks_repo.create_many([
        Chunk(chunk_id="c0", document_id="doc-1", chunk_index=0, text="t", vector_id=1)
    ])
    documents_repo.delete("doc-1")
    assert chunks_repo.get_by_document("doc-1") == []


def test_vector_ids_are_unique_and_never_reused(database):
    first = database.allocate_vector_ids(3)
    second = database.allocate_vector_ids(2)
    assert first == [1, 2, 3]
    assert second == [4, 5]
    assert not set(first) & set(second)


def test_resolve_vectors_requires_ready_status(documents_repo, chunks_repo):
    documents_repo.create(make_document("doc-1", status=DocumentStatus.PROCESSING))
    chunks_repo.create_many([
        Chunk(chunk_id="c0", document_id="doc-1", chunk_index=0, text="hidden", vector_id=1)
    ])
    # Mid-ingest chunks must be invisible even though the rows exist.
    assert chunks_repo.resolve_vectors([(1, 0.9)]) == []

    documents_repo.update_status("doc-1", DocumentStatus.READY)
    assert [c.text for c in chunks_repo.resolve_vectors([(1, 0.9)])] == ["hidden"]


def test_resolve_vectors_filters_knowledge_type(documents_repo, chunks_repo):
    documents_repo.create(make_document("r", KnowledgeType.RESUME, DocumentStatus.READY))
    documents_repo.create(make_document("t", KnowledgeType.TECHNICAL, DocumentStatus.READY))
    chunks_repo.create_many([
        Chunk(chunk_id="c1", document_id="r", chunk_index=0, text="my job", vector_id=1),
        Chunk(chunk_id="c2", document_id="t", chunk_index=0, text="b-trees", vector_id=2),
    ])

    personal = chunks_repo.resolve_vectors([(1, 0.9), (2, 0.8)], [KnowledgeType.RESUME])
    assert [c.text for c in personal] == ["my job"]


def test_resolve_vectors_orders_by_score(documents_repo, chunks_repo):
    documents_repo.create(make_document("doc-1", status=DocumentStatus.READY))
    chunks_repo.create_many([
        Chunk(chunk_id=f"c{i}", document_id="doc-1", chunk_index=i, text=f"t{i}", vector_id=i)
        for i in range(3)
    ])
    resolved = chunks_repo.resolve_vectors([(0, 0.1), (1, 0.9), (2, 0.5)])
    assert [c.score for c in resolved] == [0.9, 0.5, 0.1]


def test_resolve_vectors_skips_orphans(documents_repo, chunks_repo):
    documents_repo.create(make_document("doc-1", status=DocumentStatus.READY))
    chunks_repo.create_many([
        Chunk(chunk_id="c0", document_id="doc-1", chunk_index=0, text="real", vector_id=1)
    ])
    # Vector 999 has no chunk row — the join drops it instead of erroring.
    resolved = chunks_repo.resolve_vectors([(1, 0.9), (999, 0.95)])
    assert [c.text for c in resolved] == ["real"]
