import pytest

from app.documents.schemas import DocumentStatus, KnowledgeType
from app.documents.service import DocumentError
from tests.fixtures import RESUME_TEXT, write_docx, write_pdf

pytestmark = pytest.mark.asyncio


async def upload_text(service, name="resume.txt", text=RESUME_TEXT, kt=KnowledgeType.RESUME):
    return await service.upload(name, text.encode("utf-8"), kt)


async def test_ingest_success(service, documents_repo, chunks_repo, vector_store):
    document = await upload_text(service)
    assert document.status == DocumentStatus.UPLOADED

    result = await service.ingest(document.document_id)

    assert result.status == DocumentStatus.READY
    assert result.chunk_count > 0
    assert documents_repo.get(document.document_id).status == DocumentStatus.READY
    assert len(chunks_repo.get_by_document(document.document_id)) == result.chunk_count
    assert vector_store.size == result.chunk_count


async def test_every_chunk_gets_a_vector_id(service, chunks_repo):
    document = await upload_text(service)
    await service.ingest(document.document_id)
    chunks = chunks_repo.get_by_document(document.document_id)
    assert all(c.vector_id is not None for c in chunks)
    assert len({c.vector_id for c in chunks}) == len(chunks)


async def test_ingest_pdf(service, tmp_path):
    path = write_pdf(tmp_path / "cv.pdf", ["Jane Doe", "I built a Kafka pipeline."])
    document = await service.upload("cv.pdf", path.read_bytes(), KnowledgeType.RESUME)
    result = await service.ingest(document.document_id)
    assert result.status == DocumentStatus.READY


async def test_ingest_docx(service, tmp_path):
    path = write_docx(tmp_path / "cv.docx", ["Jane Doe", "I led a migration."])
    document = await service.upload("cv.docx", path.read_bytes(), KnowledgeType.RESUME)
    result = await service.ingest(document.document_id)
    assert result.status == DocumentStatus.READY


async def test_failed_ingest_leaves_no_partial_state(
    service, documents_repo, chunks_repo, vector_store
):
    """A scanned-style PDF with no text layer must fail cleanly."""
    document = await service.upload("scanned.pdf", write_pdf.__doc__.encode(), KnowledgeType.RESUME)
    # Overwrite with bytes that are not a valid PDF at all.
    result = await service.ingest(document.document_id)

    assert result.status == DocumentStatus.FAILED
    assert result.error
    assert result.chunk_count == 0

    stored = documents_repo.get(document.document_id)
    assert stored.status == DocumentStatus.FAILED
    assert stored.error
    assert chunks_repo.get_by_document(document.document_id) == []
    assert vector_store.size == 0


async def test_failed_ingest_is_invisible_to_retrieval(service, retriever, tmp_path):
    good = await upload_text(service)
    await service.ingest(good.document_id)

    bad = await service.upload("broken.pdf", b"not a pdf at all", KnowledgeType.RESUME)
    await service.ingest(bad.document_id)

    hits = await retriever.retrieve("Kafka ingestion pipeline", min_similarity=0.0)
    assert all(h.document_id != bad.document_id for h in hits)


async def test_missing_source_file_fails_gracefully(service, documents_repo, tmp_path):
    document = await upload_text(service)
    (tmp_path / "documents" / document.document_id / "resume.txt").unlink()

    result = await service.ingest(document.document_id)
    assert result.status == DocumentStatus.FAILED
    assert "missing" in (result.error or "").lower()


async def test_reingest_replaces_rather_than_duplicates(service, chunks_repo, vector_store):
    document = await upload_text(service)
    first = await service.ingest(document.document_id)
    second = await service.ingest(document.document_id)

    assert second.status == DocumentStatus.READY
    assert second.chunk_count == first.chunk_count
    assert len(chunks_repo.get_by_document(document.document_id)) == first.chunk_count
    assert vector_store.size == first.chunk_count


async def test_ingest_unknown_document_raises(service):
    with pytest.raises(DocumentError, match="Unknown document"):
        await service.ingest("does-not-exist")


async def test_unsupported_file_type_rejected(service):
    with pytest.raises(DocumentError, match="Unsupported file type"):
        await service.upload("notes.pptx", b"data", KnowledgeType.REFERENCE)


async def test_delete_removes_everything(service, documents_repo, chunks_repo, vector_store, tmp_path):
    document = await upload_text(service)
    await service.ingest(document.document_id)
    assert vector_store.size > 0

    result = await service.delete(document.document_id)

    assert result.deleted
    assert result.chunks_removed > 0
    assert result.vectors_removed == result.chunks_removed
    assert documents_repo.get(document.document_id) is None
    assert chunks_repo.get_by_document(document.document_id) == []
    assert vector_store.size == 0
    assert not (tmp_path / "documents" / document.document_id).exists()


async def test_delete_unknown_raises(service):
    with pytest.raises(DocumentError, match="Unknown document"):
        await service.delete("nope")


async def test_delete_one_document_leaves_others_searchable(service, retriever):
    keep = await upload_text(service, "keep.txt", "I built a Kafka streaming pipeline at Acme.")
    drop = await upload_text(service, "drop.txt", "I wrote a lemon tart recipe blog.")
    await service.ingest(keep.document_id)
    await service.ingest(drop.document_id)

    await service.delete(drop.document_id)

    hits = await retriever.retrieve("Kafka streaming pipeline", min_similarity=0.0)
    assert hits
    assert all(h.document_id == keep.document_id for h in hits)
