from tests.fixtures import RESUME_TEXT


def upload(client, name="resume.txt", content=None, knowledge_type="RESUME"):
    body = RESUME_TEXT.encode() if content is None else content
    return client.post(
        "/documents",
        params={"filename": name, "knowledge_type": knowledge_type},
        content=body,
    )


def test_upload_returns_201(documents_client):
    response = upload(documents_client)
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"]
    assert body["filename"] == "resume.txt"
    assert body["status"] == "UPLOADED"


def test_upload_rejects_empty_file(documents_client):
    assert upload(documents_client, content=b"").status_code == 422


def test_upload_rejects_unsupported_type(documents_client):
    response = upload(documents_client, name="deck.pptx", content=b"data")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_bad_knowledge_type(documents_client):
    assert upload(documents_client, knowledge_type="NONSENSE").status_code == 422


def test_ingest_then_get(documents_client):
    document_id = upload(documents_client).json()["document_id"]

    ingest = documents_client.post(f"/documents/{document_id}/ingest")
    assert ingest.status_code == 200
    assert ingest.json()["status"] == "READY"
    assert ingest.json()["chunk_count"] > 0

    document = documents_client.get(f"/documents/{document_id}").json()
    assert document["status"] == "READY"
    assert document["chunk_count"] > 0
    assert document["ingested_at"]


def test_ingest_unknown_document_returns_400(documents_client):
    assert documents_client.post("/documents/nope/ingest").status_code == 400


def test_get_unknown_document_returns_400(documents_client):
    assert documents_client.get("/documents/nope").status_code == 400


def test_list_and_filter(documents_client):
    upload(documents_client, "a.txt", knowledge_type="RESUME")
    b = upload(documents_client, "b.txt", knowledge_type="TECHNICAL").json()["document_id"]
    documents_client.post(f"/documents/{b}/ingest")

    assert len(documents_client.get("/documents").json()) == 2
    assert len(documents_client.get("/documents?knowledge_type=TECHNICAL").json()) == 1
    assert len(documents_client.get("/documents?status=READY").json()) == 1
    assert documents_client.get("/documents?knowledge_type=RESUME&status=READY").json() == []


def test_delete(documents_client):
    document_id = upload(documents_client).json()["document_id"]
    documents_client.post(f"/documents/{document_id}/ingest")

    response = documents_client.delete(f"/documents/{document_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["chunks_removed"] > 0
    assert documents_client.get(f"/documents/{document_id}").status_code == 400


def test_delete_unknown_returns_400(documents_client):
    assert documents_client.delete("/documents/nope").status_code == 400


def test_failed_ingest_reports_error(documents_client):
    document_id = upload(documents_client, "broken.pdf", b"not a pdf").json()["document_id"]
    body = documents_client.post(f"/documents/{document_id}/ingest").json()

    assert body["status"] == "FAILED"
    assert body["error"]
    assert body["chunk_count"] == 0
