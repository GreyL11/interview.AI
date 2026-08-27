import pytest


@pytest.fixture
def sessions_client(fake_llm, database):
    from fastapi.testclient import TestClient

    from app.api.question import get_orchestrator
    from app.core.deps import get_session_repository
    from app.intelligence.orchestrator import Orchestrator
    from app.main import app
    from app.retrieval.mock_retriever import MockRetriever
    from app.storage.session_repository import SessionRepository

    repo = SessionRepository(database)
    app.dependency_overrides[get_orchestrator] = lambda: Orchestrator(
        retriever=MockRetriever(), llm=fake_llm
    )
    app.dependency_overrides[get_session_repository] = lambda: repo
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_create_session(sessions_client):
    response = sessions_client.post("/sessions", params={"title": "Mock interview"})
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"]
    assert body["status"] == "ACTIVE"
    assert body["title"] == "Mock interview"


def test_list_sessions(sessions_client):
    sessions_client.post("/sessions")
    sessions_client.post("/sessions")
    assert len(sessions_client.get("/sessions").json()) == 2


def test_get_session_detail(sessions_client):
    session_id = sessions_client.post("/sessions").json()["session_id"]
    detail = sessions_client.get(f"/sessions/{session_id}").json()

    assert detail["session"]["session_id"] == session_id
    assert detail["turns"] == []
    assert detail["transcript"] == []


def test_get_unknown_session_404(sessions_client):
    assert sessions_client.get("/sessions/nope").status_code == 404


def test_end_session(sessions_client):
    session_id = sessions_client.post("/sessions").json()["session_id"]
    body = sessions_client.post(f"/sessions/{session_id}/end").json()
    assert body["status"] == "ENDED"
    assert body["ended_at"]


def test_end_unknown_session_404(sessions_client):
    assert sessions_client.post("/sessions/nope/end").status_code == 404


def test_delete_session(sessions_client):
    session_id = sessions_client.post("/sessions").json()["session_id"]
    assert sessions_client.delete(f"/sessions/{session_id}").json()["deleted"] is True
    assert sessions_client.get(f"/sessions/{session_id}").status_code == 404


def test_delete_unknown_session_404(sessions_client):
    assert sessions_client.delete("/sessions/nope").status_code == 404
