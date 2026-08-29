import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import deps
from app.documents.schemas import utcnow
from app.main import app
from app.realtime.events import EventType
from app.realtime.manager import session_manager
from app.sessions.schemas import Session
from app.storage.session_repository import SessionRepository
from tests.fakes import SlowStreamingLLM


@pytest.fixture
def ws_client(database, retriever, monkeypatch):
    """A TestClient with the realtime stack wired to tmp_path storage and a
    fake streaming LLM. Overrides the composition root directly because the
    WebSocket route resolves its collaborators at connect time, not via
    FastAPI's dependency injection."""
    repo = SessionRepository(database)
    llm = SlowStreamingLLM(chunk_delay=0)

    monkeypatch.setattr(deps, "get_session_repository", lambda: repo)
    monkeypatch.setattr(deps, "get_retriever", lambda: retriever)
    monkeypatch.setattr(deps, "get_llm_client", lambda: llm)
    monkeypatch.setattr(deps, "get_summarizer", lambda: None)
    monkeypatch.setattr(
        deps, "get_session_memory", lambda: __import__(
            "app.memory.session_memory", fromlist=["InMemorySessionMemory"]
        ).InMemorySessionMemory()
    )

    import app.api.ws as ws_module

    for name in ("get_session_repository", "get_retriever", "get_llm_client",
                 "get_summarizer", "get_session_memory"):
        monkeypatch.setattr(ws_module, name, getattr(deps, name))

    with TestClient(app) as client:
        yield client, repo

    session_manager._sessions.clear()


def read_until(socket, wanted, limit=40):
    """Collect events until `wanted` arrives. Returns everything seen."""
    seen = []
    for _ in range(limit):
        event = json.loads(socket.receive_text())
        seen.append(event)
        if event["type"] == wanted:
            return seen
    raise AssertionError(f"never saw {wanted}; got {[e['type'] for e in seen]}")


def new_session(repo) -> str:
    sid = str(uuid.uuid4())
    repo.create(Session(session_id=sid, started_at=utcnow()))
    return sid


def test_connect_emits_session_started(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        event = json.loads(socket.receive_text())
        assert event["type"] == EventType.SESSION_STARTED
        assert event["data"]["session_id"] == session_id
        assert event["seq"] == 1


def test_manual_question_streams_to_completion(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()  # session.started
        socket.send_text(json.dumps({
            "type": EventType.QUESTION_MANUAL,
            "data": {"text": "How would you handle duplicate records in a pipeline?"},
        }))

        seen = read_until(socket, EventType.ANSWER_COMPLETED)
        types = [e["type"] for e in seen]

        assert EventType.TRANSCRIPT_FINAL in types
        assert EventType.ANSWER_STARTED in types
        assert seen[-1]["data"]["answer"]["summary"]


def test_ping_pong(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()
        socket.send_text(json.dumps({"type": EventType.PING}))
        assert json.loads(socket.receive_text())["type"] == EventType.PONG


def test_malformed_message_reports_error_without_closing(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()
        socket.send_text("this is not json")

        error = json.loads(socket.receive_text())
        assert error["type"] == EventType.ERROR
        assert error["data"]["code"] == "BadMessage"

        # Socket must still work afterwards.
        socket.send_text(json.dumps({"type": EventType.PING}))
        assert json.loads(socket.receive_text())["type"] == EventType.PONG


def test_unsupported_message_type_is_reported(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()
        socket.send_text(json.dumps({"type": EventType.ANSWER_COMPLETED}))
        error = json.loads(socket.receive_text())
        assert error["data"]["code"] == "UnsupportedMessage"


def test_filler_is_rejected_over_the_socket(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()
        socket.send_text(json.dumps({
            "type": EventType.QUESTION_MANUAL, "data": {"text": "yeah"}
        }))
        seen = read_until(socket, EventType.QUESTION_REJECTED)
        assert seen[-1]["data"]["reason"]


def test_reconnect_replays_missed_events(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()
        socket.send_text(json.dumps({
            "type": EventType.QUESTION_MANUAL, "data": {"text": "What is a database index?"},
        }))
        read_until(socket, EventType.ANSWER_COMPLETED)

    # Reconnecting from seq 0 replays the session so far instead of restarting it.
    with client.websocket_connect(f"/ws/session/{session_id}?since_seq=0") as socket:
        first = json.loads(socket.receive_text())
        assert first["seq"] == 1
        assert first["type"] == EventType.SESSION_STARTED


def test_session_survives_disconnect(ws_client):
    client, repo = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        socket.receive_text()

    assert session_manager.get(session_id) is not None


def test_bad_token_is_rejected(ws_client, monkeypatch):
    from app.core.config import settings

    client, repo = ws_client
    session_id = new_session(repo)
    monkeypatch.setattr(settings, "api_token", "expected-token")

    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/session/{session_id}?token=wrong") as socket:
            socket.receive_text()


def test_socket_survives_a_missing_provider_key(database, retriever, monkeypatch):
    """Regression: the provider client used to raise at construction, which happened
    before websocket.accept() and rejected the handshake with a 502 — taking
    transcription and the whole session down over a missing key. A session must
    start, transcribe, and record without one; only answers should fail."""
    from app.core.config import settings as app_settings
    from app.llm.groq_client import GroqClient

    monkeypatch.setattr(app_settings, "groq_api_key", "")

    repo = SessionRepository(database)
    real_client = GroqClient()  # must not raise
    monkeypatch.setattr(deps, "get_session_repository", lambda: repo)
    monkeypatch.setattr(deps, "get_retriever", lambda: retriever)
    monkeypatch.setattr(deps, "get_llm_client", lambda: real_client)
    monkeypatch.setattr(deps, "get_summarizer", lambda: None)
    monkeypatch.setattr(
        deps, "get_session_memory",
        lambda: __import__("app.memory.session_memory",
                           fromlist=["InMemorySessionMemory"]).InMemorySessionMemory(),
    )
    import app.api.ws as ws_module

    for name in ("get_session_repository", "get_retriever", "get_llm_client",
                 "get_summarizer", "get_session_memory"):
        monkeypatch.setattr(ws_module, name, getattr(deps, name))

    with TestClient(app) as client:
        session_id = new_session(repo)
        with client.websocket_connect(f"/ws/session/{session_id}") as socket:
            assert json.loads(socket.receive_text())["type"] == EventType.SESSION_STARTED

            socket.send_text(json.dumps({
                "type": EventType.QUESTION_MANUAL,
                "data": {"text": "How would you handle duplicate records?"},
            }))
            seen = read_until(socket, EventType.ANSWER_ERROR)

        types = [e["type"] for e in seen]
        # The transcript still lands; only the answer fails.
        assert EventType.TRANSCRIPT_FINAL in types
        # The sentence names what to do, not an environment variable.
        assert "Groq API key" in seen[-1]["data"]["message"]
        assert repo.get_transcript(session_id)

    session_manager._sessions.clear()


def test_correct_token_is_accepted(ws_client, monkeypatch):
    from app.core.config import settings

    client, repo = ws_client
    session_id = new_session(repo)
    monkeypatch.setattr(settings, "api_token", "expected-token")

    with client.websocket_connect(f"/ws/session/{session_id}?token=expected-token") as socket:
        assert json.loads(socket.receive_text())["type"] == EventType.SESSION_STARTED
