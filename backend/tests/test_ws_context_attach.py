"""Contract test: the frame the frontend actually sends, over a real socket.

`tests/test_context_attachment.py` drives `LiveSession` directly. This one goes
through `app/api/ws.py` with the exact JSON shape
`frontend/src/api/attachments.ts` builds, so a drift between the two -- a
renamed key, a different kind string -- fails here rather than during a live
interview.

The assertion that matters:

    exact spoken question  +  exact pasted content  =  one effective prompt

No real provider: the fake streaming LLM records the prompts it was handed.
"""

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

SCHEMA = """customers
---------
customer_id
name

orders
---------
order_id
customer_id
order_date"""

QUESTION = "Can you write a SQL query to find customers who haven't placed an order in the last 90 days?"


@pytest.fixture
def ws_client(database, retriever, monkeypatch):
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
        yield client, repo, llm

    session_manager._sessions.clear()


def read_until(socket, wanted, limit=40):
    seen = []
    for _ in range(limit):
        seen.append(json.loads(socket.receive_text()))
        if seen[-1]["type"] == wanted:
            return seen
    raise AssertionError(f"never saw {wanted}; got {[e['type'] for e in seen]}")


def new_session(repo) -> str:
    sid = str(uuid.uuid4())
    repo.create(Session(session_id=sid, started_at=utcnow()))
    return sid


def attach_frame(kind: str, content: str, name: str = "") -> str:
    """Exactly what `prepareTextPaste` in the frontend emits."""
    return json.dumps({
        "type": "context.attach",
        "data": {"kind": kind, "content": content, "name": name},
    })


def asked(llm) -> list[str]:
    out = []
    for prompt in llm.prompts:
        for line in prompt.splitlines():
            if "CURRENT INTERVIEWER QUESTION" in line:
                out.append(line.split("): ", 1)[-1])
    return out


def test_a_frontend_shaped_paste_is_acknowledged(ws_client):
    client, repo, _ = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())  # session.started
        socket.send_text(attach_frame("table", SCHEMA, "schema.txt"))

        seen = read_until(socket, EventType.CONTEXT_ATTACHED)
        ack = seen[-1]["data"]
        assert ack["kind"] == "table"
        assert ack["name"] == "schema.txt"
        assert ack["chars"] == len(SCHEMA)
        assert ack["from_image"] is False
        # Metadata only: the content must not be echoed back onto the socket.
        assert SCHEMA not in json.dumps(seen[-1])


def test_a_paste_alone_produces_no_question_and_no_provider_call(ws_client):
    client, repo, llm = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(attach_frame("sql", "SELECT 1;"))
        seen = read_until(socket, EventType.CONTEXT_ATTACHED)

        types = [e["type"] for e in seen]
        assert EventType.QUESTION_DETECTED not in types
        assert EventType.ANSWER_STARTED not in types
        assert llm.prompts == [], "a paste alone reached the provider"


def test_paste_then_question_reaches_the_prompt_as_one_turn(ws_client):
    """The success criterion from the brief: exact question + exact schema,
    one effective answer."""
    client, repo, llm = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(attach_frame("table", SCHEMA))
        read_until(socket, EventType.CONTEXT_ATTACHED)

        socket.send_text(json.dumps({
            "type": "question.manual", "data": {"text": QUESTION},
        }))
        read_until(socket, EventType.ANSWER_COMPLETED)

    assert len(llm.prompts) == 1, asked(llm)
    prompt = llm.prompts[0]
    # Exact pasted content, byte-for-byte, including the divider rules.
    assert SCHEMA in prompt, "pasted schema did not reach the prompt verbatim"
    # Exact spoken wording.
    assert QUESTION in prompt
    # And it is presented as interviewer-provided material, not as retrieved
    # candidate context.
    assert "MATERIAL THE INTERVIEWER PROVIDED" in prompt


def test_multiple_pastes_reach_the_prompt_in_arrival_order(ws_client):
    client, repo, llm = ws_client
    session_id = new_session(repo)
    query = "SELECT * FROM orders;"
    notes = "must run under 200ms"

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        for kind, content in (("table", SCHEMA), ("sql", query), ("text", notes)):
            socket.send_text(attach_frame(kind, content))
            read_until(socket, EventType.CONTEXT_ATTACHED)

        socket.send_text(json.dumps({
            "type": "question.manual", "data": {"text": "How would you solve this?"},
        }))
        read_until(socket, EventType.ANSWER_COMPLETED)

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    for content in (SCHEMA, query, notes):
        assert content in prompt, f"missing {content[:24]!r}"
    assert prompt.index(SCHEMA) < prompt.index(query) < prompt.index(notes)


def test_an_oversized_paste_is_rejected_with_a_usable_reason(ws_client):
    client, repo, llm = ws_client
    session_id = new_session(repo)

    from app.core.config import settings

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(
            attach_frame("text", "x" * (settings.context_attachment_max_chars + 1))
        )
        seen = read_until(socket, EventType.CONTEXT_REJECTED)

        data = seen[-1]["data"]
        assert data["reason"] == "too_large"
        assert "limit" in data["message"]
        # A refusal is not an application error frame.
        assert EventType.ERROR not in [e["type"] for e in seen]
        assert llm.prompts == []


def test_an_empty_paste_is_rejected(ws_client):
    client, repo, _ = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(attach_frame("text", "   \n  "))
        seen = read_until(socket, EventType.CONTEXT_REJECTED)
        assert seen[-1]["data"]["reason"] == "empty"


def test_an_undecodable_image_is_rejected_not_crashed(ws_client):
    """The ws-level base64 guard. Real OCR is never invoked here."""
    client, repo, _ = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(json.dumps({
            "type": "context.attach",
            "data": {"kind": "image", "content": "", "image_base64": "not!base64!"},
        }))
        seen = read_until(socket, EventType.CONTEXT_REJECTED)
        assert "could not be decoded" in seen[-1]["data"]["message"]

        # The socket is still usable afterwards.
        socket.send_text(attach_frame("text", "still working"))
        read_until(socket, EventType.CONTEXT_ATTACHED)


def test_a_pasted_schema_never_appears_as_transcript(ws_client):
    """Attachment content must not enter the spoken record."""
    client, repo, _ = ws_client
    session_id = new_session(repo)

    with client.websocket_connect(f"/ws/session/{session_id}") as socket:
        json.loads(socket.receive_text())
        socket.send_text(attach_frame("table", SCHEMA))
        seen = read_until(socket, EventType.CONTEXT_ATTACHED)

        transcripts = [
            e for e in seen
            if e["type"] in (EventType.TRANSCRIPT_FINAL, EventType.TRANSCRIPT_PARTIAL)
        ]
        assert transcripts == [], "attachment surfaced as transcript"

    stored = repo.get_transcript(session_id) if hasattr(repo, "get_transcript") else []
    assert all(SCHEMA not in entry.text for entry in stored), (
        "attachment was persisted as a transcript entry"
    )
