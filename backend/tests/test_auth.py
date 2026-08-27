import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "shell-issued-token")
    with TestClient(app) as client:
        yield client


def test_health_stays_open(secured):
    """The shell polls /health before it has anything to authenticate with."""
    assert secured.get("/health").status_code == 200


def test_protected_route_rejects_a_missing_token(secured):
    response = secured.get("/documents")
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


def test_protected_route_rejects_a_wrong_token(secured):
    response = secured.get("/documents", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_bearer_token_is_accepted(secured):
    response = secured.get(
        "/documents", headers={"Authorization": "Bearer shell-issued-token"}
    )
    assert response.status_code == 200


def test_query_token_is_accepted(secured):
    """WebSocket clients cannot set headers, so the token also travels as a
    query parameter."""
    assert secured.get("/documents?token=shell-issued-token").status_code == 200


def test_token_check_is_disabled_when_unset(client):
    """Development default: no token configured, no check."""
    assert client.get("/documents").status_code == 200


def test_settings_route_is_protected(secured):
    assert secured.get("/settings").status_code == 401
    assert (
        secured.get("/settings", headers={"Authorization": "Bearer shell-issued-token"}).status_code
        == 200
    )
