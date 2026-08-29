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


# ------------------------------------------------- CORS preflight vs the token
# Reproduces a packaged-app failure: with a token set, every cross-origin
# request died as "TypeError: Failed to fetch" while /health kept working, so
# the app reported "Engine ready" and could do nothing else. The cause was the
# preflight being rejected before CORS could answer it. None of this is
# reachable in dev, where API_TOKEN is empty and the check short-circuits.

TAURI_ORIGIN = "http://tauri.localhost"


@pytest.fixture
def tokened(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "packaged-token")
    return client


def _preflight(client, path: str, method: str = "POST"):
    # Deliberately no Authorization header: browsers never send one on a
    # preflight, which is the whole point of this test.
    return client.options(
        path,
        headers={
            "Origin": TAURI_ORIGIN,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


def test_a_preflight_on_a_protected_route_is_answered_not_rejected(tokened):
    response = _preflight(tokened, "/sessions")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == TAURI_ORIGIN


def test_the_packaged_webview_origin_is_allowed(tokened):
    response = _preflight(tokened, "/settings", method="GET")
    assert response.headers.get("access-control-allow-origin") == TAURI_ORIGIN


def test_a_foreign_origin_is_still_refused(tokened):
    response = client_options_from(tokened, "https://evil.example")
    assert response.headers.get("access-control-allow-origin") is None


def client_options_from(client, origin: str):
    return client.options(
        "/sessions",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_a_real_request_without_a_token_is_still_rejected(tokened):
    """Letting preflights through must not open the actual route."""
    assert tokened.post("/sessions").status_code == 401


def test_a_rejection_still_carries_cors_headers(tokened):
    """CORS runs outside the token check, so a genuine 401 reaches the UI as a
    401 rather than as an opaque network failure."""
    response = tokened.post("/sessions", headers={"Origin": TAURI_ORIGIN})

    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == TAURI_ORIGIN
