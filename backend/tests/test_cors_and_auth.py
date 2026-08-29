"""The exact chain that made the packaged app say "Cannot reach the backend".

That failure was invisible to every tool used to diagnose it: the process was
running, the port was open, and `curl /health` returned healthy. It only
appeared in the WebView, because a browser makes two requests where curl makes
one -- a CORS preflight, then the real request -- and the preflight was being
rejected before CORS middleware ever saw it.

So these tests deliberately imitate a *browser*, not curl: every request carries
an `Origin`, and the preflights are real `OPTIONS` requests with
`Access-Control-Request-*` headers. A test that omits the Origin header passes
against a backend the desktop app cannot talk to, which is precisely how this
shipped.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

TOKEN = "TEST_TOKEN_DO_NOT_LEAK"

#: What the packaged Tauri v2 WebView sends on Windows. Not a guess: it is the
#: origin of the custom protocol Tauri serves the bundled frontend from, and it
#: is what appeared in the rejected preflight in production.
DESKTOP_ORIGIN = "http://tauri.localhost"

#: The Vite dev server, for `npm run dev` in a browser.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


@pytest.fixture
def secured(monkeypatch):
    """A backend with the token check switched on, as the desktop shell runs it."""
    monkeypatch.setattr(settings, "api_token", TOKEN)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def open_backend(monkeypatch):
    """No token: the development default, where the check is disabled."""
    monkeypatch.setattr(settings, "api_token", "")
    with TestClient(app) as client:
        yield client


def _preflight(client, path: str, origin: str, method: str = "GET"):
    """A real browser preflight, headers and all."""
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


# ------------------------------------------------------------------- health


def test_health_answers_without_a_token(secured):
    """The shell polls this before the UI exists; it must stay open, and that
    is a deliberate decision rather than an oversight."""
    response = secured.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_answers_a_browser_request_with_cors_headers(secured):
    """`curl /health` succeeding proves nothing about the WebView: without this
    header the browser discards a perfectly good 200."""
    response = secured.get("/health", headers={"Origin": DESKTOP_ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DESKTOP_ORIGIN


def test_ready_reports_the_provider_that_actually_exists(secured):
    body = secured.get("/ready", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert "groq_configured" in body
    assert "gemini_configured" not in body


# ---------------------------------------------------------------- preflight


def test_the_desktop_origin_survives_preflight(secured):
    """The regression. This preflight returned 400 "Disallowed CORS origin" in
    production, so every authenticated request from the packaged app failed
    before it was sent."""
    response = _preflight(secured, "/settings", DESKTOP_ORIGIN)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DESKTOP_ORIGIN


def test_a_preflight_is_not_rejected_by_the_token_check(secured):
    """Browsers strip Authorization from OPTIONS, so gating the preflight on a
    token rejects every cross-origin request before it is even attempted."""
    response = _preflight(secured, "/settings", DESKTOP_ORIGIN)
    assert response.status_code != 401


def test_the_preflight_allows_the_authorization_header(secured):
    """Without this the browser refuses to send the Bearer token at all."""
    response = _preflight(secured, "/settings", DESKTOP_ORIGIN)
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed


@pytest.mark.parametrize("method", ["GET", "PUT", "POST", "DELETE"])
def test_every_method_the_client_uses_survives_preflight(secured, method):
    response = _preflight(secured, "/settings", DESKTOP_ORIGIN, method=method)
    assert response.status_code == 200


@pytest.mark.parametrize("origin", DEV_ORIGINS)
def test_the_development_origins_survive_preflight(secured, origin):
    response = _preflight(secured, "/settings", origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_an_unrelated_web_origin_is_refused(secured):
    """The backend listens on loopback, but any page the user has open could
    still try to reach it."""
    response = _preflight(secured, "/settings", "https://evil.example.com")
    assert response.headers.get("access-control-allow-origin") is None


def test_a_dynamic_port_on_the_desktop_origin_is_allowed(secured):
    """The port is chosen at spawn time, so the allowed origin cannot be pinned
    to one."""
    response = _preflight(secured, "/settings", "http://127.0.0.1:49312")
    assert response.status_code == 200


# --------------------------------------------------------------------- auth


def test_an_authorized_request_succeeds(secured):
    response = secured.get(
        "/settings",
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": DESKTOP_ORIGIN},
    )
    assert response.status_code == 200


def test_an_unauthorized_request_is_rejected(secured):
    response = secured.get("/settings", headers={"Origin": DESKTOP_ORIGIN})
    assert response.status_code == 401


def test_a_rejection_still_carries_cors_headers(secured):
    """A 401 without them reaches the UI as an opaque "Failed to fetch", which
    is why TokenMiddleware has to run *inside* CORSMiddleware."""
    response = secured.get("/settings", headers={"Origin": DESKTOP_ORIGIN})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == DESKTOP_ORIGIN


@pytest.mark.parametrize(
    "header",
    [
        "",                       # empty
        "Bearer",                 # scheme only
        "Bearer ",                # scheme and nothing else
        f"{TOKEN}",               # no scheme
        f"Basic {TOKEN}",         # wrong scheme
        f"Bearer {TOKEN} extra",  # trailing junk
        "Bearer " + "x" * 5000,   # absurdly long
    ],
)
def test_a_malformed_authorization_header_fails_closed(secured, header):
    response = secured.get("/settings", headers={"Authorization": header})
    assert response.status_code == 401


def test_the_scheme_is_matched_case_insensitively(secured):
    """RFC 7235 makes the scheme case-insensitive; rejecting `bearer` would be
    a bug that only shows up against some clients."""
    response = secured.get("/settings", headers={"Authorization": f"bearer {TOKEN}"})
    assert response.status_code == 200


def test_a_wrong_token_is_rejected(secured):
    response = secured.get("/settings", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_the_token_never_appears_in_a_rejection_body(secured):
    response = secured.get("/settings", headers={"Authorization": "Bearer wrong"})
    assert TOKEN not in response.text


def test_the_token_is_never_logged(secured, caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        secured.get("/settings", headers={"Authorization": f"Bearer {TOKEN}"})
        secured.get("/settings", headers={"Authorization": "Bearer wrong"})

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert TOKEN not in logged


def test_no_token_configured_disables_the_check(open_backend):
    """The development default: uvicorn started by hand, no shell to mint one."""
    assert open_backend.get("/settings").status_code == 200


# ---------------------------------------------------------------- websocket


def test_a_websocket_without_a_token_is_refused(secured):
    """WebSocket clients cannot set headers, so the token travels as a query
    parameter -- and must still be checked."""
    with pytest.raises(Exception):
        with secured.websocket_connect("/ws/session/some-id") as socket:
            socket.receive_text()


def test_a_websocket_with_the_wrong_token_is_refused(secured):
    with pytest.raises(Exception):
        with secured.websocket_connect("/ws/session/some-id?token=wrong") as socket:
            socket.receive_text()


def test_a_token_in_the_query_string_does_not_authenticate_http(secured):
    """A URL-borne credential leaks into proxy logs, history and crash reports.
    The only client that cannot set a header is the WebSocket, and that does not
    pass through this middleware at all."""
    assert secured.get(f"/settings?token={TOKEN}").status_code == 401
