import hmac

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

#: Reachable without a token, deliberately.
#:
#: `/health` has to be open because the desktop shell polls it to decide the
#: backend is up, and the UI polls it for liveness -- both before and
#: independently of anything that could carry a token. It returns one constant
#: word and touches no user state, so it discloses nothing beyond "a server is
#: here", which anyone who can connect to the port already knows.
#:
#: The docs routes are open for the same reason they exist at all: they describe
#: the shape of the API, never its contents.
OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class TokenMiddleware(BaseHTTPMiddleware):
    """Require the shared token issued by the desktop shell at spawn time.

    Binding to 127.0.0.1 keeps the backend off the network but does nothing
    about other processes on the same machine. When API_TOKEN is empty -- the
    development default -- the check is disabled entirely.

    Only HTTP passes through here: Starlette's BaseHTTPMiddleware leaves
    WebSocket scopes alone, so `app.api.ws` does its own check on the token it
    receives as a query parameter.
    """

    async def dispatch(self, request: Request, call_next):
        expected = settings.api_token
        if not expected or request.url.path in OPEN_PATHS:
            return await call_next(request)

        # A CORS preflight never carries credentials -- browsers strip the
        # Authorization header from OPTIONS -- so gating it on the token
        # rejects every cross-origin request before it is even attempted. The
        # preflight performs no action; CORSMiddleware is what decides whether
        # the origin may proceed, and the real request that follows is still
        # token-checked here.
        if request.method == "OPTIONS":
            return await call_next(request)

        if not _matches(_presented(request), expected):
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing token"}
            )
        return await call_next(request)


def _presented(request: Request) -> str | None:
    """The token this request offers, from the Authorization header only.

    Deliberately *not* also accepting `?token=`: a URL-borne credential ends up
    in proxy logs, browser history and crash reports, and the only client that
    genuinely cannot set a header is the WebSocket, which is not routed through
    this middleware at all.
    """
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def _matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison.

    The token is local and short-lived, so this is not the app's load-bearing
    defence -- but a byte-by-byte early exit leaks its prefix to anything on the
    machine that can time requests, and `compare_digest` costs nothing to use.
    """
    if presented is None:
        return False
    return hmac.compare_digest(presented, expected)
