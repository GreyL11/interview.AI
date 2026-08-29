from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

#: /health must stay reachable so the shell can poll for readiness before it
#: has anything to authenticate with.
OPEN_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class TokenMiddleware(BaseHTTPMiddleware):
    """Require the shared token issued by the desktop shell at spawn time.

    Binding to 127.0.0.1 keeps the backend off the network but does nothing
    about other processes on the same machine. When API_TOKEN is empty -- the
    development default -- the check is disabled entirely.
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

        if _presented(request) != expected:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing token"})
        return await call_next(request)


def _presented(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    # WebSocket clients cannot set headers, so the token also travels as a
    # query parameter; the WS route checks it itself.
    return request.query_params.get("token")
