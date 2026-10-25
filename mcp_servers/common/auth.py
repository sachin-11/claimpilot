"""Bearer-token check for HTTP mode (pure ASGI, so lifespan and streaming pass through)."""

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send

CALLER_SCOPE_KEY = "mcp.caller"
HEALTH_PATH = "/healthz"


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, tokens: dict[str, str]) -> None:
        self.app = app
        self.tokens = tokens  # token -> caller identity

    def _caller(self, scope: Scope) -> str | None:
        header = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not presented.strip():
            return None
        found: str | None = None
        for token, caller in self.tokens.items():  # no early exit: constant work per request
            if hmac.compare_digest(token.encode(), presented.strip().encode()):
                found = caller
        return found

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] == HEALTH_PATH:
            await _respond(send, 200, {"status": "ok"})
            return
        caller = self._caller(scope)
        if caller is None:
            await _respond(
                send,
                401,
                {"error": "missing or invalid bearer token"},
                {b"www-authenticate": b"Bearer"},
            )
            return
        scope[CALLER_SCOPE_KEY] = caller
        await self.app(scope, receive, send)


async def _respond(
    send: Send, status: int, body: dict[str, str], headers: dict[bytes, bytes] | None = None
) -> None:
    payload = json.dumps(body).encode()
    raw = {b"content-type": b"application/json", b"content-length": str(len(payload)).encode()}
    raw.update(headers or {})
    await send({"type": "http.response.start", "status": status, "headers": list(raw.items())})
    await send({"type": "http.response.body", "body": payload})
