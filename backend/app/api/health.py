import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

CHECK_TIMEOUT_SECONDS = 3.0


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. Deliberately checks no dependencies."""
    return {"status": "ok"}


async def check_postgres(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        row = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        if row.first() is None:
            raise RuntimeError("pgvector extension is not installed (run `make migrate`)")


async def check_redis(client: Redis) -> None:
    await client.ping()


async def _run(name: str, check: Awaitable[None]) -> dict[str, Any]:
    try:
        await asyncio.wait_for(check, CHECK_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("readiness_check_failed", extra={"dependency": name, "error": repr(exc)})
        return {"status": "error", "detail": str(exc) or type(exc).__name__}
    return {"status": "ok"}


async def check_mcp(server: str, url: str) -> None:
    """The server's open health endpoint (`/healthz`, next to the /mcp endpoint)."""
    base = url.removesuffix("/mcp")
    async with httpx.AsyncClient(timeout=CHECK_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base}/healthz")
        response.raise_for_status()


async def check_worker(redis: Redis) -> None:
    """The arq worker refreshes this key every few seconds; no key = no live worker."""
    from app.worker.queue import HEALTH_KEY  # noqa: PLC0415

    if not await redis.exists(HEALTH_KEY):
        raise RuntimeError("no worker heartbeat (is the worker running?)")


async def check_langfuse() -> dict[str, Any]:
    """Reported, never required: tracing is optional and must not take the API out."""
    from app.observability import tracing  # noqa: PLC0415

    if not tracing.tracing_enabled():
        return {"status": "disabled", "optional": True}
    try:
        ok = await asyncio.wait_for(
            asyncio.to_thread(tracing._client.auth_check),
            CHECK_TIMEOUT_SECONDS,  # noqa: SLF001
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc) or type(exc).__name__, "optional": True}
    return {"status": "ok" if ok else "error", "optional": True}


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: Postgres (+ pgvector), Redis, the three MCP servers and, when a worker is
    used, its heartbeat. 503 if any of those is not ok. Langfuse is only reported."""
    from app.mcp_client.client import SERVERS, server_url  # noqa: PLC0415

    state = request.app.state
    settings = state.settings
    checks = {
        "postgres": check_postgres(state.engine),
        "redis": check_redis(state.redis),
    }
    if settings.readiness_extra_checks:
        for server in SERVERS:
            checks[f"mcp_{server}"] = check_mcp(server, server_url(server))
        if settings.use_worker:
            checks["worker"] = check_worker(state.redis)
    results = await asyncio.gather(*(_run(name, c) for name, c in checks.items()))
    deps: dict[str, Any] = dict(zip(checks, results, strict=True))
    ok = all(d["status"] == "ok" for d in deps.values())  # before the optional ones
    if settings.readiness_extra_checks:
        deps["langfuse"] = await check_langfuse()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "unavailable", "dependencies": deps},
    )
