import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

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


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: Postgres (+ pgvector) and Redis are reachable. 503 if any is not."""
    state = request.app.state
    postgres, redis = await asyncio.gather(
        _run("postgres", check_postgres(state.engine)),
        _run("redis", check_redis(state.redis)),
    )
    deps = {"postgres": postgres, "redis": redis}
    ok = all(d["status"] == "ok" for d in deps.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "unavailable", "dependencies": deps},
    )
