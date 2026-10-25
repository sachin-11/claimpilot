"""Idempotent write tools: the first call with a key does the work and stores its response; a
repeat returns that response untouched."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ToolIdempotency
from mcp_servers.common.errors import ToolError


def request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


async def run_idempotent(
    session: AsyncSession,
    *,
    tool: str,
    key: str,
    payload: dict[str, Any],
    caller: str,
    work: Callable[[], Awaitable[tuple[UUID | None, dict[str, Any]]]],
) -> dict[str, Any]:
    """`work` returns (claim_id, response). Same key + same payload -> stored response with
    `idempotent_replay: true`; same key + different payload -> error. Commits on success."""
    # Serialises concurrent calls that share a key; released at commit/rollback.
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": f"{tool}:{key}"}
    )
    digest = request_hash(payload)
    existing = (
        await session.execute(
            select(ToolIdempotency).where(
                ToolIdempotency.tool == tool, ToolIdempotency.idempotency_key == key
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.request_hash != digest:
            raise ToolError(
                f"idempotency_key {key!r} was already used with different arguments. "
                "Use a new key for a different request."
            )
        return {**existing.response, "idempotent_replay": True}

    claim_id, response = await work()
    session.add(
        ToolIdempotency(
            tool=tool,
            idempotency_key=key,
            request_hash=digest,
            claim_id=claim_id,
            caller=caller,
            response=response,
        )
    )
    await session.commit()
    return {**response, "idempotent_replay": False}
