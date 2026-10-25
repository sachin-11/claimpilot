"""Audit rows for tool writes. Same transaction as the change (flush only, caller commits)."""

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActorType
from app.repositories.audit import AuditLogRepository


async def audit_agent_action(
    session: AsyncSession,
    *,
    caller: str,
    action: str,
    entity_type: str,
    entity_id: UUID | str,
    details: dict[str, Any] | None = None,
) -> None:
    await AuditLogRepository(session).log(
        actor_type=ActorType.AGENT,
        actor_id=caller,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details={"via": "mcp", **(details or {})},
    )
