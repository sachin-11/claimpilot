from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActorType, AuditLog


class AuditLogRepository:
    """Append-only. Like every repository here it flushes but never commits: the caller owns
    the transaction, so an audit row always lives or dies with the change it describes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def log(
        self,
        *,
        actor_type: ActorType,
        action: str,
        entity_type: str,
        entity_id: UUID | str,
        actor_id: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id),
            details=details or {},
            request_id=request_id,
        )
        self.session.add(entry)
        await self.session.flush()
        return entry
