from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import ActorType, Claim, ClaimStatus
from app.repositories.audit import AuditLogRepository
from app.schemas.claim import ClaimCreate


class ClaimRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.audit = AuditLogRepository(session)

    async def create(
        self,
        data: ClaimCreate,
        *,
        actor_type: ActorType = ActorType.SYSTEM,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> Claim:
        claim = Claim(**data.model_dump(), status=ClaimStatus.SUBMITTED)
        self.session.add(claim)
        await self.session.flush()
        await self.audit.log(
            actor_type=actor_type,
            actor_id=actor_id,
            action="claim.created",
            entity_type="claim",
            entity_id=claim.id,
            details={"claim_number": claim.claim_number, "status": claim.status.value},
            request_id=request_id,
        )
        return claim

    async def get_by_number(self, claim_number: str) -> Claim | None:
        """The claim with its documents loaded."""
        result = await self.session.execute(
            select(Claim)
            .where(Claim.claim_number == claim_number)
            .options(selectinload(Claim.documents))
        )
        return result.scalar_one_or_none()

    async def list(
        self, *, status: ClaimStatus | None = None, limit: int = 50, offset: int = 0
    ) -> list[Claim]:
        stmt = select(Claim).order_by(Claim.claim_number).limit(limit).offset(offset)
        if status is not None:
            stmt = stmt.where(Claim.status == status)
        result = await self.session.execute(stmt)
        return list(result.scalars())

    async def update_status(
        self,
        claim: Claim,
        new_status: ClaimStatus,
        *,
        actor_type: ActorType,
        actor_id: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Claim:
        """Change status and write the audit row in the same transaction."""
        old_status = claim.status
        claim.status = new_status
        await self.session.flush()
        await self.audit.log(
            actor_type=actor_type,
            actor_id=actor_id,
            action="claim.status_changed",
            entity_type="claim",
            entity_id=claim.id,
            details={"from": old_status.value, "to": new_status.value, **(details or {})},
            request_id=request_id,
        )
        return claim
