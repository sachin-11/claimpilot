from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClaimDecision
from app.schemas.claim import ClaimDecisionCreate


class DecisionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: ClaimDecisionCreate) -> ClaimDecision:
        # JSONB cannot store UUID objects, so cited clauses are dumped in JSON mode.
        decision = ClaimDecision(
            **data.model_dump(exclude={"cited_clauses"}),
            cited_clauses=[c.model_dump(mode="json") for c in data.cited_clauses],
        )
        self.session.add(decision)
        await self.session.flush()
        return decision

    async def latest_for_claim(self, claim_id: UUID) -> ClaimDecision | None:
        result = await self.session.execute(
            select(ClaimDecision)
            .where(ClaimDecision.claim_id == claim_id)
            .order_by(ClaimDecision.created_at.desc(), ClaimDecision.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
