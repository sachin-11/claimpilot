from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClaimExtraction


class ExtractionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        claim_id: UUID,
        *,
        extracted_data: dict[str, Any],
        model_used: str | None,
        confidence: float | None,
        validation_errors: list[str],
    ) -> ClaimExtraction:
        """A NEW version (previous extractions are kept)."""
        current = (
            await self.session.execute(
                select(func.max(ClaimExtraction.version)).where(
                    ClaimExtraction.claim_id == claim_id
                )
            )
        ).scalar_one()
        row = ClaimExtraction(
            claim_id=claim_id,
            extracted_data=extracted_data,
            model_used=model_used,
            confidence=confidence,
            validation_errors=validation_errors,
            version=(current or 0) + 1,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def latest_for_claim(self, claim_id: UUID) -> ClaimExtraction | None:
        result = await self.session.execute(
            select(ClaimExtraction)
            .where(ClaimExtraction.claim_id == claim_id)
            .order_by(ClaimExtraction.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
