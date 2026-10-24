from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import IngestionRun, IngestionStatus


class IngestionRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, policy_id: UUID) -> IngestionRun:
        run = IngestionRun(policy_id=policy_id, status=IngestionStatus.RUNNING)
        self.session.add(run)
        await self.session.flush()
        return run

    async def get(self, run_id: UUID) -> IngestionRun | None:
        return await self.session.get(IngestionRun, run_id)

    async def last_success(
        self, policy_id: UUID, *, exclude_run_id: UUID | None = None
    ) -> IngestionRun | None:
        stmt = (
            select(IngestionRun)
            .where(
                IngestionRun.policy_id == policy_id,
                IngestionRun.status == IngestionStatus.SUCCESS,
                IngestionRun.file_sha256.is_not(None),
            )
            .order_by(IngestionRun.started_at.desc(), IngestionRun.id.desc())
            .limit(1)
        )
        if exclude_run_id is not None:
            stmt = stmt.where(IngestionRun.id != exclude_run_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def last_for_policy(self, policy_id: UUID) -> IngestionRun | None:
        stmt = (
            select(IngestionRun)
            .where(IngestionRun.policy_id == policy_id)
            .order_by(IngestionRun.started_at.desc(), IngestionRun.id.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def has_recent_running(
        self, policy_id: UUID, within: timedelta = timedelta(minutes=30)
    ) -> bool:
        """A run still 'running' after `within` is treated as crashed, not as a lock."""
        stmt = (
            select(IngestionRun.id)
            .where(
                IngestionRun.policy_id == policy_id,
                IngestionRun.status == IngestionStatus.RUNNING,
                IngestionRun.started_at > datetime.now(UTC) - within,
            )
            .limit(1)
        )
        return (await self.session.execute(stmt)).first() is not None
