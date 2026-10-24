from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Policy, PolicyChunk
from app.schemas.policy import BulkUpsertResult, PolicyChunkCreate


class PolicyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_code(self, policy_code: str) -> Policy | None:
        result = await self.session.execute(select(Policy).where(Policy.policy_code == policy_code))
        return result.scalar_one_or_none()

    async def list_active(self) -> list[Policy]:
        result = await self.session.execute(
            select(Policy).where(Policy.is_active.is_(True)).order_by(Policy.policy_code)
        )
        return list(result.scalars())

    async def upsert_from_rules(self, rules: dict[str, Any]) -> tuple[Policy, bool]:
        """Create or refresh a policy from one entry of policy_rules.yaml.

        Returns (policy, changed). The version is bumped only when the rules differ.
        """
        code = rules["policy_id"]
        policy = await self.get_by_code(code)
        if policy is None:
            policy = Policy(
                policy_code=code,
                name=rules["name"],
                source_pdf=rules.get("source_pdf"),
                rules=rules,
            )
            self.session.add(policy)
            await self.session.flush()
            return policy, True

        if (
            policy.rules == rules
            and policy.name == rules["name"]
            and policy.source_pdf == rules.get("source_pdf")
        ):
            return policy, False
        policy.name = rules["name"]
        policy.source_pdf = rules.get("source_pdf")
        policy.rules = rules
        policy.version += 1
        await self.session.flush()
        return policy, True


class PolicyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_upsert(
        self, policy_id: UUID, chunks: list[PolicyChunkCreate]
    ) -> BulkUpsertResult:
        """Insert new chunks; skip the ones whose content_hash already exists.

        A matching hash means identical content, so the embedding is never touched. Only the
        position fields (index, section, pages) are refreshed if they moved.
        """
        result = BulkUpsertResult()
        existing_rows = await self.session.execute(
            select(PolicyChunk).where(PolicyChunk.policy_id == policy_id)
        )
        existing = {c.content_hash: c for c in existing_rows.scalars()}

        position = ("chunk_index", "section_path", "page_start", "page_end", "token_count")
        seen: set[str] = set()
        for data in chunks:
            if data.content_hash in seen:  # duplicate inside the same batch
                result.unchanged += 1
                continue
            seen.add(data.content_hash)

            current = existing.get(data.content_hash)
            if current is None:
                self.session.add(PolicyChunk(policy_id=policy_id, **data.model_dump()))
                result.inserted += 1
            elif any(getattr(current, f) != getattr(data, f) for f in position):
                for f in position:
                    setattr(current, f, getattr(data, f))
                result.updated += 1
            else:
                result.unchanged += 1
        await self.session.flush()
        return result

    async def delete_stale(self, policy_id: UUID, keep_hashes: set[str]) -> int:
        """Delete this policy's chunks whose hash is not in keep_hashes. Returns rows deleted."""
        stmt = delete(PolicyChunk).where(PolicyChunk.policy_id == policy_id)
        if keep_hashes:
            stmt = stmt.where(PolicyChunk.content_hash.not_in(keep_hashes))
        result = await self.session.execute(stmt)
        return result.rowcount or 0

    async def count_by_policy(self, policy_id: UUID) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(PolicyChunk).where(PolicyChunk.policy_id == policy_id)
        )
        return result.scalar_one()
