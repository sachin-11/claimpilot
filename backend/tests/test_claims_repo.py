from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActorType, AuditLog, ClaimSource, ClaimStatus, DecisionType
from app.repositories import ClaimRepository, DecisionRepository
from app.schemas.claim import CitedClause, ClaimCreate, ClaimDecisionCreate


async def make_claim(session: AsyncSession, number: str = "CLM-T1"):
    return await ClaimRepository(session).create(
        ClaimCreate(claim_number=number, claimed_amount=Decimal("1000.50"), source=ClaimSource.SEED)
    )


async def audit_rows(session: AsyncSession, claim_id) -> list[AuditLog]:
    result = await session.execute(
        select(AuditLog)
        .where(AuditLog.entity_type == "claim", AuditLog.entity_id == str(claim_id))
        .order_by(AuditLog.created_at, AuditLog.action)
    )
    return list(result.scalars())


async def test_status_update_writes_audit_log(session: AsyncSession) -> None:
    repo = ClaimRepository(session)
    claim = await make_claim(session)

    await repo.update_status(
        claim,
        ClaimStatus.PROCESSING,
        actor_type=ActorType.AGENT,
        actor_id="supervisor",
        request_id="req-1",
        details={"reason": "pipeline started"},
    )

    rows = [r for r in await audit_rows(session, claim.id) if r.action == "claim.status_changed"]
    assert len(rows) == 1
    assert rows[0].actor_type == ActorType.AGENT
    assert rows[0].request_id == "req-1"
    assert rows[0].details == {
        "from": "submitted",
        "to": "processing",
        "reason": "pipeline started",
    }
    assert claim.status == ClaimStatus.PROCESSING


async def test_create_logs_and_list_filters_by_status(session: AsyncSession) -> None:
    repo = ClaimRepository(session)
    first = await make_claim(session, "CLM-T1")
    await make_claim(session, "CLM-T2")
    await repo.update_status(first, ClaimStatus.REJECTED, actor_type=ActorType.SYSTEM)

    assert [c.claim_number for c in await repo.list(status=ClaimStatus.REJECTED)] == ["CLM-T1"]
    assert len(await repo.list(limit=1)) == 1
    assert (await repo.get_by_number("CLM-T1")).documents == []
    assert any(r.action == "claim.created" for r in await audit_rows(session, first.id))


async def test_latest_decision_for_claim(session: AsyncSession) -> None:
    claim = await make_claim(session)
    repo = DecisionRepository(session)
    assert await repo.latest_for_claim(claim.id) is None

    older = await repo.create(
        ClaimDecisionCreate(claim_id=claim.id, decision=DecisionType.FLAG_FOR_REVIEW)
    )
    # created_at defaults to now(), which is the transaction start time, so two rows made in one
    # transaction tie. Real decisions arrive in separate transactions; make the order explicit.
    older.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    await session.flush()
    newest = await repo.create(
        ClaimDecisionCreate(
            claim_id=claim.id,
            decision=DecisionType.PARTIAL_APPROVE,
            payable_amount=Decimal("900.00"),
            cited_clauses=[CitedClause(chunk_id="00000000-0000-0000-0000-000000000001", page=4)],
        )
    )

    latest = await repo.latest_for_claim(claim.id)
    assert latest is not None and latest.id == newest.id
    assert latest.cited_clauses[0]["page"] == 4


async def test_audit_log_is_append_only(session: AsyncSession) -> None:
    claim = await make_claim(session)
    with pytest.raises(DBAPIError, match="append-only"):
        await session.execute(
            text("UPDATE audit_logs SET action = 'x' WHERE entity_id = :id"), {"id": str(claim.id)}
        )
