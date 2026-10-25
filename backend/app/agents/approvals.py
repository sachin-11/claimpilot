"""The human approval, as the system of record. Only this path (never an MCP tool or an agent)
writes `claim_approvals` and moves a claim to approved / partially_approved / rejected."""

import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.outcome import ApprovalError, final_outcome
from app.agents.state import HumanApproval
from app.models import (
    ActorType,
    ApprovalAction,
    Claim,
    ClaimApproval,
    ClaimDecision,
    ClaimStatus,
    User,
)
from app.repositories.audit import AuditLogRepository
from app.repositories.claim import ClaimRepository

logger = logging.getLogger(__name__)

ACTION = {
    "approved": ApprovalAction.APPROVED,
    "rejected": ApprovalAction.REJECTED,
    "edited": ApprovalAction.EDITED,
}


class ApprovalConflictError(RuntimeError):
    """The claim or decision is not in a state that accepts this approval (HTTP 409)."""


async def record_approval(
    session: AsyncSession,
    *,
    claim_id: UUID,
    run_id: str,
    decision: dict[str, Any],
    approval: HumanApproval,
) -> tuple[ClaimStatus, Decimal | None, bool]:
    """Write the approval row, the final claim status, `approved_amount` and the audit rows in ONE
    transaction. Returns (final status, approved amount, already_recorded).

    Safe to call again for the same run (e.g. a retry after a failed graph resume): when the row
    exists with the same action and amount nothing is written and already_recorded is True."""
    status, amount = final_outcome(decision, approval)  # ApprovalError -> 422 for the reviewer

    reviewer_id = _uuid(approval.reviewer)
    reviewer = (
        await session.execute(select(User).where(User.id == reviewer_id, User.is_active))
    ).scalar_one_or_none()
    if reviewer is None:
        raise ApprovalError("reviewer_id is not an active user.")

    claim = (
        await session.execute(select(Claim).where(Claim.id == claim_id).with_for_update())
    ).scalar_one()
    decision_row = None
    if decision.get("decision_id"):
        decision_row = (
            await session.execute(
                select(ClaimDecision).where(
                    ClaimDecision.id == UUID(decision["decision_id"]),
                    ClaimDecision.claim_id == claim_id,
                    ClaimDecision.agent_run_id == run_id,
                )
            )
        ).scalar_one_or_none()
    if decision_row is None:
        raise ApprovalConflictError("the decision draft of this run was not found")

    existing = (
        await session.execute(
            select(ClaimApproval).where(ClaimApproval.decision_id == decision_row.id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        same = existing.action == ACTION[approval.action] and existing.final_amount == amount
        if same and claim.status == status:
            return status, amount, True
        raise ApprovalConflictError("this decision has already been reviewed")
    if claim.status != ClaimStatus.AWAITING_APPROVAL:
        raise ApprovalConflictError(f"claim is '{claim.status.value}', not awaiting approval")

    session.add(
        ClaimApproval(
            claim_id=claim_id,
            decision_id=decision_row.id,
            reviewer_id=reviewer_id,
            action=ACTION[approval.action],
            final_amount=amount,
            comment=approval.comment,
        )
    )
    if status in (ClaimStatus.APPROVED, ClaimStatus.PARTIALLY_APPROVED):
        claim.approved_amount = amount
    await ClaimRepository(session).update_status(
        claim,
        status,
        actor_type=ActorType.USER,
        actor_id=str(reviewer_id),
        details={"run_id": run_id, "action": approval.action},
    )
    await AuditLogRepository(session).log(
        actor_type=ActorType.USER,
        actor_id=str(reviewer_id),
        action="claim.approval_recorded",
        entity_type="claim",
        entity_id=claim_id,
        details={
            "run_id": run_id,
            "decision_id": str(decision_row.id),
            "action": approval.action,
            "draft_decision": decision["decision"],
            "final_status": status.value,
            "final_amount": None if amount is None else str(amount),
        },
    )
    await session.commit()
    return status, amount, False


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ApprovalError("reviewer_id must be a user id (UUID).") from exc
