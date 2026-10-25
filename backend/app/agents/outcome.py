"""What a human approval does to the claim (pure). Shared by the approval API and finalize."""

from decimal import Decimal

from app.agents.calculator import money
from app.agents.state import HumanApproval
from app.models import ClaimStatus

DRAFT_TO_STATUS = {
    "approve": ClaimStatus.APPROVED,
    "partial_approve": ClaimStatus.PARTIALLY_APPROVED,
    "reject": ClaimStatus.REJECTED,
    "request_documents": ClaimStatus.DOCUMENTS_REQUESTED,
}


class ApprovalError(ValueError):
    """The message is shown to the reviewer (HTTP 422)."""


def final_outcome(decision: dict, approval: HumanApproval) -> tuple[ClaimStatus, Decimal | None]:
    """(final claim status, approved amount) for a decision draft and the reviewer's action.

    approved : accept the draft as it is (a flag_for_review draft has nothing to accept)
    edited   : the reviewer sets the amount (0 < amount <= claimed); partial if below the claim
    rejected : always rejected, amount 0
    """
    kind = decision["decision"]
    if approval.action == "rejected":
        return ClaimStatus.REJECTED, money(0)

    if approval.action == "approved":
        if kind == "flag_for_review":
            raise ApprovalError(
                "This draft is flagged for review and has nothing to approve as it is. "
                "Use 'edited' with a final amount and comment, or 'rejected'."
            )
        status = DRAFT_TO_STATUS[kind]
        payable = decision.get("payable_amount")
        if status in (ClaimStatus.APPROVED, ClaimStatus.PARTIALLY_APPROVED):
            return status, money(payable if payable is not None else 0)
        return status, None if status == ClaimStatus.DOCUMENTS_REQUESTED else money(0)

    # edited
    if kind == "request_documents":
        raise ApprovalError("A document request has no amount to edit.")
    amount = approval.final_amount
    claimed = decision.get("claimed_amount")
    if amount is None or amount <= 0:
        raise ApprovalError("An edited amount must be above 0; use 'rejected' to pay nothing.")
    if claimed is not None and amount > Decimal(str(claimed)):
        raise ApprovalError(f"The final amount cannot be more than the claimed Rs {claimed}.")
    full = claimed is not None and amount == Decimal(str(claimed))
    return (ClaimStatus.APPROVED if full else ClaimStatus.PARTIALLY_APPROVED), money(amount)
