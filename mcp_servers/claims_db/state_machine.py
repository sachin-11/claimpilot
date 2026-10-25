"""Claim status transitions available to agents.

    submitted -> processing -> awaiting_approval -> approved | partially_approved | rejected
                      |               |
                      +---------------+--> documents_requested -> processing
    any open status -> failed

approved / partially_approved / rejected are HUMAN-ONLY: they are reached through the approval
API (Module 8), never through an MCP tool. They are in the table so the whole machine is in one
place, but `check_agent_transition` refuses them.
"""

from app.models import ClaimStatus

S = ClaimStatus

HUMAN_ONLY: frozenset[ClaimStatus] = frozenset({S.APPROVED, S.PARTIALLY_APPROVED, S.REJECTED})
FINAL: frozenset[ClaimStatus] = HUMAN_ONLY | {S.FAILED}

TRANSITIONS: dict[ClaimStatus, frozenset[ClaimStatus]] = {
    S.SUBMITTED: frozenset({S.PROCESSING, S.FAILED}),
    S.PROCESSING: frozenset({S.AWAITING_APPROVAL, S.DOCUMENTS_REQUESTED, S.FAILED}),
    S.AWAITING_APPROVAL: frozenset(
        {S.APPROVED, S.PARTIALLY_APPROVED, S.REJECTED, S.DOCUMENTS_REQUESTED, S.FAILED}
    ),
    S.DOCUMENTS_REQUESTED: frozenset({S.PROCESSING, S.FAILED}),
    S.APPROVED: frozenset(),
    S.PARTIALLY_APPROVED: frozenset(),
    S.REJECTED: frozenset(),
    S.FAILED: frozenset(),
}


class TransitionError(ValueError):
    """The message is shown to the calling agent."""


def check_agent_transition(old: ClaimStatus, new: ClaimStatus) -> None:
    if new in HUMAN_ONLY:
        raise TransitionError(
            f"Agents cannot set a claim to '{new.value}': approvals and rejections are made "
            "by a human reviewer. Move the claim to 'awaiting_approval' instead."
        )
    if new not in TRANSITIONS[old]:
        allowed = sorted(s.value for s in TRANSITIONS[old] - HUMAN_ONLY)
        hint = (
            f"Allowed next statuses for an agent: {', '.join(allowed)}."
            if allowed
            else f"'{old.value}' is a final status; it cannot be changed by an agent."
        )
        raise TransitionError(f"Cannot change status from '{old.value}' to '{new.value}'. {hint}")
