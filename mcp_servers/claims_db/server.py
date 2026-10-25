"""claims_db MCP server: claim data for agents (read) and two restricted writes."""

from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ActorType, Claim, ClaimStatus, DecisionType, Policy, PolicyChunk
from app.repositories.claim import ClaimRepository
from app.repositories.decision import DecisionRepository
from app.repositories.extraction import ExtractionRepository
from app.schemas.claim import CitedClause, ClaimDecisionCreate
from mcp_servers.claims_db.state_machine import TransitionError, check_agent_transition
from mcp_servers.common.audit import audit_agent_action
from mcp_servers.common.deps import ServerDeps, default_deps
from mcp_servers.common.errors import ToolError, claim_not_found
from mcp_servers.common.idempotency import run_idempotent
from mcp_servers.common.instrument import caller_id, instrumented
from mcp_servers.common.runtime import run_cli

NAME = "claims_db"

INSTRUCTIONS = (
    "Claim records for the claims workflow. Read tools return summaries without contact details. "
    "Write tools (if enabled) change claim status along an allowed path or save a DRAFT decision "
    "for a human reviewer; they cannot approve or reject a claim."
)

ClaimNumber = Annotated[
    str,
    Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9._-]+$", description="e.g. CLM-001"),
]
IdempotencyKey = Annotated[
    str,
    Field(
        min_length=8,
        max_length=128,
        description="Unique per intended action (e.g. a UUID). Repeating a call with the same key "
        "returns the first result instead of writing again.",
    ),
]

# Statuses an agent may ask for. The human-only ones are rejected with a clear message by the
# state machine, so they still appear in the schema (and fail with a helpful error).
_STATUS_ENUM = ClaimStatus


class DeductionItem(BaseModel):
    description: str = Field(min_length=1, max_length=200)
    amount: Decimal = Field(ge=0, le=Decimal("100000000"))
    clause: str | None = Field(default=None, max_length=100, description="Clause number, if any")


class DraftCitation(BaseModel):
    chunk_id: str = Field(description="chunk_id returned by policy_docs.search_policy")


def _money(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _doc_summary(claim: Claim) -> list[dict[str, Any]]:
    return [
        {
            "doc_type": d.doc_type.value,
            "mime_type": d.mime_type,
            "page_count": d.page_count,
            "is_scanned": d.is_scanned,
        }
        for d in claim.documents
    ]


def create_server(deps: ServerDeps | None = None, *, host: str = "127.0.0.1") -> FastMCP:
    deps = deps or default_deps()
    mcp = FastMCP(NAME, instructions=INSTRUCTIONS, host=host)
    tool = instrumented(NAME, deps.settings)

    async def load_claim(session: AsyncSession, claim_number: str) -> Claim:
        claim = await ClaimRepository(session).get_by_number(claim_number)
        if claim is None:
            raise claim_not_found(claim_number)
        await session.refresh(claim, ["customer_policy"])  # async: no implicit lazy loading
        return claim

    # ------------------------------------------------------------------ read tools

    @mcp.tool()
    @tool
    async def get_claim(ctx: Context, claim_number: ClaimNumber) -> dict[str, Any]:
        """Summary of one claim: status, dates, claimed / approved amount, hospital, diagnosis,
        policy code and the list of uploaded documents. Contains no customer contact details."""
        async with deps.session() as session:
            claim = await load_claim(session, claim_number)
            policy_code = None
            if claim.customer_policy is not None:
                policy_code = (
                    await session.execute(
                        select(Policy.policy_code).where(
                            Policy.id == claim.customer_policy.policy_id
                        )
                    )
                ).scalar_one_or_none()
            return {
                "claim_number": claim.claim_number,
                "status": claim.status.value,
                "policy_code": policy_code,
                "hospital_name": claim.hospital_name,
                "diagnosis_text": claim.diagnosis_text,
                "icd10_code": claim.icd10_code,
                "admission_date": claim.admission_date,
                "discharge_date": claim.discharge_date,
                "claimed_amount": _money(claim.claimed_amount),
                "approved_amount": _money(claim.approved_amount),
                "documents": _doc_summary(claim),
            }

    @mcp.tool()
    @tool
    async def get_claim_extraction(ctx: Context, claim_number: ClaimNumber) -> dict[str, Any]:
        """Latest structured data extracted from the claim documents (procedure, room, ICU days,
        line items, bill totals), with consistency_issues found across documents and the
        missing_documents list. Patient name and source snippets are left out."""
        async with deps.session() as session:
            claim = await load_claim(session, claim_number)
            row = await ExtractionRepository(session).latest_for_claim(claim.id)
            if row is None:
                raise ToolError(
                    f"Claim {claim_number} has no extraction yet (intake has not run on it)."
                )
            data = dict(row.extracted_data)
            return {
                "claim_number": claim.claim_number,
                "version": row.version,
                "confidence": row.confidence,
                "validation_errors": row.validation_errors,
                "consistency_issues": data.pop("consistency_issues", []),
                "missing_documents": data.pop("missing_documents", []),
                "patient_name_present": bool(data.pop("patient_name", None)),
                "extracted": {
                    k: v for k, v in data.items() if k not in ("field_sources", "intake")
                },
            }

    @mcp.tool()
    @tool
    async def get_customer_history(ctx: Context, claim_number: ClaimNumber) -> dict[str, Any]:
        """Other claims on the same customer policy as this claim (count, dates, amounts,
        statuses) plus policy start date, sum insured and remaining sum insured (sum insured
        minus amounts already approved on the other claims). Use it for waiting-period and
        repeat-claim checks."""
        async with deps.session() as session:
            claim = await load_claim(session, claim_number)
            cp = claim.customer_policy
            if cp is None:
                raise ToolError(f"Claim {claim_number} is not linked to a customer policy.")
            others = (
                (
                    await session.execute(
                        select(Claim)
                        .where(Claim.customer_policy_id == cp.id, Claim.id != claim.id)
                        .order_by(Claim.admission_date, Claim.claim_number)
                    )
                )
                .scalars()
                .all()
            )
            paid = sum(
                (
                    c.approved_amount or Decimal(0)
                    for c in others
                    if c.status in (ClaimStatus.APPROVED, ClaimStatus.PARTIALLY_APPROVED)
                ),
                Decimal(0),
            )
            days = (claim.admission_date - cp.start_date).days if claim.admission_date else None
            return {
                "claim_number": claim.claim_number,
                "policy_start_date": cp.start_date,
                "days_from_policy_start_to_admission": days,
                "sum_insured": _money(cp.sum_insured),
                "already_approved_amount": _money(paid),
                "remaining_sum_insured": _money(cp.sum_insured - paid),
                "previous_claim_count": len(others),
                "previous_claims": [
                    {
                        "claim_number": c.claim_number,
                        "admission_date": c.admission_date,
                        "claimed_amount": _money(c.claimed_amount),
                        "approved_amount": _money(c.approved_amount),
                        "status": c.status.value,
                    }
                    for c in others
                ],
            }

    @mcp.tool()
    @tool
    async def list_claims(
        ctx: Context,
        status: Annotated[
            ClaimStatus | None, Field(description="Only claims in this status")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> list[dict[str, Any]]:
        """List claims (claim number, status, hospital, amount, admission date), ordered by
        claim number. Optionally filter by status."""
        async with deps.session() as session:
            claims = await ClaimRepository(session).list(status=status, limit=limit)
            return [
                {
                    "claim_number": c.claim_number,
                    "status": c.status.value,
                    "hospital_name": c.hospital_name,
                    "claimed_amount": _money(c.claimed_amount),
                    "admission_date": c.admission_date,
                }
                for c in claims
            ]

    # ------------------------------------------------------------------ write tools

    if not deps.settings.write_enabled:
        return mcp

    @mcp.tool()
    @tool
    async def update_claim_status(
        ctx: Context,
        claim_number: ClaimNumber,
        new_status: Annotated[
            _STATUS_ENUM,
            Field(
                description="processing | awaiting_approval | documents_requested | failed. "
                "approved / partially_approved / rejected are refused: humans decide those."
            ),
        ],
        reason: Annotated[str, Field(min_length=3, max_length=500)],
        idempotency_key: IdempotencyKey,
    ) -> dict[str, Any]:
        """Move a claim to its next status. Allowed: submitted -> processing -> awaiting_approval
        or documents_requested (documents_requested -> processing when documents arrive); any
        open claim -> failed. Approving or rejecting is NOT possible here: a human does that
        after the claim reaches awaiting_approval. Idempotent per idempotency_key."""
        caller = caller_id(ctx, deps.settings)
        async with deps.session() as session:

            async def work() -> tuple[Any, dict[str, Any]]:
                claim = await load_claim(session, claim_number)
                old = claim.status
                try:
                    check_agent_transition(old, new_status)
                except TransitionError as exc:
                    raise ToolError(str(exc)) from exc
                await ClaimRepository(session).update_status(
                    claim,
                    new_status,
                    actor_type=ActorType.AGENT,
                    actor_id=caller,
                    details={"reason": reason, "via": "mcp"},
                )
                return claim.id, {
                    "claim_number": claim.claim_number,
                    "previous_status": old.value,
                    "status": new_status.value,
                }

            return await run_idempotent(
                session,
                tool="update_claim_status",
                key=idempotency_key,
                payload={"claim_number": claim_number, "new_status": new_status, "reason": reason},
                caller=caller,
                work=work,
            )

    @mcp.tool()
    @tool
    async def save_decision_draft(
        ctx: Context,
        claim_number: ClaimNumber,
        decision: DecisionType,
        payable_amount: Annotated[Decimal | None, Field(ge=0, le=Decimal("100000000"))],
        deductions: Annotated[list[DeductionItem], Field(max_length=50)],
        reasons: Annotated[
            list[Annotated[str, Field(min_length=1, max_length=500)]], Field(max_length=20)
        ],
        cited_clauses: Annotated[list[DraftCitation], Field(max_length=20)],
        risk_flags: Annotated[
            list[Annotated[str, Field(min_length=1, max_length=200)]], Field(max_length=20)
        ],
        agent_run_id: Annotated[str, Field(min_length=1, max_length=100)],
        idempotency_key: IdempotencyKey,
    ) -> dict[str, Any]:
        """Save a DRAFT decision for human review: decision type, payable amount, deductions,
        reasons, the policy clauses relied on (chunk_ids from search_policy) and risk flags.
        It does not change the claim status and approves nothing; the claim must be in
        'processing'. Idempotent per idempotency_key."""
        caller = caller_id(ctx, deps.settings)
        async with deps.session() as session:

            async def work() -> tuple[Any, dict[str, Any]]:
                claim = await load_claim(session, claim_number)
                if claim.status != ClaimStatus.PROCESSING:
                    raise ToolError(
                        f"Claim {claim_number} is '{claim.status.value}'; drafts can only be saved "
                        "while it is 'processing'."
                    )
                clauses = await _resolve_clauses(session, [c.chunk_id for c in cited_clauses])
                row = await DecisionRepository(session).create(
                    ClaimDecisionCreate(
                        claim_id=claim.id,
                        decision=decision,
                        payable_amount=payable_amount,
                        deductions=[d.model_dump(mode="json") for d in deductions],
                        reasons=reasons,
                        cited_clauses=clauses,
                        risk_flags=risk_flags,
                        agent_run_id=agent_run_id,
                    )
                )
                await audit_agent_action(
                    session,
                    caller=caller,
                    action="decision.draft_saved",
                    entity_type="claim_decision",
                    entity_id=row.id,
                    details={
                        "claim_number": claim.claim_number,
                        "decision": decision.value,
                        "agent_run_id": agent_run_id,
                    },
                )
                return claim.id, {
                    "claim_number": claim.claim_number,
                    "decision_id": str(row.id),
                    "decision": decision.value,
                    "payable_amount": _money(row.payable_amount),
                    "status": "draft_saved_for_human_review",
                }

            return await run_idempotent(
                session,
                tool="save_decision_draft",
                key=idempotency_key,
                payload={
                    "claim_number": claim_number,
                    "decision": decision,
                    "payable_amount": payable_amount,
                    "deductions": [d.model_dump(mode="json") for d in deductions],
                    "reasons": reasons,
                    "cited_clauses": [c.chunk_id for c in cited_clauses],
                    "risk_flags": risk_flags,
                    "agent_run_id": agent_run_id,
                },
                caller=caller,
                work=work,
            )

    return mcp


async def _resolve_clauses(session: AsyncSession, chunk_ids: list[str]) -> list[CitedClause]:
    """Cited chunk_ids must exist; section and page come from our records, not the agent."""
    resolved: list[CitedClause] = []
    for raw in chunk_ids:
        try:
            chunk_id = UUID(raw)
        except ValueError as exc:
            raise ToolError(f"cited chunk_id {raw!r} is not a valid id.") from exc
        chunk = (
            await session.execute(select(PolicyChunk).where(PolicyChunk.id == chunk_id))
        ).scalar_one_or_none()
        if chunk is None:
            raise ToolError(
                f"cited chunk_id {raw} does not exist. "
                "Cite only chunk_ids returned by search_policy."
            )
        resolved.append(
            CitedClause(chunk_id=chunk.id, section_path=chunk.section_path, page=chunk.page_start)
        )
    return resolved


def main() -> None:
    run_cli(NAME, create_server)


if __name__ == "__main__":
    main()
