"""Intake node (no LLM agent): get the extraction, the claim's missing documents and the policy /
customer facts, and move the claim to `processing`."""

from typing import Any
from uuid import UUID

from app.agents.claim_form import load_claim_form_facts
from app.agents.deps import AgentDeps
from app.agents.state import ClaimState
from app.agents.tools import ToolFailure, toolbox_for

# claim statuses a run may start from
STARTABLE = {"submitted", "processing", "documents_requested"}


def make_intake_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def intake(state: ClaimState) -> dict[str, Any]:
        tools = await toolbox_for(deps.tools, "intake")
        number = state["claim_number"]
        claim_id = UUID(state["claim_id"])

        claim = await tools.call("get_claim", {"claim_number": number})
        if claim["status"] not in STARTABLE:
            raise ToolFailure(
                f"claim {number} is '{claim['status']}'; only submitted, processing or "
                "documents_requested claims can be processed"
            )

        try:
            extraction = await tools.call("get_claim_extraction", {"claim_number": number})
        except ToolFailure as exc:
            if "no extraction" not in str(exc) or deps.intake is None:
                raise
            result = await deps.intake.process(claim_id)  # Module 6: OCR + LLM extraction
            if getattr(result, "status", "") != "completed":
                raise ToolFailure(
                    f"intake failed: {getattr(result, 'error', 'unknown error')}"
                ) from exc
            extraction = await tools.call("get_claim_extraction", {"claim_number": number})
            claim = await tools.call("get_claim", {"claim_number": number})

        if claim["status"] != "processing":
            await tools.call(
                "update_claim_status",
                {
                    "claim_number": number,
                    "new_status": "processing",
                    "reason": "agent run started",
                    "idempotency_key": f"{state['run_id']}-processing",
                },
            )

        try:
            history = await tools.call("get_customer_history", {"claim_number": number})
        except ToolFailure:  # claim without a customer policy
            history = {}

        async with deps.session_factory() as session:
            form_facts = await load_claim_form_facts(session, deps.data_dir, claim_id)

        return {
            "extraction": {
                **extraction["extracted"],
                "confidence": extraction.get("confidence"),
            },
            "missing_documents": extraction.get("missing_documents") or [],
            "consistency_issues": extraction.get("consistency_issues") or [],
            "policy_code": claim.get("policy_code"),
            "customer_history": history,
            "claim_facts": form_facts,
            "completed_steps": ["intake"],
        }

    return intake
