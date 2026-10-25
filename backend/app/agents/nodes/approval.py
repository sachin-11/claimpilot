"""Approval node: pause the run for a human.

`interrupt()` stops the graph and surfaces the review payload; when the approval API resumes the
run, `interrupt()` returns the reviewer's input. On resume LangGraph re-runs this node from the
top, so nothing with a side effect may come before the interrupt call."""

from typing import Any

from langgraph.types import interrupt

from app.agents.state import ClaimState, HumanApproval


def review_payload(state: ClaimState) -> dict[str, Any]:
    ext = state.get("extraction") or {}
    return {
        "claim_number": state.get("claim_number"),
        "claim_id": state.get("claim_id"),
        "run_id": state.get("run_id"),
        "decision": state.get("decision"),
        "coverage": state.get("coverage") or [],
        "risk": state.get("risk"),
        "extraction_highlights": {
            k: ext.get(k)
            for k in (
                "diagnosis_text",
                "icd10_code",
                "procedure",
                "admission_date",
                "discharge_date",
                "room_type",
                "room_rent_per_day",
                "icu_days",
                "bill_total",
                "claimed_amount",
            )
        },
        "missing_documents": state.get("missing_documents") or [],
    }


async def approval(state: ClaimState) -> dict[str, Any]:
    answer = interrupt(review_payload(state))
    human = HumanApproval.model_validate(answer)  # edited needs final_amount + comment
    return {"approval": human.model_dump(mode="json"), "completed_steps": ["approval"]}
