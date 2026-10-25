"""Decision node. The decision TYPE and every NUMBER come from fixed rules and the calculator; the
LLM only phrases the reasons, and its text is rejected if it contains a number it was not given."""

import json
import logging
import re
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.calculator import PolicyRules, check_eligibility, compute_payout
from app.agents.decision_rules import Verdict, decide
from app.agents.deps import AgentDeps
from app.agents.facts import build_facts
from app.agents.llm import ask_structured
from app.agents.prompts import load_prompt
from app.agents.state import ClaimState, CoverageCheck, DecisionDraft, RiskAssessment
from app.agents.tools import toolbox_for

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


class ReasonsLLM(BaseModel):
    reasons: list[str] = Field(description="2-5 short reasons", max_length=5)


def _numbers(text: str) -> set[str]:
    found = set()
    for raw in NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            found.add(str(Decimal(cleaned).normalize()))
        except ArithmeticError:
            continue
    return found


def reasons_are_grounded(reasons: list[str], given_text: str) -> bool:
    """Every number in the reasons must already appear in what the model was given."""
    return _numbers(" ".join(reasons)) <= _numbers(given_text)


def build_verdict(state: ClaimState, deps: AgentDeps) -> Verdict:
    rules = PolicyRules.model_validate(state.get("policy_rules") or {})
    coverage = [CoverageCheck.model_validate(c) for c in state.get("coverage") or []]
    risk = RiskAssessment.model_validate(state["risk"]) if state.get("risk") else None

    payout = None
    blockers: list = []
    extraction = state.get("extraction")
    if extraction and state.get("policy_rules"):
        facts, _ = build_facts(
            extraction,
            state.get("claim_facts") or {},
            state.get("customer_history") or {},
            rules,
        )
        if facts is not None:
            blockers = check_eligibility(rules, facts)
            payout = compute_payout(rules, facts)
    return decide(
        missing_documents=state.get("missing_documents") or [],
        policy_code=state.get("policy_code"),
        coverage=coverage,
        blockers=blockers,
        payout=payout,
        risk=risk,
        review_threshold=deps.settings.risk_review_threshold,
    )


def make_decision_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def decision(state: ClaimState) -> dict[str, Any]:
        verdict = build_verdict(state, deps)
        risk = RiskAssessment.model_validate(state["risk"]) if state.get("risk") else None
        flag_codes = [f.code for f in risk.flags] if risk else []

        given = {
            "decision": verdict.decision,
            "facts": verdict.facts,
            "payable_amount": verdict.payable_amount,
            "claimed_amount": verdict.claimed_amount,
            "deductions": [d.model_dump(mode="json") for d in verdict.deductions],
            "coverage_findings": [
                {
                    "check": c["check"],
                    "result": c["result"],
                    "explanation": c["explanation"],
                    "clauses": [x.get("section_path") for x in c.get("cited_clauses", [])],
                }
                for c in state.get("coverage") or []
                if c.get("applies")
            ],
            "risk_flags": [{"code": f.code, "detail": f.detail} for f in risk.flags]
            if risk
            else [],
        }
        given_text = json.dumps(given, default=str, indent=1)
        reasons, model_name = verdict.facts, None
        try:
            out, model_name = await ask_structured(
                deps.chat_model(),
                ReasonsLLM,
                [SystemMessage(content=load_prompt("decision")), HumanMessage(content=given_text)],
            )
            if out.reasons and reasons_are_grounded(out.reasons, given_text):
                reasons = [r.strip()[:480] for r in out.reasons if r.strip()]
            else:
                logger.warning("decision_reasons_rejected", extra={"run_id": state.get("run_id")})
        except Exception as exc:  # noqa: BLE001 - the deterministic facts are a fine fallback
            logger.warning("decision_reasons_llm_failed", extra={"error": type(exc).__name__})

        draft = DecisionDraft(
            decision=verdict.decision,
            payable_amount=verdict.payable_amount,
            claimed_amount=verdict.claimed_amount,
            deductions=verdict.deductions,
            reasons=reasons,
            cited_clauses=verdict.cited_clauses,
            risk_flags=flag_codes,
        )

        tools = await toolbox_for(deps.tools, "decision")
        run_id = state["run_id"]
        saved = await tools.call(
            "save_decision_draft",
            {
                "claim_number": state["claim_number"],
                "decision": draft.decision,
                "payable_amount": None
                if draft.payable_amount is None
                else str(draft.payable_amount),
                "deductions": [
                    {
                        "description": d.description[:200],
                        "amount": str(d.amount),
                        "clause": d.rule[:100],
                    }
                    for d in draft.deductions
                ],
                "reasons": draft.reasons,
                "cited_clauses": [{"chunk_id": c.chunk_id} for c in draft.cited_clauses],
                "risk_flags": draft.risk_flags,
                "agent_run_id": run_id,
                "idempotency_key": f"{run_id}-decision",
            },
        )
        draft.decision_id = saved.get("decision_id")
        await tools.call(
            "update_claim_status",
            {
                "claim_number": state["claim_number"],
                "new_status": "awaiting_approval",
                "reason": f"decision draft: {draft.decision}",
                "idempotency_key": f"{run_id}-awaiting",
            },
        )
        update: dict[str, Any] = {
            "decision": draft.model_dump(mode="json"),
            "completed_steps": ["decision"],
        }
        if model_name:
            update["models_used"] = {"decision": model_name}
        return update

    return decision
