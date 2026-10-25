"""Risk agent: fixed rules produce the flags and the score; the LLM only writes the summary and
may add a few justified extra flags (capped at medium severity)."""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.catalog import resolve_diagnosis
from app.agents.deps import AgentDeps
from app.agents.llm import ask_structured
from app.agents.prompts import load_prompt
from app.agents.risk_rules import risk_score, rule_flags
from app.agents.state import ClaimState, RiskAssessment, RiskFlag
from app.agents.tools import ToolFailure, toolbox_for
from app.core.redaction import redact

MAX_EXTRA_FLAGS = 3


class ExtraFlagLLM(BaseModel):
    code: str = Field(description="snake_case code")
    detail: str
    justification: str = Field(description="what in the facts supports this flag")


class RiskLLMOutput(BaseModel):
    summary: str
    extra_flags: list[ExtraFlagLLM] = Field(default_factory=list)


def merge_extra_flags(flags: list[RiskFlag], extra: list[ExtraFlagLLM]) -> list[RiskFlag]:
    """LLM flags need a justification, cannot duplicate a rule flag, and are never above medium."""
    have = {f.code for f in flags}
    merged = list(flags)
    for item in extra[:MAX_EXTRA_FLAGS]:
        code = "llm_" + item.code.strip().lower().replace(" ", "_")[:60]
        if not item.justification.strip() or not item.detail.strip() or code in have:
            continue
        have.add(code)
        merged.append(
            RiskFlag(code=code, severity="medium", detail=redact(item.detail)[:300], source="llm")
        )
    return merged


def make_risk_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def risk(state: ClaimState) -> dict[str, Any]:
        tools = await toolbox_for(deps.tools, "risk")
        ext = state.get("extraction") or {}
        try:
            history = await tools.call(
                "get_customer_history", {"claim_number": state["claim_number"]}
            )
        except ToolFailure:
            history = {}

        diagnosis = resolve_diagnosis(ext.get("icd10_code"), ext.get("diagnosis_text"))
        flags = rule_flags(
            extraction=ext,
            issues=state.get("consistency_issues") or [],
            history=history,
            diagnosis=diagnosis,
            settings=deps.settings,
        )
        facts = {
            "diagnosis": redact(str(ext.get("diagnosis_text") or "")),
            "bill_total": ext.get("bill_total"),
            "typical_range": [diagnosis.cost_min, diagnosis.cost_max] if diagnosis else None,
            "days_from_policy_start_to_admission": history.get(
                "days_from_policy_start_to_admission"
            ),
            "previous_claim_count": history.get("previous_claim_count"),
            "rule_flags": [f.model_dump() for f in flags],
        }
        output, model_name = await ask_structured(
            deps.chat_model(),
            RiskLLMOutput,
            [
                SystemMessage(content=load_prompt("risk")),
                HumanMessage(content=json.dumps(facts, default=str, indent=1)),
            ],
        )
        all_flags = (
            merge_extra_flags(flags, output.extra_flags) if deps.settings.risk_llm_flags else flags
        )
        assessment = RiskAssessment(
            flags=all_flags, score=risk_score(all_flags), summary=redact(output.summary)[:800]
        )
        return {
            "risk": assessment.model_dump(mode="json"),
            "models_used": {"risk": model_name or "unknown"},
            "completed_steps": ["risk"],
        }

    return risk
