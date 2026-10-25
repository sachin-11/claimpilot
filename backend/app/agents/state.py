"""Graph state. Sub-models are Pydantic (validation, schemas for LLM structured output); the state
itself holds plain JSON dicts (`model_dump(mode="json")`) so checkpoints stay portable."""

import operator
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

CheckName = Literal[
    "initial_waiting",
    "specific_disease_waiting",
    "pre_existing",
    "exclusion",
    "room_rent",
    "icu_limit",
    "sub_limit",
    "co_pay",
    "other",
]
CheckResult = Literal["pass", "fail", "partial", "unclear"]
DecisionName = Literal[
    "approve", "partial_approve", "reject", "flag_for_review", "request_documents"
]
Severity = Literal["low", "medium", "high"]

# Checks whose failure rejects the claim (the rest only reduce the amount).
BLOCKING_CHECKS: frozenset[str] = frozenset(
    {"initial_waiting", "specific_disease_waiting", "pre_existing", "exclusion"}
)


class CitedClause(BaseModel):
    chunk_id: str
    section_path: str | None = None
    pages: str | None = None  # "8" or "8-9"


class CoverageCheck(BaseModel):
    check: CheckName
    applies: bool
    result: CheckResult
    explanation: str
    cited_clauses: list[CitedClause] = Field(default_factory=list)


class RiskFlag(BaseModel):
    code: str
    severity: Severity
    detail: str
    source: Literal["rule", "llm"] = "rule"


class RiskAssessment(BaseModel):
    flags: list[RiskFlag] = Field(default_factory=list)
    score: int = Field(default=0, ge=0, le=100)
    summary: str = ""


class DeductionLine(BaseModel):
    rule: str
    description: str
    amount: Decimal


class DecisionDraft(BaseModel):
    decision: DecisionName
    payable_amount: Decimal | None = None
    claimed_amount: Decimal | None = None
    deductions: list[DeductionLine] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cited_clauses: list[CitedClause] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    decision_id: str | None = None  # claim_decisions row written through MCP


class HumanApproval(BaseModel):
    action: Literal["approved", "rejected", "edited"]
    final_amount: Decimal | None = Field(default=None, ge=0)
    comment: str | None = None
    reviewer: str  # reviewer id (users.id)

    @model_validator(mode="after")
    def edited_needs_amount_and_comment(self) -> "HumanApproval":
        if self.action == "edited":
            if self.final_amount is None:
                raise ValueError("an edited approval needs final_amount")
            if not (self.comment or "").strip():
                raise ValueError("an edited approval needs a comment")
        return self


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class ClaimState(TypedDict, total=False):
    # identity
    claim_number: str
    claim_id: str
    run_id: str
    # intake
    extraction: dict[str, Any] | None
    claim_facts: dict[str, Any]  # pre_existing_conditions, is_accident (from the claim form)
    missing_documents: list[str]
    consistency_issues: list[dict[str, Any]]
    customer_history: dict[str, Any]
    policy_code: str | None
    policy_rules: dict[str, Any]
    # agents
    coverage: list[dict[str, Any]]
    risk: dict[str, Any] | None
    decision: dict[str, Any] | None
    approval: dict[str, Any] | None
    notification: dict[str, Any] | None
    # control
    completed_steps: Annotated[list[str], operator.add]
    next_steps: list[str]
    errors: Annotated[list[dict[str, str]], operator.add]
    models_used: Annotated[dict[str, str], merge_dicts]
