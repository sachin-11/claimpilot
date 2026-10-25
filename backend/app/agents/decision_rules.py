"""Decision rules (pure). The decision type comes from here, never from an LLM.

Priority:
  1. missing documents / no policy         -> request_documents
  2. blocking rule (waiting period, exclusion):
       policy rules AND the cited wording agree  -> reject
       they disagree (or wording is unclear)     -> flag_for_review
  3. risk score >= threshold, or any applicable coverage check unclear -> flag_for_review
  4. deductions > 0                         -> partial_approve
  5. otherwise                              -> approve
"""

from dataclasses import dataclass, field
from decimal import Decimal

from app.agents.calculator import Blocker, PayoutResult
from app.agents.state import (
    BLOCKING_CHECKS,
    CitedClause,
    CoverageCheck,
    DecisionName,
    DeductionLine,
    RiskAssessment,
)

MAX_CITED = 20


@dataclass
class Verdict:
    decision: DecisionName
    payable_amount: Decimal | None
    claimed_amount: Decimal | None
    deductions: list[DeductionLine] = field(default_factory=list)
    cited_clauses: list[CitedClause] = field(default_factory=list)
    # Plain statements of the facts behind the decision, numbers included (calculator output).
    facts: list[str] = field(default_factory=list)


def _cited(coverage: list[CoverageCheck]) -> list[CitedClause]:
    seen: dict[str, CitedClause] = {}
    for check in coverage:
        for clause in check.cited_clauses:
            seen.setdefault(clause.chunk_id, clause)
    return list(seen.values())[:MAX_CITED]


def decide(
    *,
    missing_documents: list[str],
    policy_code: str | None,
    coverage: list[CoverageCheck],
    blockers: list[Blocker],
    payout: PayoutResult | None,
    risk: RiskAssessment | None,
    review_threshold: int,
) -> Verdict:
    claimed = payout.claimed_amount if payout else None
    cited = _cited(coverage)

    # 1. documents
    if missing_documents or not policy_code:
        facts = []
        if not policy_code:
            facts.append("The claim has no usable policy number.")
        if missing_documents:
            facts.append("Missing or unreadable: " + ", ".join(missing_documents) + ".")
        return Verdict("request_documents", None, claimed, facts=facts)

    applicable = [c for c in coverage if c.applies]

    # 2. blocking rules
    from_rules = {b.kind for b in blockers}
    from_wording = {
        c.check for c in applicable if c.check in BLOCKING_CHECKS and c.result == "fail"
    }
    if from_rules or from_wording:
        if from_rules == from_wording:
            lines = [b.reason for b in blockers]
            amount = claimed or Decimal(0)
            return Verdict(
                "reject",
                Decimal(0),
                claimed,
                deductions=[
                    DeductionLine(
                        rule=blockers[0].rule if blockers else "exclusion",
                        description="Not payable: " + "; ".join(lines),
                        amount=amount,
                    )
                ],
                cited_clauses=[c for c in cited if _is_failing(c, applicable)] or cited,
                facts=[f"Not payable: {line}." for line in lines],
            )
        return Verdict(
            "flag_for_review",
            payout.payable_amount if payout else None,
            claimed,
            deductions=_lines(payout),
            cited_clauses=cited,
            facts=[
                "The policy rules and the policy wording disagree: rules say "
                f"{sorted(from_rules) or 'no blocking rule'}, wording says "
                f"{sorted(from_wording) or 'no blocking rule'}. A reviewer must decide."
            ],
        )

    # 3. unclear / risk
    facts: list[str] = []
    unclear = [c.check for c in applicable if c.result == "unclear"]
    if unclear:
        facts.append("Coverage could not be confirmed for: " + ", ".join(unclear) + ".")
    if risk is not None and risk.score >= review_threshold:
        facts.append(f"Risk score {risk.score}/100 is at or above {review_threshold}.")
    if payout is None:
        facts.append("The sum insured is unknown, so no amount could be calculated.")
        return Verdict("flag_for_review", None, claimed, cited_clauses=cited, facts=facts)
    if facts:
        return Verdict(
            "flag_for_review",
            payout.payable_amount,
            claimed,
            deductions=_lines(payout),
            cited_clauses=cited,
            facts=facts + _deduction_facts(payout),
        )

    # 4 / 5
    if payout.deductions:
        return Verdict(
            "partial_approve",
            payout.payable_amount,
            claimed,
            deductions=_lines(payout),
            cited_clauses=cited,
            facts=_deduction_facts(payout),
        )
    return Verdict(
        "approve",
        payout.payable_amount,
        claimed,
        cited_clauses=cited,
        facts=[
            f"All charges are within policy limits; Rs {payout.payable_amount:,.2f} is payable."
        ],
    )


def _is_failing(clause: CitedClause, checks: list[CoverageCheck]) -> bool:
    return any(
        c.result == "fail" and any(cc.chunk_id == clause.chunk_id for cc in c.cited_clauses)
        for c in checks
    )


def _lines(payout: PayoutResult | None) -> list[DeductionLine]:
    if payout is None:
        return []
    return [
        DeductionLine(rule=d.rule, description=d.description, amount=d.amount)
        for d in payout.deductions
    ]


def _deduction_facts(payout: PayoutResult) -> list[str]:
    facts = [d.description + f": Rs {d.amount:,.2f} deducted." for d in payout.deductions]
    facts.append(
        f"Claimed Rs {payout.claimed_amount:,.2f}, payable Rs {payout.payable_amount:,.2f}."
    )
    return facts
