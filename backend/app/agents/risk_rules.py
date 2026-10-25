"""Deterministic risk rules and score (pure). An LLM may add flags and write the summary, but the
score is always computed here from flag severities."""

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from app.agents.catalog import Diagnosis
from app.agents.config import AgentSettings
from app.agents.state import RiskFlag, Severity

SEVERITY_POINTS: dict[str, int] = {"low": 10, "medium": 25, "high": 60}

# Module 6 consistency issue type -> (risk code, severity). Other types (missing documents and
# fields, superseded copy, low OCR confidence) are about completeness, not risk.
CONSISTENCY_FLAGS: dict[str, tuple[str, Severity]] = {
    "date_mismatch": ("date_mismatch_between_documents", "high"),
    "discharge_before_admission": ("discharge_before_admission", "high"),
    "duplicate_line_items": ("duplicate_line_items", "high"),
    "claimed_amount_differs_from_bill": ("claimed_amount_differs_from_bill", "medium"),
    "patient_name_mismatch": ("patient_name_mismatch", "medium"),
    "policy_number_mismatch": ("policy_number_mismatch", "medium"),
}


def risk_score(flags: list[RiskFlag]) -> int:
    return min(100, sum(SEVERITY_POINTS[f.severity] for f in flags))


def rule_flags(
    *,
    extraction: dict[str, Any],
    issues: list[dict[str, Any]],
    history: dict[str, Any],
    diagnosis: Diagnosis | None,
    settings: AgentSettings,
) -> list[RiskFlag]:
    flags: dict[str, RiskFlag] = {}

    days = history.get("days_from_policy_start_to_admission")
    if days is not None and days < settings.early_admission_days:
        flags["admission_soon_after_policy_start"] = RiskFlag(
            code="admission_soon_after_policy_start",
            severity="high",
            detail=f"Admitted {days} days after the policy started "
            f"(threshold {settings.early_admission_days} days).",
        )

    bill = extraction.get("bill_total") or extraction.get("claimed_amount")
    if bill is not None and diagnosis is not None and Decimal(str(bill)) > diagnosis.cost_max:
        flags["bill_exceeds_typical_range"] = RiskFlag(
            code="bill_exceeds_typical_range",
            severity="high",
            detail=f"Bill Rs {Decimal(str(bill)):,.0f} is above the typical range for "
            f"{diagnosis.name} (Rs {diagnosis.cost_min:,} to Rs {diagnosis.cost_max:,}).",
        )

    for issue in issues:
        mapped = CONSISTENCY_FLAGS.get(str(issue.get("type")))
        if mapped and mapped[0] not in flags:
            flags[mapped[0]] = RiskFlag(
                code=mapped[0], severity=mapped[1], detail=str(issue.get("detail", ""))[:300]
            )

    recent = _recent_claims(history, extraction, settings.frequent_claims_window_days)
    if recent > settings.frequent_claims_count:
        flags["frequent_claims"] = RiskFlag(
            code="frequent_claims",
            severity="medium",
            detail=f"{recent} other claims on this policy within "
            f"{settings.frequent_claims_window_days} days.",
        )
    return list(flags.values())


def _recent_claims(history: dict[str, Any], extraction: dict[str, Any], window_days: int) -> int:
    admission = extraction.get("admission_date")
    if not admission:
        return int(history.get("previous_claim_count") or 0)
    anchor = date.fromisoformat(str(admission))
    since = anchor - timedelta(days=window_days)
    count = 0
    for claim in history.get("previous_claims") or []:
        when = claim.get("admission_date")
        if when and since <= date.fromisoformat(str(when)) <= anchor:
            count += 1
    return count
