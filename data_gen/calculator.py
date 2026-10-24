"""Pure payable-amount calculator. Reads only policy rules + claim facts.

Order of evaluation:
  1. Eligibility (reject): exclusion, maternity, initial waiting, specific disease
     waiting, pre-existing waiting.
  2. Room rent proportionate deduction (room + associated categories).
  3. ICU per-day cap.
  4. Sub-limit for the diagnosis.
  5. Sum insured cap.
  6. Co-pay on what remains.
All amounts are whole rupees, so claimed - sum(deductions) == payable exactly.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from data_gen.config import Policy
from data_gen.models import Deduction


class ClaimFacts(BaseModel):
    sum_insured: int
    diagnosis_key: str
    policy_start_date: date
    admission_date: date
    is_accident: bool = False
    is_maternity: bool = False
    ped_related: bool = False
    exclusion_tag: str | None = None
    specific_wait_key: str | None = None
    sub_limit_key: str | None = None
    units: int = 1  # e.g. number of eyes operated, for per_eye sub-limits
    room_rate_per_day: int = 0
    icu_rate_per_day: int = 0
    icu_days: int = 0
    # billed amount per category, e.g. {"room_rent": 9000, "medicines": 12000}
    charges: dict[str, int] = Field(default_factory=dict)
    room_gst: int = 0


class PayoutResult(BaseModel):
    outcome: Literal["approve", "partial_approve", "reject"]
    claimed_amount: int
    payable_amount: int
    deductions: list[Deduction]
    primary_rule: str
    reasons: list[str]
    rejection_rule: str | None = None


def months_between(start: date, end: date) -> int:
    """Whole months elapsed from start to end (never negative)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def _reject(claimed: int, rule: str, reason: str) -> PayoutResult:
    return PayoutResult(
        outcome="reject",
        claimed_amount=claimed,
        payable_amount=0,
        deductions=[Deduction(rule=rule, description=reason, amount=claimed)],
        primary_rule=rule,
        reasons=[reason],
        rejection_rule=rule,
    )


def _check_eligibility(policy: Policy, facts: ClaimFacts, claimed: int) -> PayoutResult | None:
    tag = facts.exclusion_tag
    if tag and tag in policy.exclusions:
        if not (tag.endswith("_unless_accident") and facts.is_accident):
            return _reject(claimed, f"exclusions.{tag}", f"Treatment excluded by policy: {tag}")

    elapsed_months = months_between(facts.policy_start_date, facts.admission_date)
    elapsed_days = (facts.admission_date - facts.policy_start_date).days

    if facts.is_maternity:
        if not policy.maternity.covered:
            return _reject(claimed, "maternity.covered", "Maternity is not covered by this policy")
        if elapsed_months < policy.maternity.waiting_months:
            return _reject(
                claimed,
                "maternity.waiting_months",
                f"Maternity waiting period of {policy.maternity.waiting_months} months not "
                f"completed ({elapsed_months} months elapsed)",
            )

    if not facts.is_accident and elapsed_days < policy.initial_waiting_days:
        return _reject(
            claimed,
            "initial_waiting_days",
            f"Admission {elapsed_days} days after policy start, within the "
            f"{policy.initial_waiting_days}-day initial waiting period",
        )

    wait_key = facts.specific_wait_key
    if wait_key and wait_key in policy.specific_disease_waiting:
        required = policy.specific_disease_waiting[wait_key]
        if elapsed_months < required:
            return _reject(
                claimed,
                f"specific_disease_waiting.{wait_key}",
                f"{wait_key} has a {required}-month waiting period; only {elapsed_months} "
                "months elapsed",
            )

    if facts.ped_related and elapsed_months < policy.pre_existing_waiting_months:
        return _reject(
            claimed,
            "pre_existing_waiting_months",
            f"Pre-existing condition waiting period of {policy.pre_existing_waiting_months} "
            f"months not completed ({elapsed_months} months elapsed)",
        )
    return None


def compute_payout(policy: Policy, facts: ClaimFacts) -> PayoutResult:
    claimed = sum(facts.charges.values()) + facts.room_gst

    rejection = _check_eligibility(policy, facts, claimed)
    if rejection is not None:
        return rejection

    deductions: list[Deduction] = []

    # Room rent proportionate deduction
    room_limit = policy.room_rent_limit.limit_per_day(facts.sum_insured)
    if room_limit is not None and facts.room_rate_per_day > room_limit:
        factor = room_limit / facts.room_rate_per_day
        deducted = 0
        for category in policy.proportionate_categories:
            base = facts.charges.get(category, 0)
            if category == "room_rent":
                base += facts.room_gst
            deducted += base - round(base * factor)
        if deducted > 0:
            deductions.append(
                Deduction(
                    rule="room_rent_limit",
                    description=(
                        f"Room rent Rs {facts.room_rate_per_day}/day exceeds limit Rs "
                        f"{room_limit}/day; proportionate deduction of {(1 - factor) * 100:.1f}% "
                        f"on {', '.join(policy.proportionate_categories)}"
                    ),
                    amount=deducted,
                )
            )

    # ICU per-day cap
    icu_limit = policy.icu_limit.limit_per_day(facts.sum_insured)
    if icu_limit is not None and facts.icu_days and facts.icu_rate_per_day > icu_limit:
        excess = min(
            (facts.icu_rate_per_day - icu_limit) * facts.icu_days, facts.charges.get("icu", 0)
        )
        deductions.append(
            Deduction(
                rule="icu_limit",
                description=(
                    f"ICU rent Rs {facts.icu_rate_per_day}/day exceeds limit Rs {icu_limit}/day "
                    f"for {facts.icu_days} day(s)"
                ),
                amount=excess,
            )
        )

    running = claimed - sum(d.amount for d in deductions)

    # Sub-limit
    sub_limit = policy.sub_limits.get(facts.sub_limit_key) if facts.sub_limit_key else None
    if sub_limit is not None:
        cap = sub_limit.amount * (facts.units if sub_limit.unit == "per_eye" else 1)
        if running > cap:
            deductions.append(
                Deduction(
                    rule=f"sub_limits.{facts.sub_limit_key}",
                    description=(
                        f"Sub-limit for {facts.sub_limit_key} is Rs {cap} ({sub_limit.unit}); "
                        f"eligible amount was Rs {running}"
                    ),
                    amount=running - cap,
                )
            )
            running = cap

    # Sum insured cap
    if running > facts.sum_insured:
        deductions.append(
            Deduction(
                rule="sum_insured",
                description=f"Eligible amount exceeds sum insured Rs {facts.sum_insured}",
                amount=running - facts.sum_insured,
            )
        )
        running = facts.sum_insured

    # Co-pay
    if policy.co_pay_percent > 0:
        co_pay = round(running * policy.co_pay_percent / 100)
        if co_pay > 0:
            deductions.append(
                Deduction(
                    rule="co_pay_percent",
                    description=f"{policy.co_pay_percent:g}% co-pay on Rs {running}",
                    amount=co_pay,
                )
            )
            running -= co_pay

    if not deductions:
        return PayoutResult(
            outcome="approve",
            claimed_amount=claimed,
            payable_amount=claimed,
            deductions=[],
            primary_rule="base_coverage",
            reasons=["All charges within policy limits; no waiting period or exclusion applies"],
        )

    primary = max(deductions, key=lambda d: d.amount)
    return PayoutResult(
        outcome="partial_approve",
        claimed_amount=claimed,
        payable_amount=running,
        deductions=deductions,
        primary_rule=primary.rule,
        reasons=[d.description for d in deductions],
    )
