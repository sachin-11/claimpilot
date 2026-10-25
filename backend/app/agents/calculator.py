"""Deterministic payable-amount calculator. Pure: policy rules + claim facts in, amounts out.
No LLM, no database, no data_gen.

Order (each step works on what the previous one left):
  1. room-rent proportionate deduction (room rent + the policy's proportionate categories),
  2. ICU per-day cap,
  3. sub-limit of the diagnosis,
  4. remaining sum insured cap,
  5. co-pay on what is left.
`check_eligibility` is separate: it lists the BLOCKING rules (exclusion, waiting periods) the
claim violates, and the decision node compares that with the coverage agent's findings.
"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pydantic import BaseModel, Field

CENT = Decimal("0.01")


def money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


class Limit(BaseModel):
    percent_of_si_per_day: float | None = None
    max_per_day: float | None = None

    def per_day(self, sum_insured: Decimal) -> Decimal | None:
        """Per-day cap: the lower of percent-of-SI and the absolute max; None = no limit."""
        caps = []
        if self.percent_of_si_per_day is not None:
            caps.append(sum_insured * Decimal(str(self.percent_of_si_per_day)) / 100)
        if self.max_per_day is not None:
            caps.append(Decimal(str(self.max_per_day)))
        return money(min(caps)) if caps else None


class SubLimit(BaseModel):
    amount: float
    unit: str = "per_admission"  # per_eye | per_admission


class Maternity(BaseModel):
    covered: bool = False
    waiting_months: int = 0


class PolicyRules(BaseModel):
    """The `policies.rules` JSON. Unknown keys are ignored."""

    room_rent_limit: Limit = Field(default_factory=Limit)
    icu_limit: Limit = Field(default_factory=Limit)
    co_pay_percent: float = 0
    initial_waiting_days: int = 0
    specific_disease_waiting: dict[str, int] = Field(default_factory=dict)
    pre_existing_waiting_months: int = 0
    sub_limits: dict[str, SubLimit] = Field(default_factory=dict)
    exclusions: list[str] = Field(default_factory=list)
    maternity: Maternity = Field(default_factory=Maternity)
    proportionate_categories: list[str] = Field(
        default_factory=lambda: ["room_rent", "professional_fees", "ot_charges"]
    )


class ClaimFacts(BaseModel):
    sum_insured: Decimal
    remaining_sum_insured: Decimal | None = None  # None = nothing used yet
    policy_start_date: date | None = None
    admission_date: date | None = None
    is_accident: bool = False
    is_maternity: bool = False
    ped_related: bool = False
    exclusion_tag: str | None = None
    specific_wait_key: str | None = None
    sub_limit_key: str | None = None
    units: int = 1  # eyes operated etc., for per_eye sub-limits
    room_rate_per_day: Decimal = Decimal(0)
    icu_rate_per_day: Decimal = Decimal(0)
    icu_days: int = 0
    # billed amount per rule category: room_rent, icu, professional_fees, ot_charges, other...
    charges: dict[str, Decimal] = Field(default_factory=dict)
    room_tax: Decimal = Decimal(0)  # tax billed on the room; follows the room-rent deduction
    claimed_amount: Decimal | None = None  # defaults to charges + room_tax

    @property
    def total_billed(self) -> Decimal:
        if self.claimed_amount is not None:
            return money(self.claimed_amount)
        return money(sum(self.charges.values(), Decimal(0)) + self.room_tax)


class Deduction(BaseModel):
    rule: str
    description: str
    amount: Decimal


class PayoutResult(BaseModel):
    claimed_amount: Decimal
    payable_amount: Decimal
    deductions: list[Deduction]


class Blocker(BaseModel):
    kind: str  # initial_waiting | specific_disease_waiting | pre_existing | exclusion
    rule: str
    reason: str


def months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def check_eligibility(rules: PolicyRules, facts: ClaimFacts) -> list[Blocker]:
    """Blocking rules this claim violates according to the policy rules (empty = none)."""
    found: list[Blocker] = []
    tag = facts.exclusion_tag
    if (
        tag
        and tag in rules.exclusions
        and not (tag.endswith("_unless_accident") and facts.is_accident)
    ):
        found.append(
            Blocker(kind="exclusion", rule=f"exclusions.{tag}", reason=f"{tag} is excluded")
        )
    if facts.policy_start_date is None or facts.admission_date is None:
        return found
    days = (facts.admission_date - facts.policy_start_date).days
    months = months_between(facts.policy_start_date, facts.admission_date)

    if facts.is_maternity:
        if not rules.maternity.covered:
            found.append(
                Blocker(kind="exclusion", rule="maternity.covered", reason="maternity not covered")
            )
        elif months < rules.maternity.waiting_months:
            found.append(
                Blocker(
                    kind="specific_disease_waiting",
                    rule="maternity.waiting_months",
                    reason=f"maternity waiting is {rules.maternity.waiting_months} months, "
                    f"{months} elapsed",
                )
            )
    if not facts.is_accident and days < rules.initial_waiting_days:
        found.append(
            Blocker(
                kind="initial_waiting",
                rule="initial_waiting_days",
                reason=f"admitted {days} days after policy start; initial waiting is "
                f"{rules.initial_waiting_days} days",
            )
        )
    key = facts.specific_wait_key
    if key and key in rules.specific_disease_waiting:
        required = rules.specific_disease_waiting[key]
        if months < required:
            found.append(
                Blocker(
                    kind="specific_disease_waiting",
                    rule=f"specific_disease_waiting.{key}",
                    reason=f"{key} waiting is {required} months, {months} elapsed",
                )
            )
    if facts.ped_related and months < rules.pre_existing_waiting_months:
        found.append(
            Blocker(
                kind="pre_existing",
                rule="pre_existing_waiting_months",
                reason=f"pre-existing waiting is {rules.pre_existing_waiting_months} months, "
                f"{months} elapsed",
            )
        )
    return found


def compute_payout(rules: PolicyRules, facts: ClaimFacts) -> PayoutResult:
    claimed = facts.total_billed
    deductions: list[Deduction] = []

    room_limit = rules.room_rent_limit.per_day(facts.sum_insured)
    if room_limit is not None and facts.room_rate_per_day > room_limit:
        factor = room_limit / facts.room_rate_per_day
        cut = Decimal(0)
        for category in rules.proportionate_categories:
            base = facts.charges.get(category, Decimal(0))
            if category == "room_rent":
                base += facts.room_tax
            cut += base - money(base * factor)
        if cut > 0:
            deductions.append(
                Deduction(
                    rule="room_rent_limit",
                    description=(
                        f"Room rent Rs {facts.room_rate_per_day:,.0f}/day is above the limit of "
                        f"Rs {room_limit:,.0f}/day, so "
                        f"{', '.join(rules.proportionate_categories)} are reduced by "
                        f"{(1 - factor) * 100:.1f}%"
                    ),
                    amount=money(cut),
                )
            )

    icu_limit = rules.icu_limit.per_day(facts.sum_insured)
    if icu_limit is not None and facts.icu_days and facts.icu_rate_per_day > icu_limit:
        excess = min(
            (facts.icu_rate_per_day - icu_limit) * facts.icu_days,
            facts.charges.get("icu", Decimal(0)),
        )
        if excess > 0:
            deductions.append(
                Deduction(
                    rule="icu_limit",
                    description=(
                        f"ICU Rs {facts.icu_rate_per_day:,.0f}/day is above the limit of "
                        f"Rs {icu_limit:,.0f}/day for {facts.icu_days} day(s)"
                    ),
                    amount=money(excess),
                )
            )

    running = claimed - sum((d.amount for d in deductions), Decimal(0))

    sub = rules.sub_limits.get(facts.sub_limit_key) if facts.sub_limit_key else None
    if sub is not None:
        cap = money(Decimal(str(sub.amount)) * (facts.units if sub.unit == "per_eye" else 1))
        if running > cap:
            deductions.append(
                Deduction(
                    rule=f"sub_limits.{facts.sub_limit_key}",
                    description=f"Sub-limit for {facts.sub_limit_key} is Rs {cap:,.0f} "
                    f"({sub.unit}); eligible amount was Rs {running:,.0f}",
                    amount=money(running - cap),
                )
            )
            running = cap

    available = facts.remaining_sum_insured
    if available is None:
        available = facts.sum_insured
    if running > available:
        deductions.append(
            Deduction(
                rule="sum_insured",
                description=f"Eligible amount is above the remaining sum insured of "
                f"Rs {available:,.0f}",
                amount=money(running - available),
            )
        )
        running = money(available)

    if rules.co_pay_percent > 0:
        co_pay = money(running * Decimal(str(rules.co_pay_percent)) / 100)
        if co_pay > 0:
            deductions.append(
                Deduction(
                    rule="co_pay_percent",
                    description=f"{rules.co_pay_percent:g}% co-pay on Rs {running:,.0f}",
                    amount=co_pay,
                )
            )
            running -= co_pay

    return PayoutResult(
        claimed_amount=claimed, payable_amount=money(running), deductions=deductions
    )
