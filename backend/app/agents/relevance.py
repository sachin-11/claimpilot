"""Which coverage checks matter for THIS claim (pure, deterministic).

The coverage agent is asked to evaluate exactly these checks, each with a suggested search query.
Everything else is noise (e.g. "no exclusion applies" needs no clause when the diagnosis is not in
an excluded category) and is dropped, so a missing citation for an irrelevant check can no longer
push a clean claim into manual review."""

from dataclasses import dataclass

from app.agents.calculator import ClaimFacts, PayoutResult, PolicyRules
from app.agents.state import CoverageCheck


@dataclass(frozen=True)
class RelevantCheck:
    check: str
    why: str
    suggested_query: str


def _words(key: str) -> str:
    return key.replace("_unless_accident", "").replace("_", " ")


def relevant_checks(rules: PolicyRules, facts: ClaimFacts | None) -> list[RelevantCheck]:
    checks = [
        RelevantCheck(
            "initial_waiting",
            "every illness claim is subject to the initial waiting period (accidents are exempt)",
            "initial waiting period first thirty days",
        )
    ]
    if facts is None:
        return checks
    key = facts.specific_wait_key
    if key and key in rules.specific_disease_waiting:
        checks.append(
            RelevantCheck(
                "specific_disease_waiting",
                f"{key} has a specific waiting period",
                f"{_words(key)} specific waiting period",
            )
        )
    if facts.ped_related:
        checks.append(
            RelevantCheck(
                "pre_existing",
                "a declared pre-existing condition is related to this diagnosis",
                "pre-existing diseases waiting period",
            )
        )
    tag = facts.exclusion_tag
    if facts.is_maternity or (tag and tag in rules.exclusions):
        topic = "maternity" if facts.is_maternity else _words(tag or "")
        checks.append(
            RelevantCheck(
                "exclusion",
                f"the diagnosis or procedure belongs to the excluded category: {topic}",
                f"{topic} exclusion",
            )
        )
    if rules.room_rent_limit.per_day(facts.sum_insured) is not None and facts.room_rate_per_day > 0:
        checks.append(
            RelevantCheck(
                "room_rent",
                "the policy limits room rent per day",
                "room rent limit proportionate deduction",
            )
        )
    if facts.icu_days and rules.icu_limit.per_day(facts.sum_insured) is not None:
        checks.append(
            RelevantCheck("icu_limit", "the stay included ICU days", "ICU charges limit per day")
        )
    if facts.sub_limit_key and facts.sub_limit_key in rules.sub_limits:
        checks.append(
            RelevantCheck(
                "sub_limit",
                f"{facts.sub_limit_key} has a sub-limit",
                f"{_words(facts.sub_limit_key)} sub-limit",
            )
        )
    if rules.co_pay_percent > 0:
        checks.append(
            RelevantCheck("co_pay", "the policy has a co-payment", "co-payment percentage")
        )
    return checks


# Checks whose outcome is a number the calculator already decides.
AMOUNT_CHECKS: dict[str, str] = {
    "room_rent": "room_rent_limit",
    "icu_limit": "icu_limit",
    "sub_limit": "sub_limits.",
}


def reconcile_amount_checks(
    checks: list[CoverageCheck], payout: PayoutResult | None
) -> list[CoverageCheck]:
    """Room rent, ICU and sub-limit results follow the calculation: `partial` when the calculator
    deducts for that rule, `pass` when it does not. The model's wording and citations are kept;
    only a result that contradicts the numbers is corrected (`unclear` stays unclear)."""
    if payout is None:
        return checks
    fixed: list[CoverageCheck] = []
    for check in checks:
        rule = AMOUNT_CHECKS.get(check.check)
        if rule is None or check.result == "unclear":
            fixed.append(check)
            continue
        deducted = any(
            d.rule == rule or (rule.endswith(".") and d.rule.startswith(rule))
            for d in payout.deductions
        )
        want = "partial" if deducted else "pass"
        if (check.result == "pass") != (want == "pass"):
            note = (
                "The calculation applies a deduction for this rule."
                if deducted
                else "The calculation applies no deduction for this rule."
            )
            check = check.model_copy(
                update={"result": want, "explanation": f"{check.explanation} [{note}]"}
            )
        fixed.append(check)
    return fixed
