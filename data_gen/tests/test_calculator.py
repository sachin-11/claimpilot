from datetime import date

import pytest

from data_gen.calculator import ClaimFacts, compute_payout, months_between
from data_gen.config import PerDayLimit, Policy, SubLimit


@pytest.fixture
def policy() -> Policy:
    return Policy(
        policy_id="POL-T",
        name="Test policy",
        source_pdf="t.pdf",
        sum_insured_options=[500000],
        room_rent_limit=PerDayLimit(percent_of_si_per_day=1.0, max_per_day=5000),
        icu_limit=PerDayLimit(percent_of_si_per_day=2.0, max_per_day=10000),
        co_pay_percent=10,
        initial_waiting_days=30,
        specific_disease_waiting={"cataract": 24},
        pre_existing_waiting_months=36,
        sub_limits={"cataract": SubLimit(amount=30000, unit="per_eye")},
        exclusions=["cosmetic_surgery", "dental_unless_accident"],
    )


def _facts(**overrides) -> ClaimFacts:
    base = {
        "sum_insured": 500000,
        "diagnosis_key": "appendicitis",
        "policy_start_date": date(2022, 1, 1),
        "admission_date": date(2025, 6, 1),
        "room_rate_per_day": 4000,
        "charges": {"room_rent": 12000, "professional_fees": 30000, "medicines": 20000},
    }
    base.update(overrides)
    return ClaimFacts(**base)


def test_full_cover(policy: Policy) -> None:
    policy = policy.model_copy(update={"co_pay_percent": 0})
    result = compute_payout(policy, _facts())
    assert result.outcome == "approve"
    assert result.payable_amount == result.claimed_amount == 62000
    assert result.deductions == []


def test_room_rent_proportionate_deduction(policy: Policy) -> None:
    policy = policy.model_copy(update={"co_pay_percent": 0})
    # limit = min(1% of 5L = 5000, 5000) = 5000; rate 10000 -> factor 0.5
    facts = _facts(
        room_rate_per_day=10000,
        charges={"room_rent": 30000, "professional_fees": 20000, "medicines": 10000},
    )
    result = compute_payout(policy, facts)
    assert result.outcome == "partial_approve"
    assert result.primary_rule == "room_rent_limit"
    # room 30000 -> 15000, fees 20000 -> 10000, medicines untouched
    assert result.payable_amount == 15000 + 10000 + 10000
    assert result.claimed_amount - sum(d.amount for d in result.deductions) == result.payable_amount


def test_co_pay(policy: Policy) -> None:
    result = compute_payout(policy, _facts())
    assert result.outcome == "partial_approve"
    assert result.primary_rule == "co_pay_percent"
    assert result.payable_amount == 62000 - 6200


def test_sub_limit_then_co_pay(policy: Policy) -> None:
    policy = policy.model_copy(update={"co_pay_percent": 10})
    facts = _facts(
        diagnosis_key="cataract",
        sub_limit_key="cataract",
        room_rate_per_day=2000,
        charges={"room_rent": 2000, "medicines": 58000},
    )
    result = compute_payout(policy, facts)
    rules = [d.rule for d in result.deductions]
    assert rules == ["sub_limits.cataract", "co_pay_percent"]
    # capped at 30000, then 10% co-pay
    assert result.payable_amount == 27000


def test_sub_limit_per_eye_scales_with_units(policy: Policy) -> None:
    policy = policy.model_copy(update={"co_pay_percent": 0})
    facts = _facts(
        sub_limit_key="cataract", units=2, charges={"medicines": 55000}, room_rate_per_day=0
    )
    assert compute_payout(policy, facts).payable_amount == 55000


def test_initial_waiting_rejection(policy: Policy) -> None:
    facts = _facts(policy_start_date=date(2025, 5, 20))
    result = compute_payout(policy, facts)
    assert result.outcome == "reject"
    assert result.rejection_rule == "initial_waiting_days"
    assert result.payable_amount == 0


def test_initial_waiting_waived_for_accident(policy: Policy) -> None:
    facts = _facts(policy_start_date=date(2025, 5, 28), is_accident=True)
    assert compute_payout(policy, facts).outcome != "reject"


def test_specific_disease_waiting_rejection(policy: Policy) -> None:
    facts = _facts(
        diagnosis_key="cataract",
        specific_wait_key="cataract",
        policy_start_date=date(2024, 3, 1),
    )
    result = compute_payout(policy, facts)
    assert result.outcome == "reject"
    assert result.rejection_rule == "specific_disease_waiting.cataract"


def test_pre_existing_waiting_rejection(policy: Policy) -> None:
    facts = _facts(ped_related=True, policy_start_date=date(2023, 6, 1))
    result = compute_payout(policy, facts)
    assert result.rejection_rule == "pre_existing_waiting_months"


def test_exclusion_rejection(policy: Policy) -> None:
    result = compute_payout(policy, _facts(exclusion_tag="cosmetic_surgery"))
    assert result.outcome == "reject"
    assert result.rejection_rule == "exclusions.cosmetic_surgery"
    assert result.deductions[0].amount == result.claimed_amount


def test_dental_excluded_unless_accident(policy: Policy) -> None:
    assert compute_payout(policy, _facts(exclusion_tag="dental_unless_accident")).outcome == (
        "reject"
    )
    covered = compute_payout(policy, _facts(exclusion_tag="dental_unless_accident", is_accident=True))
    assert covered.outcome != "reject"


def test_maternity_not_covered(policy: Policy) -> None:
    result = compute_payout(policy, _facts(is_maternity=True))
    assert result.rejection_rule == "maternity.covered"


def test_sum_insured_cap(policy: Policy) -> None:
    policy = policy.model_copy(update={"co_pay_percent": 0})
    facts = _facts(sum_insured=50000, room_rate_per_day=0, charges={"medicines": 80000})
    result = compute_payout(policy, facts)
    assert result.payable_amount == 50000
    assert result.primary_rule == "sum_insured"


def test_months_between() -> None:
    assert months_between(date(2024, 1, 15), date(2025, 1, 14)) == 11
    assert months_between(date(2024, 1, 15), date(2025, 1, 15)) == 12
