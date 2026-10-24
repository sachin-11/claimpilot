"""Scenario planner: turns (count, seed, mix) into fully specified claims + golden records.

Each claim is built from a seeded RNG. A candidate is re-rolled (new attempt seed) until the
calculator confirms it really lands in the requested scenario, e.g. a "rejected / specific
waiting" claim must really be rejected by that waiting rule.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import date, timedelta

from faker import Faker

from data_gen.calculator import ClaimFacts, PayoutResult, compute_payout
from data_gen.config import Diagnosis, Policy, PolicyRules, TypicalCosts
from data_gen.models import (
    BillLine,
    ClaimSpec,
    Decision,
    Deduction,
    GoldenRecord,
    Hospital,
    Patient,
    PlannedClaim,
    ScenarioType,
)

logger = logging.getLogger(__name__)

DEFAULT_MIX: dict[str, int] = {"covered": 15, "partial": 8, "rejected": 7, "risk": 6, "incomplete": 4}
MAX_ATTEMPTS = 300

PARTIAL_CYCLE = ["room_rent", "copay", "sub_limit"]
REJECTED_CYCLE = [
    "initial_waiting",
    "specific_waiting:cataract",
    "specific_waiting:hernia",
    "pre_existing",
    "exclusion:cosmetic_surgery",
    "exclusion:obesity_treatment",
    "exclusion:dental_unless_accident",
    "specific_waiting:knee_replacement",
    "specific_waiting:cholecystectomy",
]
RISK_CYCLE = [
    "high_bill",
    "early_admission",
    "date_mismatch",
    "duplicate_items",
    "discharge_before_admission",
    "high_bill",
]
INCOMPLETE_CYCLE = [
    "missing_discharge_summary",
    "blank_field",
    "missing_policy_number",
    "unreadable_bill_total",
]

RISK_FLAG_BY_SUBTYPE = {
    "high_bill": "bill_exceeds_typical_range",
    "early_admission": "admission_soon_after_policy_start",
    "date_mismatch": "date_mismatch_between_documents",
    "duplicate_items": "duplicate_line_items",
    "discharge_before_admission": "discharge_before_admission",
}
RISK_REASON = {
    "bill_exceeds_typical_range": "Bill total is far above the typical cost range for this diagnosis",
    "admission_soon_after_policy_start": "Admission occurred within days of policy inception",
    "date_mismatch_between_documents": "Admission/discharge dates differ between bill and discharge summary",
    "duplicate_line_items": "Bill contains duplicated line items",
    "discharge_before_admission": "Discharge date is earlier than the admission date",
}

HOSPITALS = [
    ("Lotus Care Multispeciality Hospital", "Pune"),
    ("Shree Dhanvantari Hospital", "Mumbai"),
    ("Navjeevan Medical Centre", "Bengaluru"),
    ("Aastha Super Speciality Hospital", "Ahmedabad"),
    ("Sanjivani Heart & General Hospital", "Nagpur"),
    ("Kaveri Institute of Medical Sciences", "Chennai"),
    ("Ganga Jamuna Multispeciality Hospital", "Lucknow"),
    ("Medicare Plus Hospital", "Hyderabad"),
    ("Arya Vaidya Hospital", "Jaipur"),
    ("Sunrise Life Hospital", "Kolkata"),
    ("Vedanta Health Institute", "Delhi"),
    ("Prakash Nursing & Maternity Home", "Indore"),
]

MEDICINES = [
    "Injection Ceftriaxone 1 g",
    "Tablet Paracetamol 650 mg",
    "IV Fluids (NS / RL) 500 ml",
    "Injection Pantoprazole 40 mg",
    "Tablet Ondansetron 4 mg",
    "Injection Tramadol 50 mg",
    "Oral Rehydration & Supplements",
]
CONSUMABLES = ["Disposables & Consumables", "IV Cannula, Syringes & Infusion Sets", "Dressing Material"]

PROFESSIONAL = {
    "surgical": ["Surgeon Fee", "Anaesthetist Fee", "Assistant Surgeon Fee"],
    "daycare": ["Surgeon Fee", "Anaesthetist Fee"],
    "medical": ["Consultant Physician Charges", "Resident Medical Officer Charges"],
}
# category -> share of the non-room, non-ICU part of the bill
SHARES = {
    "surgical": {
        "professional_fees": 0.20,
        "ot_charges": 0.12,
        "medicines": 0.18,
        "consumables": 0.18,
        "diagnostics": 0.10,
        "other": 0.04,
    },
    "daycare": {
        "professional_fees": 0.22,
        "ot_charges": 0.12,
        "medicines": 0.10,
        "consumables": 0.30,
        "diagnostics": 0.10,
        "other": 0.03,
    },
    "medical": {
        "professional_fees": 0.18,
        "medicines": 0.38,
        "consumables": 0.10,
        "diagnostics": 0.25,
        "other": 0.05,
    },
}


class PlanningError(RuntimeError):
    """The policy/cost config cannot produce the requested scenario."""


# --------------------------------------------------------------------------- helpers


def resolve_mix(count: int, override: dict[str, int] | None = None) -> dict[str, int]:
    """Scale DEFAULT_MIX to `count` (largest remainder), unless an explicit mix is given."""
    if override:
        return {k: int(override.get(k, 0)) for k in DEFAULT_MIX}
    total = sum(DEFAULT_MIX.values())
    exact = {k: count * v / total for k, v in DEFAULT_MIX.items()}
    mix = {k: math.floor(v) for k, v in exact.items()}
    leftovers = sorted(exact, key=lambda k: exact[k] - mix[k], reverse=True)
    for key in leftovers[: count - sum(mix.values())]:
        mix[key] += 1
    return mix


def shift_months(d: date, months: int) -> date:
    index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(index, 12)
    day = min(d.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month])
    return date(year, month + 1, day)


def _round_up(value: float, step: int = 100) -> int:
    return int(math.ceil(value / step) * step)


def _room_type(rate: int) -> str:
    if rate <= 2500:
        return "General Ward"
    if rate <= 4000:
        return "Semi-Private Room"
    if rate <= 7000:
        return "Private Room"
    return "Deluxe Room"


def _split(rng: random.Random, category: str, total: int, names: list[str], qty_max: int = 1) -> list[BillLine]:
    weights = [rng.uniform(0.6, 1.4) for _ in names]
    scale = total / sum(weights)
    lines = []
    for name, weight in zip(names, weights, strict=True):
        share = max(int(weight * scale), 100)
        qty = rng.randint(1, qty_max) if qty_max > 1 else 1
        rate = max(round(share / qty), 1)
        lines.append(BillLine(category=category, description=name, qty=qty, rate=rate, amount=qty * rate))
    return lines


def build_bill(
    rng: random.Random,
    diag: Diagnosis,
    *,
    room_rate: int,
    room_days: int,
    icu_rate: int,
    icu_days: int,
    target: int,
) -> tuple[list[BillLine], int]:
    """Itemised bill lines + GST on room rent (5% when room rent exceeds Rs 5,000/day)."""
    lines: list[BillLine] = []
    room_total = room_rate * room_days
    lines.append(
        BillLine(
            category="room_rent",
            description=f"Room Rent - {_room_type(room_rate)}",
            qty=room_days,
            rate=room_rate,
            amount=room_total,
        )
    )
    if icu_days:
        lines.append(
            BillLine(category="icu", description="ICU Charges", qty=icu_days, rate=icu_rate, amount=icu_rate * icu_days)
        )
    gst = round(room_total * 0.05) if room_rate > 5000 else 0
    rest = max(target - room_total - icu_rate * icu_days - gst, int(target * 0.35))

    shares = SHARES[diag.kind]
    noisy = {c: s * rng.uniform(0.85, 1.15) for c, s in shares.items()}
    norm = sum(noisy.values())
    amounts = {c: int(rest * s / norm) for c, s in noisy.items()}

    prof_names = list(PROFESSIONAL[diag.kind])
    if diag.procedure and diag.kind != "medical":
        prof_names[0] = f"Surgeon Fee - {diag.procedure}"
    lines += _split(rng, "professional_fees", amounts["professional_fees"], prof_names)
    if "ot_charges" in amounts:
        lines += _split(rng, "ot_charges", amounts["ot_charges"], ["Operation Theatre Charges"])
    lines += _split(rng, "medicines", amounts["medicines"], rng.sample(MEDICINES, rng.randint(3, 4)), qty_max=10)

    cons_total = amounts["consumables"]
    if diag.implant:
        implant_amount = int(cons_total * 0.6)
        lines += _split(rng, "consumables", implant_amount, [diag.implant])
        cons_total -= implant_amount
    lines += _split(rng, "consumables", cons_total, rng.sample(CONSUMABLES, 2))

    tests = [*diag.tests, "Radiology / Imaging"][:4]
    lines += _split(rng, "diagnostics", amounts["diagnostics"], tests)
    lines += _split(rng, "other", amounts["other"], ["Nursing & Service Charges", "Registration & Admission Charges"])
    return lines, gst


def _charges(lines: list[BillLine], *, include_duplicates: bool) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in lines:
        if line.is_duplicate and not include_duplicates:
            continue
        out[line.category] = out.get(line.category, 0) + line.amount
    return out


def _add_duplicates(rng: random.Random, lines: list[BillLine]) -> list[BillLine]:
    candidates = [i for i, ln in enumerate(lines) if ln.category in ("medicines", "diagnostics")]
    picks = sorted(rng.sample(candidates, 2), reverse=True)
    out = list(lines)
    for index in picks:
        out.insert(index + 1, out[index].model_copy(update={"is_duplicate": True}))
    return out


def _fake_gstin(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return (
        f"{rng.randint(10, 36)}{''.join(rng.choices(letters, k=5))}{rng.randint(1000, 9999)}"
        f"{rng.choice(letters)}1Z{rng.choice(letters)}"
    )


# ---------------------------------------------------------------------- candidate pools


def _common_pool(costs: TypicalCosts, policy: Policy, *, allow_maternity: bool = True) -> list[Diagnosis]:
    pool = []
    for d in costs.diagnoses:
        if d.exclusion_tag or d.ped_condition:
            continue
        if d.is_maternity and not (allow_maternity and policy.maternity.covered):
            continue
        pool.append(d)
    return pool


def _candidate_policies(rules: PolicyRules, scenario: str, subtype: str) -> list[Policy]:
    pols = rules.policies
    if scenario in ("covered", "incomplete") or (scenario == "risk" and subtype != "early_admission"):
        clean = [p for p in pols if p.co_pay_percent == 0]
        return clean or pols
    if scenario == "partial":
        if subtype == "room_rent":
            return [p for p in pols if p.room_rent_limit.percent_of_si_per_day or p.room_rent_limit.max_per_day]
        if subtype == "copay":
            return [p for p in pols if p.co_pay_percent > 0]
        return [p for p in pols if p.sub_limits]
    if scenario == "rejected":
        if subtype.startswith("specific_waiting:"):
            return [p for p in pols if subtype.split(":", 1)[1] in p.specific_disease_waiting]
        if subtype == "pre_existing":
            return [p for p in pols if p.pre_existing_waiting_months > 2]
        if subtype.startswith("exclusion:"):
            return [p for p in pols if subtype.split(":", 1)[1] in p.exclusions]
        return [p for p in pols if p.initial_waiting_days > 5]
    return pols


def _choose_diagnosis(
    rng: random.Random, costs: TypicalCosts, policy: Policy, scenario: str, subtype: str
) -> Diagnosis | None:
    if scenario == "rejected":
        if subtype.startswith("specific_waiting:"):
            return costs.get(subtype.split(":", 1)[1])
        if subtype == "pre_existing":
            peds = [d for d in costs.diagnoses if d.ped_condition]
            return rng.choice(peds) if peds else None
        if subtype.startswith("exclusion:"):
            tag = subtype.split(":", 1)[1]
            tagged = [d for d in costs.diagnoses if d.exclusion_tag == tag]
            return rng.choice(tagged) if tagged else None
        # initial waiting: non-accident, plain illness
        pool = [d for d in _common_pool(costs, policy, allow_maternity=False) if not d.accident]
        return rng.choice(pool)
    if scenario == "risk" and subtype == "early_admission":
        pool = [d for d in costs.diagnoses if d.accident]
        return rng.choice(pool) if pool else None
    if scenario == "partial" and subtype == "sub_limit":
        pool = [d for d in _common_pool(costs, policy) if d.sub_limit_key in policy.sub_limits]
        return rng.choice(pool) if pool else None
    return rng.choice(_common_pool(costs, policy))


# ------------------------------------------------------------------------- core builder


def _try_build(
    rng: random.Random,
    fake: Faker,
    idx: int,
    scenario: str,
    subtype: str,
    rules: PolicyRules,
    costs: TypicalCosts,
) -> PlannedClaim | None:
    policies = _candidate_policies(rules, scenario, subtype)
    if not policies:
        return None
    policy = rng.choice(policies)
    si = rng.choice(policy.sum_insured_options)
    diag = _choose_diagnosis(rng, costs, policy, scenario, subtype)
    if diag is None:
        return None
    lo, hi = diag.cost_range

    # ---- dates
    admission = date(2025, 6, 1) + timedelta(days=rng.randint(0, 480))
    required = max(
        policy.specific_disease_waiting.get(diag.specific_wait_key or "", 0),
        policy.pre_existing_waiting_months if diag.ped_condition else 0,
        policy.maternity.waiting_months if diag.is_maternity else 0,
        2,
    )
    if scenario == "rejected" and subtype == "initial_waiting":
        start = admission - timedelta(days=rng.randint(5, 25))
    elif scenario == "rejected" and subtype.startswith("specific_waiting:"):
        wait = policy.specific_disease_waiting[diag.specific_wait_key]
        if wait < 4:
            return None
        start = shift_months(admission, -rng.randint(3, wait - 1)) - timedelta(days=rng.randint(0, 20))
    elif scenario == "rejected" and subtype == "pre_existing":
        wait = policy.pre_existing_waiting_months
        start = shift_months(admission, -rng.randint(3, wait - 1)) - timedelta(days=rng.randint(0, 20))
    elif scenario == "risk" and subtype == "early_admission":
        start = admission - timedelta(days=rng.randint(2, 9))
    else:
        start = shift_months(admission, -(required + rng.randint(3, 24))) - timedelta(days=rng.randint(0, 20))

    stay = rng.randint(*diag.stay_days)
    discharge = admission + timedelta(days=stay)

    # ---- room, ICU
    icu_days = 0
    if diag.icu_probability and stay > 1 and rng.random() < diag.icu_probability:
        icu_days = rng.randint(1, min(2, stay - 1))
    room_days = max(1, stay - icu_days)
    room_limit = policy.room_rent_limit.limit_per_day(si)
    icu_limit = policy.icu_limit.limit_per_day(si)

    if scenario == "partial" and subtype == "room_rent":
        if room_limit is None:
            return None
        room_rate = _round_up(room_limit * rng.uniform(1.3, 2.2))
    elif room_limit is None:
        room_rate = rng.choice([3000, 4500, 6000, 7500])
    else:
        floor = min(1500, room_limit)
        room_rate = rng.randrange(floor, room_limit + 1, 100) if room_limit > floor else room_limit
    icu_rate = 0
    if icu_days:
        icu_rate = (
            rng.choice([6000, 8000, 10000])
            if icu_limit is None
            else rng.randrange(min(4000, icu_limit), icu_limit + 1, 500)
        )

    # ---- cost
    if scenario == "partial" and subtype == "sub_limit":
        sub = policy.sub_limits[diag.sub_limit_key]
        if sub.amount * 1.1 >= hi:
            return None
        target = int(rng.uniform(sub.amount * 1.1, hi))
    elif scenario == "risk" and subtype == "high_bill":
        target = int(hi * rng.uniform(1.7, 2.5))
    else:
        target = int(rng.triangular(lo, hi, (lo + hi) / 2))

    lines, gst = build_bill(
        rng, diag, room_rate=room_rate, room_days=room_days, icu_rate=icu_rate, icu_days=icu_days, target=target
    )
    if scenario == "risk" and subtype == "duplicate_items":
        lines = _add_duplicates(rng, lines)

    clean = _charges(lines, include_duplicates=False)
    facts = ClaimFacts(
        sum_insured=si,
        diagnosis_key=diag.key,
        policy_start_date=start,
        admission_date=admission,
        is_accident=diag.accident,
        is_maternity=diag.is_maternity,
        ped_related=diag.ped_condition is not None,
        exclusion_tag=diag.exclusion_tag,
        specific_wait_key=diag.specific_wait_key,
        sub_limit_key=diag.sub_limit_key,
        room_rate_per_day=room_rate,
        icu_rate_per_day=icu_rate,
        icu_days=icu_days,
        charges=clean,
        room_gst=gst,
    )
    result = compute_payout(policy, facts)
    if not _scenario_ok(scenario, subtype, result, diag, policy):
        return None

    clean_total = sum(clean.values()) + gst
    billed_total = sum(ln.amount for ln in lines) + gst
    if scenario == "risk" and subtype == "high_bill":
        if billed_total <= hi * 1.4:
            return None
    elif billed_total > hi * 1.35:
        return None

    return _assemble(
        rng, fake, idx, scenario, subtype, policy, si, diag, start, admission, discharge, stay, icu_days,
        lines, gst, billed_total, clean_total, result,
    )  # fmt: skip


def _scenario_ok(scenario: str, subtype: str, result: PayoutResult, diag: Diagnosis, policy: Policy) -> bool:
    rules = {d.rule for d in result.deductions}
    if scenario == "covered":
        return result.outcome == "approve"
    if scenario == "partial":
        if result.outcome != "partial_approve":
            return False
        if subtype == "room_rent":
            return "room_rent_limit" in rules
        if subtype == "copay":
            return rules == {"co_pay_percent"}
        return f"sub_limits.{diag.sub_limit_key}" in rules
    if scenario == "rejected":
        rule = result.rejection_rule or ""
        if subtype == "initial_waiting":
            return rule == "initial_waiting_days"
        if subtype.startswith("specific_waiting:"):
            return rule == f"specific_disease_waiting.{subtype.split(':', 1)[1]}"
        if subtype == "pre_existing":
            return rule == "pre_existing_waiting_months"
        return rule == f"exclusions.{subtype.split(':', 1)[1]}"
    return result.outcome != "reject"  # risk / incomplete are otherwise-payable claims


def _assemble(
    rng: random.Random,
    fake: Faker,
    idx: int,
    scenario: str,
    subtype: str,
    policy: Policy,
    si: int,
    diag: Diagnosis,
    start: date,
    admission: date,
    discharge: date,
    stay: int,
    icu_days: int,
    lines: list[BillLine],
    gst: int,
    billed_total: int,
    clean_total: int,
    result: PayoutResult,
) -> PlannedClaim:
    claim_id = f"CLM-{idx + 1:03d}"

    # ---- people
    female = diag.gender == "female" or rng.random() < 0.45
    name = fake.name_female() if female else fake.name_male()
    age = rng.randint(*diag.age_range)
    first = name.split()[0].lower()
    patient = Patient(
        name=name,
        age=age,
        gender="Female" if female else "Male",
        phone=fake.phone_number(),
        email=f"{first}{rng.randint(10, 99)}@example.com",
        address=fake.address().replace("\n", ", "),
        uhid=f"UH{rng.randint(1000000, 9999999)}",
    )
    h_name, h_city = rng.choice(HOSPITALS)
    hospital = Hospital(
        name=h_name,
        city=h_city,
        address=f"{fake.street_address()}, {h_city}",
        phone=fake.phone_number(),
        gstin=_fake_gstin(rng),
        registration_no=f"REG/{rng.randint(1000, 9999)}/{rng.randint(2005, 2020)}",
    )

    # ---- document-level inconsistencies / omissions
    form_discharge = discharge
    summary_adm, summary_dis = admission, discharge
    risk_flags: list[str] = []
    missing: list[str] = []
    blank_fields: list[str] = []
    omit_summary = False
    policy_number: str | None = f"{policy.policy_id}/{start.year}/{rng.randint(1000000, 9999999)}"
    claimed = billed_total

    if scenario == "risk":
        flag = RISK_FLAG_BY_SUBTYPE[subtype]
        risk_flags.append(flag)
        if subtype == "date_mismatch":
            shift = timedelta(days=rng.randint(2, 5))
            summary_adm, summary_dis = admission + shift, discharge + shift
        elif subtype == "discharge_before_admission":
            form_discharge = admission - timedelta(days=rng.randint(1, 3))
            summary_dis = form_discharge
    elif scenario == "incomplete":
        if subtype == "missing_discharge_summary":
            omit_summary = True
            missing.append("discharge_summary")
        elif subtype == "blank_field":
            blank_fields.append("discharge_summary.admission_date")
            missing.append("discharge_summary.admission_date")
        elif subtype == "missing_policy_number":
            policy_number = None
            missing.append("claim_form.policy_number")
        else:
            blank_fields.append("hospital_bill.total_amount")
            missing.append("hospital_bill.total_amount")

    spec = ClaimSpec(
        claim_id=claim_id,
        scenario=ScenarioType(scenario),
        subtype=subtype,
        policy_id=policy.policy_id,
        policy_number=policy_number,
        policy_start_date=start,
        sum_insured=si,
        patient=patient,
        hospital=hospital,
        doctor=f"Dr. {fake.name_male() if rng.random() < 0.6 else fake.name_female()}",
        diagnosis_key=diag.key,
        diagnosis_name=diag.name,
        icd10=diag.icd10,
        procedure=diag.procedure,
        is_accident=diag.accident,
        ped_conditions=[f"{diag.ped_condition} (since {admission.year - rng.randint(3, 9)})"]
        if diag.ped_condition
        else [],
        admission_date=admission,
        discharge_date=form_discharge,
        summary_admission_date=summary_adm,
        summary_discharge_date=summary_dis,
        stay_days=stay,
        icu_days=icu_days,
        bill_no=f"BILL/{admission.year}/{rng.randint(10000, 99999)}",
        bill_lines=lines,
        gst_on_room=gst,
        claimed_amount=claimed,
        omit_discharge_summary=omit_summary,
        blank_fields=blank_fields,
    )

    golden = _golden(spec, result, risk_flags, missing, billed_total - clean_total)
    return PlannedClaim(spec=spec, golden=golden)


def _golden(
    spec: ClaimSpec, result: PayoutResult, risk_flags: list[str], missing: list[str], duplicate_amount: int
) -> GoldenRecord:
    deductions = list(result.deductions)
    if duplicate_amount:
        deductions.insert(
            0,
            Deduction(
                rule="duplicate_line_items",
                description="Duplicated bill line items removed from payable amount",
                amount=duplicate_amount,
            ),
        )

    common = {
        "claim_id": spec.claim_id,
        "scenario_type": spec.scenario,
        "subtype": spec.subtype,
        "policy_id": spec.policy_id,
        "diagnosis": spec.diagnosis_name,
        "icd10": spec.icd10,
        "claimed_amount": spec.claimed_amount,
    }
    if spec.scenario == ScenarioType.RISK:
        return GoldenRecord(
            **common,
            expected_decision=Decision.FLAG_FOR_REVIEW,
            expected_payable_amount=result.payable_amount,
            payable_basis="provisional",
            deductions=deductions,
            reasons=[RISK_REASON[f] for f in risk_flags],
            expected_policy_rule=f"risk.{risk_flags[0]}",
            expected_risk_flags=risk_flags,
        )
    if spec.scenario == ScenarioType.INCOMPLETE:
        return GoldenRecord(
            **common,
            expected_decision=Decision.REQUEST_DOCUMENTS,
            expected_payable_amount=None,
            payable_basis="not_determinable",
            reasons=[f"Missing or unreadable: {m}" for m in missing],
            expected_policy_rule="document_completeness",
            expected_missing_documents=missing,
        )
    decision = {
        "approve": Decision.APPROVE,
        "partial_approve": Decision.PARTIAL_APPROVE,
        "reject": Decision.REJECT,
    }[result.outcome]
    return GoldenRecord(
        **common,
        expected_decision=decision,
        expected_payable_amount=result.payable_amount,
        payable_basis="final",
        deductions=deductions,
        reasons=result.reasons,
        expected_policy_rule=result.primary_rule,
    )


# --------------------------------------------------------------------------- public API


def plan_claims(
    count: int,
    seed: int,
    rules: PolicyRules,
    costs: TypicalCosts,
    mix: dict[str, int] | None = None,
) -> list[PlannedClaim]:
    resolved = resolve_mix(count, mix)
    cycles = {
        "covered": ["base"],
        "partial": PARTIAL_CYCLE,
        "rejected": REJECTED_CYCLE,
        "risk": RISK_CYCLE,
        "incomplete": INCOMPLETE_CYCLE,
    }
    slots: list[tuple[str, str]] = []
    for scenario, n in resolved.items():
        cycle = cycles[scenario]
        slots += [(scenario, cycle[i % len(cycle)]) for i in range(n)]
    random.Random(seed).shuffle(slots)

    planned: list[PlannedClaim] = []
    for idx, (scenario, subtype) in enumerate(slots):
        fake = Faker("en_IN")
        fake.seed_instance(f"{seed}:{idx}")
        for attempt in range(MAX_ATTEMPTS):
            rng = random.Random(f"{seed}:{idx}:{attempt}")
            claim = _try_build(rng, fake, idx, scenario, subtype, rules, costs)
            if claim is not None:
                planned.append(claim)
                break
        else:
            raise PlanningError(
                f"Could not build a '{scenario}/{subtype}' claim in {MAX_ATTEMPTS} attempts; "
                "check policy_rules.yaml / typical_costs.yaml for a compatible policy."
            )
    logger.info("Planned %d claims: %s", len(planned), resolved)
    return planned
