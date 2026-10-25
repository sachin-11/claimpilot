"""Claim extraction + customer history + policy rules -> calculator facts (pure)."""

import re
from datetime import date
from decimal import Decimal
from typing import Any

from app.agents.calculator import ClaimFacts, PolicyRules
from app.agents.catalog import Diagnosis, resolve_diagnosis

# Bill line categories (Module 6) -> rule categories used by policy `proportionate_categories`.
CATEGORY_MAP = {
    "room": "room_rent",
    "icu": "icu",
    "surgeon": "professional_fees",
    "ot": "ot_charges",
    "medicines": "medicines",
    "consumables": "consumables",
    "diagnostics": "diagnostics",
    "other": "other",
}

# "surgeon" in the extraction also catches items like a pre-anaesthetic check-up; only real
# professional fees are subject to the room-rent proportionate deduction.
_PROFESSIONAL = re.compile(
    r"\b(surgeon|anaesthetist|anesthetist|assistant|consultant|physician|doctor|specialist|visit|officer|rmo)\b",
    re.IGNORECASE,
)
# A misleading diagnosis must not hide an excluded procedure: when the catalogue gives no exclusion
# category, the procedure and the bill line items are screened for these words as well.
_EXCLUSION_WORDS: dict[str, re.Pattern[str]] = {
    "cosmetic_surgery": re.compile(
        r"\b(cosmetic|aesthetic|rhinoplasty|liposuction|botox|facelift|tummy tuck)\b", re.IGNORECASE
    ),
    "obesity_treatment": re.compile(
        r"\b(bariatric|obesity|gastric bypass|sleeve gastrectomy|gastric band)\b", re.IGNORECASE
    ),
    "dental_unless_accident": re.compile(
        r"\b(dental|teeth|tooth|denture|orthodontic)\b", re.IGNORECASE
    ),
}
_MEDICAL_OFFICER = re.compile(r"\b(medical officer|rmo)\b", re.IGNORECASE)
_BILATERAL = re.compile(r"bilateral|both eyes|\bBE\b", re.IGNORECASE)


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _dec(value: Any) -> Decimal | None:
    return None if value in (None, "") else Decimal(str(value))


def _date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)) if value else None


def is_ped_related(conditions: list[str], diagnosis: Diagnosis | None) -> bool:
    """True when a declared pre-existing condition is the one this diagnosis is related to."""
    if diagnosis is None or not diagnosis.ped_condition:
        return False
    target = _norm(diagnosis.ped_condition)
    return any(target in _norm(c) or _norm(c) in target for c in conditions if _norm(c))


def rule_category(item: dict[str, Any]) -> str:
    """Rule category of a bill line. The extraction label is corrected both ways: a "surgeon" item
    that is no professional fee is `other`, and a medical-officer charge labelled `other` is one."""
    category = CATEGORY_MAP.get(str(item.get("category")), "other")
    description = str(item.get("description") or "")
    if category == "professional_fees" and not _PROFESSIONAL.search(description):
        return "other"
    if category == "other" and _MEDICAL_OFFICER.search(description):
        return "professional_fees"
    return category


def screen_exclusion(extraction: dict[str, Any]) -> str | None:
    """Excluded category named by the procedure or a bill line item (None when nothing matches)."""
    text = " ".join(
        [
            str(extraction.get("procedure") or ""),
            *(str(i.get("description") or "") for i in extraction.get("line_items") or []),
        ]
    )
    return next((tag for tag, words in _EXCLUSION_WORDS.items() if words.search(text)), None)


def build_facts(
    extraction: dict[str, Any],
    claim_facts: dict[str, Any],
    history: dict[str, Any],
    rules: PolicyRules,
) -> tuple[ClaimFacts | None, Diagnosis | None]:
    """Calculator facts, or (None, diagnosis) when the sum insured is unknown."""
    diagnosis = resolve_diagnosis(extraction.get("icd10_code"), extraction.get("diagnosis_text"))
    sum_insured = _dec(history.get("sum_insured"))
    if sum_insured is None:
        return None, diagnosis

    charges: dict[str, Decimal] = {}
    duplicates = Decimal(0)
    seen: set[tuple[str, str, str]] = set()
    for item in extraction.get("line_items") or []:
        signature = (_norm(item.get("description")), str(item["amount"]), str(item.get("quantity")))
        if signature in seen:  # the same line billed twice: the repeat is not payable
            duplicates += Decimal(str(item["amount"]))
            continue
        seen.add(signature)
        category = rule_category(item)
        charges[category] = charges.get(category, Decimal(0)) + Decimal(str(item["amount"]))

    free_text = f"{extraction.get('procedure') or ''} {extraction.get('diagnosis_text') or ''}"
    uses_eyes = diagnosis is not None and diagnosis.sub_limit_key == "cataract"
    claimed = _dec(extraction.get("claimed_amount")) or _dec(extraction.get("bill_total"))

    facts = ClaimFacts(
        sum_insured=sum_insured,
        remaining_sum_insured=_dec(history.get("remaining_sum_insured")),
        policy_start_date=_date(history.get("policy_start_date")),
        admission_date=_date(extraction.get("admission_date")),
        is_accident=bool(claim_facts.get("is_accident")) or bool(diagnosis and diagnosis.accident),
        is_maternity=bool(diagnosis and diagnosis.is_maternity),
        ped_related=is_ped_related(claim_facts.get("pre_existing_conditions") or [], diagnosis),
        exclusion_tag=(diagnosis.exclusion_tag if diagnosis else None)
        or screen_exclusion(extraction),
        specific_wait_key=diagnosis.specific_wait_key if diagnosis else None,
        sub_limit_key=diagnosis.sub_limit_key if diagnosis else None,
        units=2 if uses_eyes and _BILATERAL.search(free_text) else 1,
        room_rate_per_day=_dec(extraction.get("room_rent_per_day")) or Decimal(0),
        icu_rate_per_day=_icu_rate(extraction, charges),
        icu_days=int(extraction.get("icu_days") or 0),
        charges=charges,
        room_tax=_dec(extraction.get("taxes")) or Decimal(0),
        duplicate_amount=duplicates,
        claimed_amount=claimed,
    )
    _ = rules  # reserved: rules decide limits inside the calculator, not here
    return facts, diagnosis


def _icu_rate(extraction: dict[str, Any], charges: dict[str, Decimal]) -> Decimal:
    """ICU rate per day: from the ICU line item (quantity x unit price) or charge / days."""
    days = int(extraction.get("icu_days") or 0)
    if not days:
        return Decimal(0)
    for item in extraction.get("line_items") or []:
        if str(item.get("category")) == "icu" and item.get("unit_price"):
            return Decimal(str(item["unit_price"]))
    return charges.get("icu", Decimal(0)) / days
