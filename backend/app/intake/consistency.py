"""Cross-document consistency and completeness checks.

Facts only (what differs, what is missing, with a severity). Whether any of it matters for the
claim decision is Module 8 (risk / decision), not here.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher

from app.intake.config import IntakeSettings
from app.intake.schemas import ClaimExtraction, ConsistencyIssue, DocumentObservation
from app.intake.validation import amount_tolerance
from app.models.enums import DocType

REQUIRED_DOCUMENTS = (DocType.HOSPITAL_BILL, DocType.DISCHARGE_SUMMARY)
_HONORIFICS = re.compile(r"\b(mr|mrs|ms|miss|dr|shri|smt|master|baby|late)\b\.?", re.I)


def missing_documents(present: Iterable[DocType]) -> list[str]:
    have = set(present)
    return [doc.value for doc in REQUIRED_DOCUMENTS if doc not in have]


def normalise_name(name: str) -> str:
    cleaned = _HONORIFICS.sub(" ", name.lower())
    tokens = re.findall(r"[a-z]+", cleaned)
    return " ".join(sorted(tokens))  # order-insensitive: "Bala Suhani" == "Suhani Bala"


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_name(a), normalise_name(b)).ratio()


def _line_signature(description: str, amount: Decimal, quantity: Decimal | None) -> tuple:
    return (" ".join(description.lower().split()), amount, quantity)


def duplicate_line_items(extraction: ClaimExtraction) -> list[str]:
    seen: dict[tuple, int] = defaultdict(int)
    names: dict[tuple, str] = {}
    for item in extraction.line_items:
        key = _line_signature(item.description, item.amount, item.quantity)
        seen[key] += 1
        names[key] = item.description
    return [f"{names[k]} (x{count})" for k, count in seen.items() if count > 1]


def _distinct_dates(
    observations: list[DocumentObservation], attribute: str
) -> dict[date, list[str]]:
    by_value: dict[date, list[str]] = defaultdict(list)
    for obs in observations:
        value = getattr(obs, attribute)
        if value is not None:
            by_value[value].append(obs.doc_type)
    return by_value


def check_consistency(
    extraction: ClaimExtraction,
    present_documents: Iterable[DocType],
    settings: IntakeSettings,
    *,
    superseded: list[str] | None = None,
) -> tuple[list[ConsistencyIssue], list[str]]:
    """(consistency_issues, missing_documents)."""
    issues: list[ConsistencyIssue] = []
    missing = missing_documents(present_documents)
    for doc in missing:
        issues.append(
            ConsistencyIssue(
                type="missing_document",
                severity="critical",
                detail=f"required document not provided: {doc}",
                fields=[doc],
            )
        )

    observations = extraction.observations
    for attribute, label in (("admission_date", "admission"), ("discharge_date", "discharge")):
        by_value = _distinct_dates(observations, attribute)
        if len(by_value) > 1:
            parts = "; ".join(
                f"{', '.join(docs)}: {value.isoformat()}"
                for value, docs in sorted(by_value.items())
            )
            issues.append(
                ConsistencyIssue(
                    type="date_mismatch",
                    severity="warning",
                    detail=f"{label} date differs between documents ({parts})",
                    fields=[attribute],
                )
            )

    if (
        extraction.admission_date
        and extraction.discharge_date
        and extraction.discharge_date < extraction.admission_date
    ):
        issues.append(
            ConsistencyIssue(
                type="discharge_before_admission",
                severity="critical",
                detail=f"discharge {extraction.discharge_date} is before admission "
                f"{extraction.admission_date}",
                fields=["admission_date", "discharge_date"],
            )
        )
    for obs in observations:  # a single document can contradict itself
        if obs.admission_date and obs.discharge_date and obs.discharge_date < obs.admission_date:
            issues.append(
                ConsistencyIssue(
                    type="discharge_before_admission",
                    severity="critical",
                    detail=f"{obs.doc_type}: discharge {obs.discharge_date} is before admission "
                    f"{obs.admission_date}",
                    fields=["admission_date", "discharge_date"],
                )
            )
            break

    named = [(o.doc_type, o.patient_name) for o in observations if o.patient_name]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            ratio = name_similarity(named[i][1], named[j][1])
            if ratio < settings.name_match_threshold:
                issues.append(
                    ConsistencyIssue(
                        type="patient_name_mismatch",
                        severity="warning",
                        detail=f"patient name differs: {named[i][0]} says {named[i][1]!r}, "
                        f"{named[j][0]} says {named[j][1]!r} (similarity {ratio:.2f})",
                        fields=["patient_name"],
                    )
                )

    for attribute, label in (("admission_date", "admission"), ("discharge_date", "discharge")):
        if not any(getattr(o, attribute) for o in observations):
            continue  # no document shows it: reported below as a missing field
        for obs in observations:
            if (
                obs.doc_type in {d.value for d in REQUIRED_DOCUMENTS}
                and getattr(obs, attribute) is None
            ):
                issues.append(
                    ConsistencyIssue(
                        type="missing_field",
                        severity="warning",
                        detail=f"{obs.doc_type} does not show the {label} date "
                        "(blank or unreadable)",
                        fields=[attribute],
                    )
                )

    policies = {o.policy_number.strip() for o in observations if o.policy_number}
    if len(policies) > 1:
        issues.append(
            ConsistencyIssue(
                type="policy_number_mismatch",
                severity="warning",
                detail=f"documents show different policy numbers: {sorted(policies)}",
                fields=["policy_number"],
            )
        )
    if not extraction.policy_number:
        issues.append(
            ConsistencyIssue(
                type="missing_field",
                severity="warning",
                detail="policy number was not found in any document",
                fields=["policy_number"],
            )
        )

    for name, label in (("admission_date", "admission date"), ("discharge_date", "discharge date")):
        if getattr(extraction, name) is None and not any(
            i.type == "missing_field" and name in i.fields for i in issues
        ):
            issues.append(
                ConsistencyIssue(
                    type="missing_field",
                    severity="warning",
                    detail=f"{label} could not be read from any document",
                    fields=[name],
                )
            )

    if extraction.bill_total is None:
        issues.append(
            ConsistencyIssue(
                type="missing_field",
                severity="warning",
                detail="bill total could not be read (unreadable or absent)",
                fields=["bill_total"],
            )
        )
    elif extraction.claimed_amount is not None:
        gap = extraction.claimed_amount - extraction.bill_total
        if abs(gap) > amount_tolerance(extraction.bill_total, settings):
            unsure = extraction.field_confidence.get("bill_total") == "low"
            severity = "info" if gap <= 0 else ("warning" if unsure else "critical")
            note = (
                " (the bill total was read with LOW confidence, e.g. OCR: check it)"
                if unsure
                else ""
            )
            issues.append(
                ConsistencyIssue(
                    type="claimed_amount_differs_from_bill",
                    severity=severity,
                    detail=f"claimed amount {extraction.claimed_amount} vs bill total "
                    f"{extraction.bill_total} (difference {gap}){note}",
                    fields=["claimed_amount", "bill_total"],
                )
            )

    duplicates = duplicate_line_items(extraction)
    if duplicates:
        issues.append(
            ConsistencyIssue(
                type="duplicate_line_items",
                severity="warning",
                detail="identical line items appear more than once: " + "; ".join(duplicates),
                fields=["line_items"],
            )
        )

    for note in superseded or []:
        issues.append(ConsistencyIssue(type="document_superseded", severity="info", detail=note))

    # fields the documents disagree on cannot keep a high confidence
    disputed = {
        field
        for issue in issues
        if issue.type in ("date_mismatch", "patient_name_mismatch", "policy_number_mismatch")
        for field in issue.fields
    }
    for field in disputed:
        if field in extraction.field_confidence or getattr(extraction, field, None) is not None:
            extraction.field_confidence[field] = "low"

    low = sorted(n for n, level in extraction.field_confidence.items() if level == "low")
    if low:
        issues.append(
            ConsistencyIssue(
                type="low_confidence_fields",
                severity="info",
                detail="fields with low confidence (not found in the text or poor OCR): "
                + ", ".join(low),
                fields=low,
            )
        )
    return issues, missing
