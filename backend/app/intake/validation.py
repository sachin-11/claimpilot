"""Extraction validation and grounding (no LLM, no I/O).

validate_extraction(): rules a correct extraction must satisfy; its messages are what the LLM is
sent on a retry. verify_grounding(): checks that extracted values really occur in the source text,
which is how confidence is decided (the model's own confidence is only a starting point).
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.intake.config import IntakeSettings
from app.intake.schemas import (
    ClaimExtraction,
    Confidence,
    DocumentText,
    FieldSource,
)

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y", "%d-%b-%Y")
_MONTH_FIRST = ("%m/%d/%Y",)  # only tried when day-first is impossible (day > 12 cannot be a month)


def parse_date(value: str | None) -> date | None:
    """A date from the formats Indian hospital documents use. Raises ValueError if unparseable."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    for fmt in _MONTH_FIRST:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{value!r} is not a recognised date (use YYYY-MM-DD)")


def to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value).replace(",", "")).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{value!r} is not a valid amount") from exc


def amount_tolerance(amount: Decimal, settings: IntakeSettings) -> Decimal:
    """The larger of pct% of the amount and the absolute tolerance (default 1% or Rs 10)."""
    return max(
        abs(amount) * Decimal(str(settings.amount_tolerance_pct)),
        Decimal(str(settings.amount_tolerance_abs)),
    )


_GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$")
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")


def plausible_policy_number(value: str | None) -> str | None:
    """The value if it can be a policy number; None for things that are not (a GSTIN, a PAN, a
    bare word). Models sometimes copy the hospital's GSTIN from the letterhead into this field."""
    if value is None:
        return None
    cleaned = value.strip()
    compact = re.sub(r"\s", "", cleaned).upper()
    if len(compact) < 6 or not any(c.isdigit() for c in compact):
        return None
    if _GSTIN_RE.match(compact) or _PAN_RE.match(compact):
        return None
    return cleaned


def _normalise_policy(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def validate_extraction(
    extraction: ClaimExtraction,
    settings: IntakeSettings,
    expected_policy_number: str | None = None,
) -> list[str]:
    """Human-readable problems; empty list = valid."""
    errors: list[str] = []
    e = extraction

    if e.admission_date and e.discharge_date and e.discharge_date < e.admission_date:
        errors.append(
            f"discharge_date {e.discharge_date} is before admission_date {e.admission_date}: "
            "check both dates against the documents (swap only if a document really says so)"
        )

    if e.line_items and e.bill_subtotal is not None:
        total = sum((item.amount for item in e.line_items), Decimal(0))
        if abs(total - e.bill_subtotal) > amount_tolerance(e.bill_subtotal, settings):
            errors.append(
                f"the line item amounts add up to {total} but bill_subtotal is {e.bill_subtotal}: "
                "a line item is missing, duplicated or misread, or the subtotal is wrong"
            )

    if e.bill_subtotal is not None and e.taxes is not None and e.bill_total is not None:
        expected_total = e.bill_subtotal + e.taxes
        if abs(expected_total - e.bill_total) > amount_tolerance(e.bill_total, settings):
            errors.append(
                f"bill_subtotal {e.bill_subtotal} + taxes {e.taxes} = {expected_total}, "
                f"but bill_total is {e.bill_total}"
            )

    for name in ("bill_subtotal", "bill_total", "claimed_amount", "room_rent_per_day"):
        value = getattr(e, name)
        if value is not None and value <= 0:
            errors.append(f"{name} must be greater than 0 (got {value})")

    if expected_policy_number and e.policy_number:
        if _normalise_policy(expected_policy_number) != _normalise_policy(e.policy_number):
            errors.append(
                f"policy_number {e.policy_number!r} does not match the policy on the claim "
                f"({expected_policy_number!r})"
            )
    return errors


# ------------------------------------------------------------------------- grounding

GROUNDED_FIELDS = (
    "policy_number",
    "admission_date",
    "discharge_date",
    "icd10_code",
    "bill_total",
    "bill_subtotal",
    "taxes",
    "claimed_amount",
    "patient_name",
    "hospital_name",
    "diagnosis_text",
)


def date_variants(value: date) -> list[str]:
    return [
        value.isoformat(),
        value.strftime("%d-%m-%Y"),
        value.strftime("%d/%m/%Y"),
        value.strftime("%d.%m.%Y"),
        value.strftime("%d %b %Y"),
        value.strftime("%d %B %Y"),
        value.strftime("%d-%b-%Y"),
        f"{value.day}-{value.month}-{value.year}",
    ]


def _digits_only(text: str) -> str:
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text)


def value_in_text(field_name: str, value: Any, text: str) -> bool:
    """Does `value` (as extracted) occur in `text` in any common printed format?"""
    lowered = text.lower()
    if isinstance(value, date):
        return any(v and v.lower() in lowered for v in date_variants(value))
    if isinstance(value, Decimal):
        plain = _digits_only(text)
        candidates = {format(value, "f"), format(value.quantize(Decimal(1)), "f")}
        candidates |= {c.rstrip("0").rstrip(".") for c in list(candidates) if "." in c}
        return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", plain) for c in candidates if c)
    needle = str(value).strip().lower()
    if not needle:
        return False
    if field_name == "policy_number":
        return _normalise_policy(str(value)) in _normalise_policy(text)
    if field_name == "diagnosis_text":
        words = [w for w in re.findall(r"[a-z0-9]+", needle) if len(w) > 2]
        return bool(words) and all(w in lowered for w in words)
    return " ".join(needle.split()) in " ".join(lowered.split())


def locate(
    field_name: str, value: Any, documents: list[DocumentText], hint: FieldSource | None
) -> FieldSource | None:
    """Where the value really is: the model's cited page first, then every page."""
    pages = [(d, p) for d in documents for p in d.pages]
    if hint is not None:
        pages.sort(
            key=lambda dp: not (dp[0].doc_type.value == hint.doc_type and dp[1].page == hint.page)
        )
    for document, page in pages:
        if value_in_text(field_name, value, page.text):
            return FieldSource(doc_type=document.doc_type.value, page=page.page, method=page.method)
    return None


def apply_grounding(
    extraction: ClaimExtraction,
    documents: list[DocumentText],
    settings: IntakeSettings,
    model_confidence: dict[str, Confidence],
    model_sources: dict[str, FieldSource],
) -> None:
    """Fill field_sources / field_confidence in place, from what the text really contains.

    high   : value found in a text-layer / structured source (and the model was not unsure)
    medium : value found, but only via OCR; or the model said medium and the source is OCR
    low    : value NOT found in any document text, or found only on an OCR page whose mean word
             confidence is below OCR_LOW_CONF
    """
    pages_by_key = {(d.doc_type.value, p.page, p.method): p for d in documents for p in d.pages}
    for name in GROUNDED_FIELDS:
        value = getattr(extraction, name)
        if value is None:
            continue
        source = locate(name, value, documents, model_sources.get(name))
        if source is None:
            extraction.field_confidence[name] = "low"
            if name in model_sources:
                extraction.field_sources[name] = model_sources[name]
            continue
        extraction.field_sources[name] = source
        page = pages_by_key.get((source.doc_type, source.page, source.method))
        asserted = model_confidence.get(name, "medium")
        if source.method == "ocr":
            low_ocr = page is not None and (page.ocr_confidence or 0.0) < settings.ocr_low_conf
            extraction.field_confidence[name] = "low" if low_ocr else "medium"
        else:
            extraction.field_confidence[name] = "medium" if asserted == "low" else "high"

    if extraction.length_of_stay_days is not None:
        extraction.field_sources["length_of_stay_days"] = FieldSource(
            doc_type="computed", method="computed"
        )
        dates = [extraction.field_confidence.get(n) for n in ("admission_date", "discharge_date")]
        extraction.field_confidence["length_of_stay_days"] = "low" if "low" in dates else "high"
