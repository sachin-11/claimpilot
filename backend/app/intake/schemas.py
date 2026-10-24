"""Pydantic models for claim intake.

Two layers on purpose: `LLMExtraction` is what the model is asked to return (lenient: strings
instead of dates, so one odd format does not fail the whole call); `ClaimExtraction` is the strict
result that is validated and stored.
"""

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.models.enums import DocType

ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$")

Confidence = Literal["low", "medium", "high"]


class LineCategory(StrEnum):
    ROOM = "room"
    ICU = "icu"
    SURGEON = "surgeon"
    OT = "ot"
    MEDICINES = "medicines"
    CONSUMABLES = "consumables"
    DIAGNOSTICS = "diagnostics"
    OTHER = "other"


class PageText(BaseModel):
    page: int  # 1-based
    text: str
    method: Literal["text_layer", "ocr", "structured_json"]
    ocr_confidence: float | None = None  # mean word confidence 0-100 (OCR pages only)


class DocumentText(BaseModel):
    """All text of one stored document."""

    document_id: str
    doc_type: DocType
    file_name: str
    is_scanned: bool
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


class FieldSource(BaseModel):
    doc_type: str
    page: int | None = None
    method: Literal["text_layer", "ocr", "structured_json", "computed", "unknown"] = "unknown"


class LineItem(BaseModel):
    description: str
    category: LineCategory = LineCategory.OTHER
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("line item amounts must be > 0")
        return value


class DocumentObservation(BaseModel):
    """What ONE document says, exactly as printed. Cross-document checks compare these."""

    doc_type: str
    patient_name: str | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    total_amount: Decimal | None = None
    policy_number: str | None = None


class ClaimExtraction(BaseModel):
    patient_name: str | None = None
    patient_age: int | None = Field(default=None, ge=0, le=130)
    gender: str | None = None
    policy_number: str | None = None
    hospital_name: str | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    length_of_stay_days: int | None = None  # computed: discharge - admission (nights)
    diagnosis_text: str | None = None
    icd10_code: str | None = None
    procedure: str | None = None
    room_type: str | None = None
    room_rent_per_day: Decimal | None = None
    icu_days: int | None = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    bill_subtotal: Decimal | None = None
    taxes: Decimal | None = None
    bill_total: Decimal | None = None
    claimed_amount: Decimal | None = None
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    field_confidence: dict[str, Confidence] = Field(default_factory=dict)
    observations: list[DocumentObservation] = Field(default_factory=list)

    @field_validator("icd10_code")
    @classmethod
    def icd10_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().upper()
        if not ICD10_RE.match(value):
            raise ValueError(f"{value!r} is not a valid ICD-10 code")
        return value

    @field_validator("bill_subtotal", "taxes", "bill_total", "claimed_amount", "room_rent_per_day")
    @classmethod
    def money_not_negative(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and value < 0:
            raise ValueError("amounts must not be negative")
        return value


class ConsistencyIssue(BaseModel):
    type: str
    severity: Literal["info", "warning", "critical"]
    detail: str
    fields: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------- LLM-facing schema


class LLMLineItem(BaseModel):
    description: str
    # required (no default): with a default "other" the model simply skipped the field
    category: Literal[
        "room", "icu", "surgeon", "ot", "medicines", "consumables", "diagnostics", "other"
    ] = Field(
        description="room, icu, surgeon (doctor/anaesthetist fees), ot (theatre), medicines, "
        "consumables, diagnostics (lab/imaging) or other"
    )
    quantity: float | None = None
    unit_price: float | None = None
    amount: float


class LLMObservation(BaseModel):
    doc_type: str = Field(description="hospital_bill, discharge_summary or claim_form")
    patient_name: str | None = None
    admission_date: str | None = Field(default=None, description="YYYY-MM-DD as printed")
    discharge_date: str | None = Field(default=None, description="YYYY-MM-DD as printed")
    total_amount: float | None = None
    policy_number: str | None = None


class LLMFieldSource(BaseModel):
    field: str
    doc_type: str
    page: int | None = None
    confidence: str = "medium"


class LLMExtraction(BaseModel):
    """Structured output requested from the model. Dates are ISO strings; null when absent."""

    patient_name: str | None = None
    patient_age: int | None = None
    gender: str | None = None
    policy_number: str | None = None
    hospital_name: str | None = None
    admission_date: str | None = Field(default=None, description="YYYY-MM-DD")
    discharge_date: str | None = Field(default=None, description="YYYY-MM-DD")
    diagnosis_text: str | None = None
    icd10_code: str | None = None
    procedure: str | None = None
    room_type: str | None = None
    room_rent_per_day: float | None = None
    icu_days: int | None = None
    line_items: list[LLMLineItem] = Field(default_factory=list)
    bill_subtotal: float | None = None
    taxes: float | None = None
    bill_total: float | None = None
    claimed_amount: float | None = None
    field_sources: list[LLMFieldSource] = Field(default_factory=list)
    observations: list[LLMObservation] = Field(default_factory=list)
