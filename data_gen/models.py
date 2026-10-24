"""Pydantic models for generated claim data and golden records."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class ScenarioType(StrEnum):
    COVERED = "covered"
    PARTIAL = "partial"
    REJECTED = "rejected"
    RISK = "risk"
    INCOMPLETE = "incomplete"


class Decision(StrEnum):
    APPROVE = "approve"
    PARTIAL_APPROVE = "partial_approve"
    REJECT = "reject"
    FLAG_FOR_REVIEW = "flag_for_review"
    REQUEST_DOCUMENTS = "request_documents"


class Deduction(BaseModel):
    rule: str
    description: str
    amount: int


class BillLine(BaseModel):
    category: str
    description: str
    qty: int
    rate: int
    amount: int
    is_duplicate: bool = False


class Patient(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    email: str
    address: str
    uhid: str


class Hospital(BaseModel):
    name: str
    city: str
    address: str
    phone: str
    gstin: str
    registration_no: str


class ClaimSpec(BaseModel):
    """Everything needed to render one claim's documents."""

    claim_id: str
    scenario: ScenarioType
    subtype: str
    policy_id: str
    policy_number: str | None
    policy_start_date: date
    sum_insured: int
    patient: Patient
    hospital: Hospital
    doctor: str
    diagnosis_key: str
    diagnosis_name: str
    icd10: str
    procedure: str | None
    is_accident: bool
    ped_conditions: list[str] = Field(default_factory=list)
    admission_date: date
    discharge_date: date
    summary_admission_date: date
    summary_discharge_date: date
    stay_days: int
    icu_days: int
    bill_no: str
    bill_lines: list[BillLine]
    gst_on_room: int
    claimed_amount: int
    omit_discharge_summary: bool = False
    blank_fields: list[str] = Field(default_factory=list)

    @property
    def bill_subtotal(self) -> int:
        return sum(line.amount for line in self.bill_lines)

    @property
    def bill_total(self) -> int:
        return self.bill_subtotal + self.gst_on_room


class ClaimantInfo(BaseModel):
    name: str
    age: int
    gender: str
    phone: str
    email: str
    address: str
    relationship_to_insured: str = "self"


class ClaimForm(BaseModel):
    """Contents of claim_form.json."""

    claim_id: str
    claimant: ClaimantInfo
    policy_id: str
    policy_number: str | None
    policy_start_date: date
    sum_insured: int
    hospital_name: str
    hospital_city: str
    diagnosis: str
    admission_date: date
    discharge_date: date
    pre_existing_conditions: list[str]
    is_accident_case: bool
    claimed_amount: int

    @classmethod
    def from_spec(cls, spec: ClaimSpec) -> ClaimForm:
        patient = spec.patient
        return cls(
            claim_id=spec.claim_id,
            claimant=ClaimantInfo(
                name=patient.name,
                age=patient.age,
                gender=patient.gender,
                phone=patient.phone,
                email=patient.email,
                address=patient.address,
            ),
            policy_id=spec.policy_id,
            policy_number=spec.policy_number,
            policy_start_date=spec.policy_start_date,
            sum_insured=spec.sum_insured,
            hospital_name=spec.hospital.name,
            hospital_city=spec.hospital.city,
            diagnosis=spec.diagnosis_name,
            admission_date=spec.admission_date,
            discharge_date=spec.discharge_date,
            pre_existing_conditions=spec.ped_conditions,
            is_accident_case=spec.is_accident,
            claimed_amount=spec.claimed_amount,
        )


class GoldenRecord(BaseModel):
    claim_id: str
    scenario_type: ScenarioType
    subtype: str
    policy_id: str
    diagnosis: str
    icd10: str
    expected_decision: Decision
    claimed_amount: int
    # null when the amount cannot be determined (incomplete claims)
    expected_payable_amount: int | None
    # final = authoritative; provisional = payable if all risk flags are cleared;
    # not_determinable = documents missing
    payable_basis: str
    deductions: list[Deduction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    expected_policy_rule: str
    expected_risk_flags: list[str] = Field(default_factory=list)
    expected_missing_documents: list[str] = Field(default_factory=list)
    has_scanned_copy: bool = False


class Narrative(BaseModel):
    history: str
    course_in_hospital: str
    discharge_advice: list[str]


class PlannedClaim(BaseModel):
    spec: ClaimSpec
    golden: GoldenRecord
