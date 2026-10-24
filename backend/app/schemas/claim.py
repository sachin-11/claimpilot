from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import (
    ApprovalAction,
    ClaimSource,
    ClaimStatus,
    DecisionType,
    DocType,
)
from app.schemas.base import ReadModel


class ClaimDocumentCreate(BaseModel):
    doc_type: DocType
    file_path: str
    mime_type: str
    is_scanned: bool = False
    page_count: int | None = None
    sha256: str


class ClaimDocumentRead(ClaimDocumentCreate, ReadModel):
    claim_id: UUID


class ClaimCreate(BaseModel):
    claim_number: str
    customer_policy_id: UUID | None = None
    hospital_name: str | None = None
    diagnosis_text: str | None = None
    icd10_code: str | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    claimed_amount: Decimal
    source: ClaimSource = ClaimSource.UPLOAD


class ClaimRead(ReadModel):
    claim_number: str
    customer_policy_id: UUID | None
    hospital_name: str | None
    diagnosis_text: str | None
    icd10_code: str | None
    admission_date: date | None
    discharge_date: date | None
    claimed_amount: Decimal
    approved_amount: Decimal | None
    status: ClaimStatus
    source: ClaimSource


class ClaimWithDocuments(ClaimRead):
    documents: list[ClaimDocumentRead] = Field(default_factory=list)


class ClaimExtractionCreate(BaseModel):
    claim_id: UUID
    extracted_data: dict[str, Any]
    model_used: str | None = None
    confidence: float | None = None
    validation_errors: list[Any] = Field(default_factory=list)
    version: int = 1


class ClaimExtractionRead(ClaimExtractionCreate, ReadModel):
    pass


class CitedClause(BaseModel):
    chunk_id: UUID
    section_path: str | None = None
    page: int | None = None


class ClaimDecisionCreate(BaseModel):
    claim_id: UUID
    decision: DecisionType
    payable_amount: Decimal | None = None
    deductions: list[dict[str, Any]] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cited_clauses: list[CitedClause] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    agent_run_id: str | None = None
    model_used: str | None = None


class ClaimDecisionRead(ReadModel):
    claim_id: UUID
    decision: DecisionType
    payable_amount: Decimal | None
    deductions: list[dict[str, Any]]
    reasons: list[Any]
    cited_clauses: list[dict[str, Any]]
    risk_flags: list[Any]
    agent_run_id: str | None
    model_used: str | None


class ClaimApprovalCreate(BaseModel):
    claim_id: UUID
    decision_id: UUID
    reviewer_id: UUID
    action: ApprovalAction
    final_amount: Decimal | None = None
    comment: str | None = None


class ClaimApprovalRead(ClaimApprovalCreate, ReadModel):
    pass
