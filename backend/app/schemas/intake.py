from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.intake.schemas import ConsistencyIssue


class UploadedDocument(BaseModel):
    id: UUID
    file_name: str
    doc_type: str
    sha256: str
    page_count: int | None
    is_scanned: bool


class ClaimUploadResponse(BaseModel):
    id: UUID
    claim_number: str
    status: str
    documents: list[UploadedDocument]
    duplicates_ignored: int = 0


class ExtractStarted(BaseModel):
    claim_id: UUID
    status: str  # "processing"
    prefer: str


class ExtractionResponse(BaseModel):
    claim_id: UUID
    claim_number: str
    claim_status: str
    version: int
    model_used: str | None
    confidence: float | None
    validation_errors: list[str]
    missing_documents: list[str]
    consistency_issues: list[ConsistencyIssue]
    extraction: dict[str, Any]  # the full stored extracted_data (fields, sources, confidence, ...)
    created_at: datetime
