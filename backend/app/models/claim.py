import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.customer import CustomerPolicy
from app.models.enums import (
    ApprovalAction,
    ClaimSource,
    ClaimStatus,
    DecisionType,
    DocType,
)
from app.models.types import pg_enum

_EMPTY_LIST = text("'[]'::jsonb")


class Claim(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index("ix_claims_status", "status"),
        Index("ix_claims_customer_policy_id", "customer_policy_id"),
    )

    claim_number: Mapped[str] = mapped_column(String(50), unique=True)
    # NULL when the claim arrived without a (resolvable) policy number.
    customer_policy_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customer_policies.id")
    )
    # Who uploaded it (separation of duties: an officer does not approve their own upload).
    # NULL for seeded claims and for claims that predate authentication.
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    hospital_name: Mapped[str | None] = mapped_column(String(255))
    diagnosis_text: Mapped[str | None] = mapped_column(Text)
    icd10_code: Mapped[str | None] = mapped_column(String(16))
    admission_date: Mapped[date | None] = mapped_column(Date)
    discharge_date: Mapped[date | None] = mapped_column(Date)
    claimed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    approved_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    status: Mapped[ClaimStatus] = mapped_column(
        pg_enum(ClaimStatus, "claim_status"), server_default=ClaimStatus.SUBMITTED.value
    )
    source: Mapped[ClaimSource] = mapped_column(
        pg_enum(ClaimSource, "claim_source"), server_default=ClaimSource.UPLOAD.value
    )

    customer_policy: Mapped[CustomerPolicy | None] = relationship()
    documents: Mapped[list["ClaimDocument"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", passive_deletes=True
    )


class ClaimDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "claim_documents"
    __table_args__ = (
        UniqueConstraint("claim_id", "file_path"),
        Index("ix_claim_documents_claim_id", "claim_id"),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    doc_type: Mapped[DocType] = mapped_column(pg_enum(DocType, "doc_type"))
    file_path: Mapped[str] = mapped_column(Text)  # relative to settings.data_dir
    mime_type: Mapped[str] = mapped_column(String(100))
    is_scanned: Mapped[bool] = mapped_column(server_default="false")
    page_count: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))

    claim: Mapped[Claim] = relationship(back_populates="documents")


class ClaimExtraction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "claim_extractions"
    __table_args__ = (
        UniqueConstraint("claim_id", "version"),
        Index("ix_claim_extractions_claim_id", "claim_id"),
        Index("ix_claim_extractions_extracted_data", "extracted_data", postgresql_using="gin"),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    extracted_data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    model_used: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)
    validation_errors: Mapped[list[Any]] = mapped_column(JSONB, server_default=_EMPTY_LIST)
    version: Mapped[int] = mapped_column(Integer, server_default=text("1"))


class ClaimDecision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "claim_decisions"
    __table_args__ = (Index("ix_claim_decisions_claim_id", "claim_id"),)

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    decision: Mapped[DecisionType] = mapped_column(pg_enum(DecisionType, "decision_type"))
    payable_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    deductions: Mapped[list[Any]] = mapped_column(JSONB, server_default=_EMPTY_LIST)
    reasons: Mapped[list[Any]] = mapped_column(JSONB, server_default=_EMPTY_LIST)
    # [{chunk_id, section_path, page}]
    cited_clauses: Mapped[list[Any]] = mapped_column(JSONB, server_default=_EMPTY_LIST)
    risk_flags: Mapped[list[Any]] = mapped_column(JSONB, server_default=_EMPTY_LIST)
    agent_run_id: Mapped[str | None] = mapped_column(String(100))
    model_used: Mapped[str | None] = mapped_column(String(100))


class ClaimApproval(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "claim_approvals"
    __table_args__ = (
        Index("ix_claim_approvals_claim_id", "claim_id"),
        Index("ix_claim_approvals_decision_id", "decision_id"),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    decision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claim_decisions.id", ondelete="CASCADE")
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    action: Mapped[ApprovalAction] = mapped_column(pg_enum(ApprovalAction, "approval_action"))
    final_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    comment: Mapped[str | None] = mapped_column(Text)
