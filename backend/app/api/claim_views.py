"""Read-only views of claims for the web UI: the list, one claim with its history, and the files.

Everything is for officers and admins. Personal data is shown the way an officer needs it: the
customer's name, but the phone number only masked."""

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.claims import get_intake_service
from app.auth.deps import CurrentUser, require_officer
from app.core.db import get_session
from app.intake.service import IntakeService
from app.models import (
    Claim,
    ClaimApproval,
    ClaimDecision,
    ClaimDocument,
    ClaimStatus,
    Customer,
    CustomerPolicy,
    Policy,
    User,
)

router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(require_officer)])

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def mask_phone(phone: str | None) -> str | None:
    """The last four digits only: "98XXXXXX10" style numbers become "******3210"."""
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    return ("*" * max(len(digits) - 4, 0)) + digits[-4:] if len(digits) > 4 else "*" * len(digits)


class ClaimListItem(BaseModel):
    id: UUID
    claim_number: str
    status: str
    customer_name: str | None
    customer_phone_masked: str | None
    hospital_name: str | None
    diagnosis_text: str | None
    claimed_amount: Decimal
    approved_amount: Decimal | None
    policy_code: str | None
    created_at: datetime
    updated_at: datetime


class ClaimListResponse(BaseModel):
    items: list[ClaimListItem]
    total: int
    page: int
    size: int


class DocumentOut(BaseModel):
    id: UUID
    file_name: str
    doc_type: str
    mime_type: str
    page_count: int | None
    is_scanned: bool
    sha256: str


class ApprovalOut(BaseModel):
    action: str
    final_amount: Decimal | None
    comment: str | None
    reviewer_id: UUID
    reviewer_name: str
    created_at: datetime


class HistoryItem(BaseModel):
    decision_id: UUID
    run_id: str | None
    decision: str
    payable_amount: Decimal | None
    model_used: str | None
    created_at: datetime
    approval: ApprovalOut | None


class CustomerOut(BaseModel):
    name: str
    phone_masked: str | None


class ClaimDetail(BaseModel):
    id: UUID
    claim_number: str
    status: str
    claimed_amount: Decimal
    approved_amount: Decimal | None
    policy_code: str | None
    policy_name: str | None
    customer: CustomerOut | None
    hospital_name: str | None
    diagnosis_text: str | None
    icd10_code: str | None
    admission_date: date | None
    discharge_date: date | None
    created_at: datetime
    updated_at: datetime
    uploaded_by_me: bool  # separation of duties: the uploader may not approve it
    latest_run_id: str | None
    documents: list[DocumentOut]
    history: list[HistoryItem]


def _claim_query() -> Any:
    return (
        select(Claim, Customer, Policy.policy_code, Policy.name)
        .outerjoin(CustomerPolicy, CustomerPolicy.id == Claim.customer_policy_id)
        .outerjoin(Customer, Customer.id == CustomerPolicy.customer_id)
        .outerjoin(Policy, Policy.id == CustomerPolicy.policy_id)
    )


@router.get("")
async def list_claims(
    status: Annotated[list[ClaimStatus] | None, Query()] = None,
    q: Annotated[str | None, Query(max_length=60, description="part of a claim number")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_session),
) -> ClaimListResponse:
    conditions = []
    if status:
        conditions.append(Claim.status.in_(status))
    if q and q.strip():
        needle = q.strip().replace("%", r"\%").replace("_", r"\_")
        conditions.append(Claim.claim_number.ilike(f"%{needle}%", escape="\\"))
    total = (
        await session.execute(select(func.count()).select_from(Claim).where(*conditions))
    ).scalar_one()
    rows = (
        await session.execute(
            _claim_query()
            .where(*conditions)
            .order_by(Claim.updated_at.desc(), Claim.claim_number)
            .limit(size)
            .offset((page - 1) * size)
        )
    ).all()
    items = [
        ClaimListItem(
            id=claim.id,
            claim_number=claim.claim_number,
            status=claim.status.value,
            customer_name=customer.full_name if customer else None,
            customer_phone_masked=mask_phone(customer.phone) if customer else None,
            hospital_name=claim.hospital_name,
            diagnosis_text=claim.diagnosis_text,
            claimed_amount=claim.claimed_amount,
            approved_amount=claim.approved_amount,
            policy_code=policy_code,
            created_at=claim.created_at,
            updated_at=claim.updated_at,
        )
        for claim, customer, policy_code, _ in rows
    ]
    return ClaimListResponse(items=items, total=total, page=page, size=size)


def _document_out(document: ClaimDocument) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        file_name=document.file_path.rsplit("/", 1)[-1],
        doc_type=document.doc_type.value,
        mime_type=document.mime_type,
        page_count=document.page_count,
        is_scanned=document.is_scanned,
        sha256=document.sha256,
    )


async def _latest_run_id(
    request: Request, claim_id: UUID, history: list[HistoryItem]
) -> str | None:
    try:  # the run a client started most recently (it may not have a decision yet)
        cached = await request.app.state.redis.get(f"claim:{claim_id}:latest_run")
    except Exception:  # noqa: BLE001 - a missing hint, not an error
        cached = None
    if cached:
        return cached.decode() if isinstance(cached, bytes) else str(cached)
    return next((h.run_id for h in history if h.run_id), None)


@router.get("/{claim_number}")
async def get_claim(
    claim_number: str,
    request: Request,
    user: CurrentUser = Depends(require_officer),
    session: AsyncSession = Depends(get_session),
) -> ClaimDetail:
    row = (await session.execute(_claim_query().where(Claim.claim_number == claim_number))).first()
    if row is None:
        raise HTTPException(404, f"claim {claim_number} not found")
    claim, customer, policy_code, policy_name = row
    documents = (
        (
            await session.execute(
                select(ClaimDocument)
                .where(ClaimDocument.claim_id == claim.id)
                .order_by(ClaimDocument.created_at, ClaimDocument.file_path)
            )
        )
        .scalars()
        .all()
    )
    decisions = (
        (
            await session.execute(
                select(ClaimDecision)
                .where(ClaimDecision.claim_id == claim.id)
                .order_by(ClaimDecision.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    approvals = (
        await session.execute(
            select(ClaimApproval, User.full_name)
            .join(User, User.id == ClaimApproval.reviewer_id)
            .where(ClaimApproval.claim_id == claim.id)
            .order_by(ClaimApproval.created_at.desc())
        )
    ).all()
    by_decision: dict[UUID, ApprovalOut] = {}
    for approval, reviewer_name in approvals:  # newest first: keep the latest per decision
        by_decision.setdefault(
            approval.decision_id,
            ApprovalOut(
                action=approval.action.value,
                final_amount=approval.final_amount,
                comment=approval.comment,
                reviewer_id=approval.reviewer_id,
                reviewer_name=reviewer_name,
                created_at=approval.created_at,
            ),
        )
    history = [
        HistoryItem(
            decision_id=d.id,
            run_id=d.agent_run_id,
            decision=d.decision.value,
            payable_amount=d.payable_amount,
            model_used=d.model_used,
            created_at=d.created_at,
            approval=by_decision.get(d.id),
        )
        for d in decisions
    ]
    return ClaimDetail(
        id=claim.id,
        claim_number=claim.claim_number,
        status=claim.status.value,
        claimed_amount=claim.claimed_amount,
        approved_amount=claim.approved_amount,
        policy_code=policy_code,
        policy_name=policy_name,
        customer=(
            CustomerOut(name=customer.full_name, phone_masked=mask_phone(customer.phone))
            if customer
            else None
        ),
        hospital_name=claim.hospital_name,
        diagnosis_text=claim.diagnosis_text,
        icd10_code=claim.icd10_code,
        admission_date=claim.admission_date,
        discharge_date=claim.discharge_date,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        uploaded_by_me=claim.created_by is not None and claim.created_by == user.id,
        latest_run_id=await _latest_run_id(request, claim.id, history),
        documents=[_document_out(d) for d in documents],
        history=history,
    )


@router.get("/{claim_id}/documents/{document_id}/file")
async def get_document_file(
    claim_id: UUID,
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
    service: IntakeService = Depends(get_intake_service),
) -> Response:
    """The stored bytes of one document, for the viewer (inline, never a download prompt)."""
    document = (
        await session.execute(
            select(ClaimDocument).where(
                ClaimDocument.id == document_id, ClaimDocument.claim_id == claim_id
            )
        )
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(404, "document not found")
    try:
        data = service.storage.read(document.file_path)
    except FileNotFoundError as exc:
        raise HTTPException(404, "the file is no longer in storage") from exc
    name = _SAFE_NAME.sub("_", document.file_path.rsplit("/", 1)[-1]) or "document"
    return Response(
        content=data,
        media_type=document.mime_type,
        headers={
            "Content-Disposition": f'inline; filename="{name}"',
            "Cache-Control": "private, max-age=300",
        },
    )
