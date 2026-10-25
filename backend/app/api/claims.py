"""Claim upload and extraction endpoints. No auth yet (Module 10)."""

import logging
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.deps import CurrentUser, llm_guard, require_officer
from app.core.config import get_settings
from app.core.db import get_session
from app.core.llm import get_chat_model
from app.intake.config import get_intake_settings
from app.intake.consistency import ConsistencyIssue
from app.intake.service import ClaimNotFoundError, IntakeService
from app.intake.storage import FileTooLargeError, UploadError
from app.models import Claim, ClaimDocument
from app.repositories.extraction import ExtractionRepository
from app.schemas.intake import (
    ClaimUploadResponse,
    ExtractionResponse,
    ExtractStarted,
    UploadedDocument,
)
from app.worker.queue import enqueue_extraction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(require_officer)])


def get_intake_service(request: Request) -> IntakeService:
    service = getattr(request.app.state, "intake_service", None)
    if service is None:
        service = IntakeService(
            request.app.state.session_factory,
            lambda: get_chat_model(temperature=0),
            data_dir=get_settings().data_dir,
        )
        request.app.state.intake_service = service
    return service


async def read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read the files with a hard size cap, so an oversized upload is refused without loading it."""
    settings = get_intake_settings()
    if not files:
        raise UploadError("send at least one file")
    if len(files) > settings.max_files_per_claim:
        raise UploadError(f"at most {settings.max_files_per_claim} files per request")
    limit = settings.max_upload_mb * 1024 * 1024
    out: list[tuple[str, bytes]] = []
    for upload in files:
        name = Path(upload.filename or "upload").name
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise FileTooLargeError(f"{name!r} is larger than {settings.max_upload_mb} MB")
        out.append((name, data))
    return out


def describe(document: ClaimDocument) -> UploadedDocument:
    return UploadedDocument(
        id=document.id,
        file_name=Path(document.file_path).name,
        doc_type=document.doc_type.value,
        sha256=document.sha256,
        page_count=document.page_count,
        is_scanned=document.is_scanned,
    )


async def load_claim(session: AsyncSession, claim_id: UUID) -> Claim:
    claim = (
        await session.execute(
            select(Claim).where(Claim.id == claim_id).options(selectinload(Claim.documents))
        )
    ).scalar_one_or_none()
    if claim is None:
        raise ClaimNotFoundError(f"claim {claim_id} not found")
    return claim


@router.post("", status_code=201)
async def create_claim(
    policy_number: Annotated[str, Form(min_length=3, max_length=100)],
    files: Annotated[list[UploadFile], File()],
    claim_number: Annotated[str | None, Form(max_length=50)] = None,
    claimed_amount: Annotated[Decimal, Form(ge=0)] = Decimal(0),
    session: AsyncSession = Depends(get_session),
    service: IntakeService = Depends(get_intake_service),
    user: CurrentUser = Depends(require_officer),
) -> ClaimUploadResponse:
    """Create a claim (source=upload). Every file is validated before anything is stored."""
    uploads = await read_uploads(files)
    claim, ignored = await service.create_claim(
        session,
        policy_number=policy_number,
        files=uploads,
        claim_number=claim_number,
        claimed_amount=claimed_amount,
    )
    claim.created_by = user.id  # who uploaded it: that officer cannot approve it
    await session.commit()
    stored = await load_claim(session, claim.id)
    return ClaimUploadResponse(
        id=claim.id,
        claim_number=claim.claim_number,
        status=stored.status.value,
        documents=[describe(d) for d in stored.documents],
        duplicates_ignored=len(ignored),
    )


@router.post("/{claim_id}/documents")
async def add_documents(
    claim_id: UUID,
    files: Annotated[list[UploadFile], File()],
    session: AsyncSession = Depends(get_session),
    service: IntakeService = Depends(get_intake_service),
) -> ClaimUploadResponse:
    uploads = await read_uploads(files)
    claim = await load_claim(session, claim_id)
    _, ignored = await service.add_documents(session, claim, uploads)
    await session.commit()
    stored = await load_claim(session, claim_id)
    return ClaimUploadResponse(
        id=claim.id,
        claim_number=claim.claim_number,
        status=stored.status.value,
        documents=[describe(d) for d in stored.documents],
        duplicates_ignored=len(ignored),
    )


@router.post("/{claim_id}/extract", status_code=202)
async def extract(
    claim_id: UUID,
    request: Request,
    background: BackgroundTasks,
    prefer: Literal["digital", "scanned"] = "digital",
    session: AsyncSession = Depends(get_session),
    service: IntakeService = Depends(get_intake_service),
    _user: CurrentUser = Depends(llm_guard),
) -> ExtractStarted:
    """Run intake in the background (works for seeded claims too). Poll GET .../extraction."""
    claim = await load_claim(session, claim_id)
    if not claim.documents:
        raise HTTPException(422, "the claim has no documents to extract from")
    queue = getattr(request.app.state, "queue", None)
    if queue is not None:  # the worker runs it; the lock keeps it to one at a time
        if not await enqueue_extraction(queue, str(claim_id), prefer):
            raise HTTPException(409, "an extraction for this claim is already running")
        return ExtractStarted(claim_id=claim_id, status="queued", prefer=prefer)
    if claim_id in service._running:
        raise HTTPException(409, "an extraction for this claim is already running")
    background.add_task(service.process, claim_id, prefer=prefer)
    return ExtractStarted(claim_id=claim_id, status="processing", prefer=prefer)


@router.get("/{claim_id}/extraction")
async def get_extraction(
    claim_id: UUID, session: AsyncSession = Depends(get_session)
) -> ExtractionResponse:
    claim = await load_claim(session, claim_id)
    row = await ExtractionRepository(session).latest_for_claim(claim_id)
    if row is None:
        raise HTTPException(404, f"no extraction yet (claim status: {claim.status.value})")
    data = dict(row.extracted_data)
    return ExtractionResponse(
        claim_id=claim.id,
        claim_number=claim.claim_number,
        claim_status=claim.status.value,
        version=row.version,
        model_used=row.model_used,
        confidence=row.confidence,
        validation_errors=list(row.validation_errors or []),
        missing_documents=list(data.get("missing_documents", [])),
        consistency_issues=[ConsistencyIssue(**i) for i in data.get("consistency_issues", [])],
        extraction=data,
        created_at=row.created_at,
    )
