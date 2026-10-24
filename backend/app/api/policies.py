"""Policy ingestion endpoints. No auth yet: that arrives with Module 10."""

import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.db import get_session
from app.core.llm import get_embeddings
from app.ingestion.config import IngestionSettings, get_ingestion_settings
from app.ingestion.embedder import AsyncEmbedder
from app.ingestion.pipeline import IngestionPipeline
from app.models import Policy, PolicyChunk
from app.rag.cache import SemanticCache
from app.rag.config import get_rag_settings
from app.repositories import IngestionRunRepository, PolicyRepository
from app.schemas.ingestion import (
    IngestRequest,
    IngestRunRead,
    IngestRunRef,
    IngestStarted,
    PolicySummary,
)
from app.schemas.policy import PolicyChunkRead

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/policies", tags=["policies"])


async def run_ingestion_task(
    session_factory: async_sessionmaker[AsyncSession],
    embedder_factory: Callable[[], AsyncEmbedder],
    settings: IngestionSettings,
    runs: list[tuple[UUID, str]],
    force: bool,
    on_policy_changed: Callable[[str], Awaitable[object]] | None = None,
) -> None:
    """Background job: one session and one transaction per policy."""
    for run_id, policy_code in runs:
        async with session_factory() as session:
            try:
                run = await IngestionRunRepository(session).get(run_id)
                policy = await PolicyRepository(session).get_by_code(policy_code)
                if run is None or policy is None:
                    continue
                pipeline = IngestionPipeline(
                    session, embedder_factory(), settings, on_policy_changed=on_policy_changed
                )
                result = await pipeline.ingest(policy, force=force, run=run)
                await session.commit()
                await pipeline.after_commit(result)
            except Exception:
                logger.exception("ingest_task_failed", extra={"policy": policy_code})
                await session.rollback()


@router.post("/ingest", status_code=202)
async def start_ingestion(
    body: IngestRequest,
    request: Request,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> IngestStarted:
    repo = PolicyRepository(session)
    if body.policy_code:
        policy = await repo.get_by_code(body.policy_code)
        if policy is None:
            raise HTTPException(404, f"unknown policy {body.policy_code!r}")
        policies = [policy]
    else:
        policies = await repo.list_active()

    run_repo = IngestionRunRepository(session)
    for policy in policies:
        if await run_repo.has_recent_running(policy.id):
            raise HTTPException(409, f"an ingestion for {policy.policy_code} is already running")

    # Runs are created (and committed) up front so the returned run_id can be polled at once.
    refs: list[IngestRunRef] = []
    for policy in policies:
        run = await run_repo.create(policy.id)
        refs.append(IngestRunRef(run_id=run.id, policy_code=policy.policy_code))
    await session.commit()
    if not refs:
        raise HTTPException(404, "no active policies to ingest")

    background.add_task(
        run_ingestion_task,
        request.app.state.session_factory,
        getattr(request.app.state, "embedder_factory", get_embeddings),
        get_ingestion_settings(),
        [(r.run_id, r.policy_code) for r in refs],
        body.force,
        SemanticCache(request.app.state.redis, get_rag_settings()).invalidate_policy,
    )
    return IngestStarted(run_id=refs[0].run_id, runs=refs)


@router.get("/ingest/{run_id}")
async def get_ingestion_run(
    run_id: UUID, session: AsyncSession = Depends(get_session)
) -> IngestRunRead:
    run = await IngestionRunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return IngestRunRead.model_validate(run)


@router.get("")
async def list_policies(session: AsyncSession = Depends(get_session)) -> list[PolicySummary]:
    counts = dict(
        (
            await session.execute(
                select(PolicyChunk.policy_id, func.count()).group_by(PolicyChunk.policy_id)
            )
        ).all()
    )
    runs = IngestionRunRepository(session)
    policies = (await session.execute(select(Policy).order_by(Policy.policy_code))).scalars()
    summaries = []
    for policy in policies:
        last = await runs.last_for_policy(policy.id)
        summaries.append(
            PolicySummary(
                policy_code=policy.policy_code,
                name=policy.name,
                source_pdf=policy.source_pdf,
                version=policy.version,
                is_active=policy.is_active,
                chunk_count=counts.get(policy.id, 0),
                last_run=IngestRunRead.model_validate(last) if last else None,
            )
        )
    return summaries


@router.get("/{policy_code}/chunks")
async def list_chunks(
    policy_code: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[PolicyChunkRead]:
    """Chunk preview for inspecting ingestion quality. Embeddings are never returned."""
    policy = await PolicyRepository(session).get_by_code(policy_code)
    if policy is None:
        raise HTTPException(404, f"unknown policy {policy_code!r}")
    rows = (
        await session.execute(
            select(PolicyChunk)
            .where(PolicyChunk.policy_id == policy.id)
            .order_by(PolicyChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
    ).scalars()
    chunks = []
    for row in rows:
        chunk = PolicyChunkRead.model_validate(row)
        chunk.meta = {k: v for k, v in chunk.meta.items() if k != "embed_text"}
        chunks.append(chunk)
    return chunks
