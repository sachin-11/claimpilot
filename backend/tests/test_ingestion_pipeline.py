import json
from pathlib import Path

import pytest
from pdf_fixtures import (
    FakeEmbedder,
    ingestion_settings,
    make_policy_pdf,
    whitespace_tokens,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.config import IngestionSettings
from app.ingestion.pipeline import IngestionPipeline
from app.models import AuditLog, IngestionRun, IngestionStatus, Policy, PolicyChunk
from app.repositories import PolicyRepository


@pytest.fixture
def cfg(tmp_path: Path) -> IngestionSettings:
    return ingestion_settings(tmp_path)


@pytest.fixture
async def policy(session: AsyncSession) -> Policy:
    created, _ = await PolicyRepository(session).upsert_from_rules(
        {"policy_id": "POL-T", "name": "Test Policy", "source_pdf": "policy.pdf"}
    )
    return created


def make_pipeline(
    session: AsyncSession, embedder: FakeEmbedder, cfg: IngestionSettings
) -> IngestionPipeline:
    return IngestionPipeline(session, embedder, cfg, tokens=whitespace_tokens)


async def chunk_count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(PolicyChunk))).scalar_one()


async def all_chunks(session: AsyncSession) -> list[PolicyChunk]:
    rows = await session.execute(select(PolicyChunk).order_by(PolicyChunk.chunk_index))
    return list(rows.scalars())


async def test_first_ingestion_stores_chunks_embeddings_run_and_audit(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    make_policy_pdf(cfg.policies_dir / "policy.pdf")
    embedder = FakeEmbedder()

    result = await make_pipeline(session, embedder, cfg).ingest(policy)

    assert result.status == "success", result.error
    assert result.chunks_total == result.chunks_new == embedder.texts_embedded > 3
    chunks = await all_chunks(session)
    assert len(chunks) == result.chunks_total
    assert all(c.embedding is not None for c in chunks)
    assert all(c.search_tsv for c in chunks)
    sample = chunks[0]
    assert sample.meta["policy_code"] == "POL-T"
    assert sample.meta["source_pdf"] == "policy.pdf"
    assert sample.meta["embed_text"].startswith("Policy: Test Policy | Section: ")
    assert not sample.content.startswith("Policy:")  # content is stored as is

    run = (await session.execute(select(IngestionRun))).scalar_one()
    assert run.status == IngestionStatus.SUCCESS
    assert run.chunks_total == result.chunks_total and run.tokens_used > 0
    assert run.file_sha256 and run.finished_at is not None
    audit = (
        await session.execute(select(AuditLog).where(AuditLog.action == "ingestion.completed"))
    ).scalar_one()
    assert audit.entity_id == str(run.id)

    debug = (cfg.debug_dir / "chunks_POL-T.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(debug) == result.chunks_total
    first = json.loads(debug[0])
    assert set(first) == {
        "chunk_index", "section_path", "pages", "token_count", "has_table", "clauses", "preview"
    }  # fmt: skip


async def test_rerun_with_unchanged_file_embeds_nothing(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    make_policy_pdf(cfg.policies_dir / "policy.pdf")
    embedder = FakeEmbedder()
    pipeline = make_pipeline(session, embedder, cfg)
    first = await pipeline.ingest(policy)
    calls_after_first = len(embedder.calls)

    skipped = await pipeline.ingest(policy)  # same file hash -> whole file skipped
    forced = await pipeline.ingest(policy, force=True)  # re-parsed, but all hashes unchanged

    assert skipped.status == "skipped" and skipped.chunks_total == first.chunks_total
    assert forced.status == "success"
    assert (forced.chunks_new, forced.chunks_deleted, forced.chunks_embedded) == (0, 0, 0)
    assert len(embedder.calls) == calls_after_first  # no embedding call at all
    assert await chunk_count(session) == first.chunks_total


async def test_changing_one_section_reembeds_only_that_chunk(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    pdf = cfg.policies_dir / "policy.pdf"
    make_policy_pdf(pdf, cataract_months=24)
    embedder = FakeEmbedder()
    pipeline = make_pipeline(session, embedder, cfg)
    first = await pipeline.ingest(policy)
    before = {c.content_hash: c.embedding for c in await all_chunks(session)}

    make_policy_pdf(pdf, cataract_months=36)  # only the waiting-period table changes
    embedder.calls.clear()
    second = await pipeline.ingest(policy)

    assert second.status == "success", second.error
    assert second.chunks_total == first.chunks_total
    assert second.chunks_new == second.chunks_deleted == second.chunks_embedded == 1
    assert embedder.texts_embedded == 1
    assert "36 months" in embedder.calls[0][0]

    after = {c.content_hash: c for c in await all_chunks(session)}
    assert len(after) == first.chunks_total
    assert not any("| Cataract | 24 months |" in c.content for c in after.values())
    assert any("| Cataract | 36 months |" in c.content for c in after.values())
    kept = set(before) & set(after)
    assert len(kept) == first.chunks_total - 1  # every other chunk is untouched
    assert all(after[h].embedding is not None for h in kept)


async def test_failure_midway_rolls_back_and_marks_run_failed(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    pdf = cfg.policies_dir / "policy.pdf"
    make_policy_pdf(pdf, cataract_months=24)
    first = await make_pipeline(session, FakeEmbedder(), cfg).ingest(policy)
    state_before = {c.content_hash for c in await all_chunks(session)}

    make_policy_pdf(pdf, cataract_months=36)
    failed = await make_pipeline(session, FakeEmbedder(fail=True), cfg).ingest(policy)

    assert failed.status == "failed" and "embedding service down" in failed.error
    # chunk changes made before the failure (the new row) were rolled back
    assert {c.content_hash for c in await all_chunks(session)} == state_before
    assert len(state_before) == first.chunks_total
    runs = (
        (await session.execute(select(IngestionRun).order_by(IngestionRun.started_at)))
        .scalars()
        .all()
    )
    assert [r.status for r in runs] == [IngestionStatus.SUCCESS, IngestionStatus.FAILED]
    assert "embedding service down" in runs[1].error
    assert runs[1].finished_at is not None
    failed_audit = (
        await session.execute(select(AuditLog).where(AuditLog.action == "ingestion.failed"))
    ).scalar_one()
    assert failed_audit.details["error"] == runs[1].error

    # a failed run is not a "last success": the next run really ingests the change
    retry = await make_pipeline(session, FakeEmbedder(), cfg).ingest(policy)
    assert retry.status == "success" and retry.chunks_new == 1


async def test_missing_pdf_fails_the_run(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    result = await make_pipeline(session, FakeEmbedder(), cfg).ingest(policy)

    assert result.status == "failed" and "policy.pdf" in result.error
    run = (await session.execute(select(IngestionRun))).scalar_one()
    assert run.status == IngestionStatus.FAILED and run.file_sha256 is None


async def test_dry_run_writes_debug_file_but_nothing_else(
    session: AsyncSession, policy: Policy, cfg: IngestionSettings
) -> None:
    make_policy_pdf(cfg.policies_dir / "policy.pdf")

    result = await IngestionPipeline(session, None, cfg, tokens=whitespace_tokens).ingest(
        policy, dry_run=True
    )

    assert result.status == "dry_run" and result.stats is not None
    assert result.stats.chunks == result.chunks_total > 3
    assert result.stats.with_tables == 1
    assert result.stats.min_tokens <= result.stats.avg_tokens <= result.stats.max_tokens
    assert await chunk_count(session) == 0
    assert (await session.execute(select(func.count()).select_from(IngestionRun))).scalar_one() == 0
    assert (cfg.debug_dir / "chunks_POL-T.jsonl").exists()
