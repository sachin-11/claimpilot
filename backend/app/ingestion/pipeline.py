"""Ingest one policy end to end: parse -> clean -> chunk -> upsert -> embed -> delete stale.

Transaction model: the caller owns the outer transaction and commits. The database work of a
policy runs inside a SAVEPOINT, so a failure rolls back every chunk change of that policy while
the ingestion_runs row (created before, marked failed after) survives and is committed.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import Chunk, TokenCounter, chunk_blocks, count_tokens, embed_text
from app.ingestion.cleaner import clean_blocks
from app.ingestion.config import IngestionSettings
from app.ingestion.embedder import AsyncEmbedder, embed_texts
from app.ingestion.parser import assign_structure, extract_blocks
from app.models import ActorType, IngestionRun, IngestionStatus, Policy, PolicyChunk
from app.repositories import (
    AuditLogRepository,
    IngestionRunRepository,
    PolicyChunkRepository,
)
from app.schemas.policy import PolicyChunkCreate

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 300


@dataclass
class ChunkStats:
    chunks: int = 0
    min_tokens: int = 0
    avg_tokens: float = 0.0
    max_tokens: int = 0
    with_tables: int = 0
    skipped_pages: list[int] = field(default_factory=list)


@dataclass
class IngestResult:
    policy_code: str
    status: str  # success | skipped | failed | dry_run
    run_id: UUID | None = None
    chunks_total: int = 0
    chunks_new: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    chunks_embedded: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    stats: ChunkStats | None = None
    error: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_stats(chunks: list[Chunk], skipped_pages: list[int]) -> ChunkStats:
    if not chunks:
        return ChunkStats(skipped_pages=skipped_pages)
    sizes = [c.token_count for c in chunks]
    return ChunkStats(
        chunks=len(chunks),
        min_tokens=min(sizes),
        avg_tokens=sum(sizes) / len(sizes),
        max_tokens=max(sizes),
        with_tables=sum(c.has_table for c in chunks),
        skipped_pages=skipped_pages,
    )


def build_chunks(
    pdf_path: Path, settings: IngestionSettings, tokens: TokenCounter = count_tokens
) -> tuple[list[Chunk], list[int]]:
    """Parse -> clean -> structure -> chunk. CPU bound, so callers run it in a thread."""
    parsed = extract_blocks(pdf_path, min_page_chars=settings.min_page_chars)
    cleaned, _ = clean_blocks(parsed.blocks)
    chunks = chunk_blocks(assign_structure(cleaned), settings, tokens)
    return chunks, parsed.skipped_pages


def write_debug_file(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_index": chunk.chunk_index,
                "section_path": chunk.section_path,
                "pages": [chunk.page_start, chunk.page_end],
                "token_count": chunk.token_count,
                "has_table": chunk.has_table,
                "clauses": chunk.clauses,
                "preview": chunk.content[:PREVIEW_CHARS],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunk_to_create(chunk: Chunk, policy: Policy) -> PolicyChunkCreate:
    return PolicyChunkCreate(
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        content_hash=chunk.content_hash,
        section_path=chunk.section_path or None,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        token_count=chunk.token_count,
        meta={
            "policy_code": policy.policy_code,
            "insurer": policy.insurer,
            "block_types": chunk.block_types,
            "has_table": chunk.has_table,
            "headings": chunk.headings,
            "clauses": chunk.clauses,
            "source_pdf": policy.source_pdf,
            # Stored (not recomputed) so the embedding input is inspectable and reproducible.
            "embed_text": embed_text(policy.name, chunk),
        },
    )


class IngestionPipeline:
    def __init__(
        self,
        session: AsyncSession,
        embedder: AsyncEmbedder | None,
        settings: IngestionSettings,
        tokens: TokenCounter = count_tokens,
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.settings = settings
        self.tokens = tokens
        self.runs = IngestionRunRepository(session)
        self.chunk_repo = PolicyChunkRepository(session)
        self.audit = AuditLogRepository(session)

    def pdf_path(self, policy: Policy) -> Path:
        if not policy.source_pdf:
            raise FileNotFoundError(f"policy {policy.policy_code} has no source_pdf")
        path = self.settings.policies_dir / policy.source_pdf
        if not path.is_file():
            raise FileNotFoundError(f"policy PDF not found: {path}")
        return path

    async def ingest(
        self,
        policy: Policy,
        *,
        force: bool = False,
        dry_run: bool = False,
        run: IngestionRun | None = None,
    ) -> IngestResult:
        """Ingest one policy. Never raises for ingestion problems: check result.status."""
        if dry_run:
            return await self._dry_run(policy)

        run = run or await self.runs.create(policy.id)
        result = IngestResult(policy_code=policy.policy_code, status="failed", run_id=run.id)
        try:
            path = self.pdf_path(policy)
            run.file_sha256 = await asyncio.to_thread(file_sha256, path)

            if not force and await self._unchanged_since_last_success(policy, run):
                result.status = "skipped"
                result.chunks_total = await self.chunk_repo.count_by_policy(policy.id)
                logger.info("ingest_skipped_unchanged", extra={"policy": policy.policy_code})
                await self._finish(run, policy, result)
                return result

            chunks, skipped_pages = await asyncio.to_thread(
                build_chunks, path, self.settings, self.tokens
            )
            result.stats = compute_stats(chunks, skipped_pages)
            await asyncio.to_thread(self._write_debug, policy, chunks)
            if not chunks:
                raise ValueError(f"no chunks produced from {path.name}")

            async with self.session.begin_nested():
                await self._store(policy, chunks, result)
            result.status = "success"
        except Exception as exc:  # noqa: BLE001 - recorded on the run, never lost
            logger.error(
                "ingest_failed",
                extra={"policy": policy.policy_code},
                exc_info=not isinstance(exc, FileNotFoundError),
            )
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"[:2000]
            # The savepoint (if any) is already rolled back: no partial chunks survive.
            result.chunks_new = result.chunks_updated = result.chunks_deleted = 0
            result.chunks_embedded = result.tokens_used = 0
        await self._finish(run, policy, result)
        return result

    # ----------------------------------------------------------------------------------

    async def _dry_run(self, policy: Policy) -> IngestResult:
        result = IngestResult(policy_code=policy.policy_code, status="dry_run")
        try:
            chunks, skipped_pages = await asyncio.to_thread(
                build_chunks, self.pdf_path(policy), self.settings, self.tokens
            )
            await asyncio.to_thread(self._write_debug, policy, chunks)
            result.stats = compute_stats(chunks, skipped_pages)
            result.chunks_total = len(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "dry_run_failed",
                extra={"policy": policy.policy_code},
                exc_info=not isinstance(exc, FileNotFoundError),
            )
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"[:2000]
        return result

    def _write_debug(self, policy: Policy, chunks: list[Chunk]) -> None:
        target = self.settings.debug_dir / f"chunks_{policy.policy_code}.jsonl"
        write_debug_file(target, chunks)

    async def _unchanged_since_last_success(self, policy: Policy, run: IngestionRun) -> bool:
        last = await self.runs.last_success(policy.id, exclude_run_id=run.id)
        if last is None or last.file_sha256 != run.file_sha256:
            return False
        return await self.chunk_repo.count_by_policy(policy.id) > 0

    async def _store(self, policy: Policy, chunks: list[Chunk], result: IngestResult) -> None:
        """Upsert, embed what lacks an embedding, delete stale chunks (inside the savepoint)."""
        creates = [chunk_to_create(c, policy) for c in chunks]
        upsert = await self.chunk_repo.bulk_upsert(policy.id, creates)
        result.chunks_total = len(chunks)
        result.chunks_new = upsert.inserted
        result.chunks_updated = upsert.updated

        # Embedding IS NULL covers new chunks and any that a previous failed run left behind;
        # unchanged chunks (same content_hash) keep their existing embedding.
        pending = (
            await self.session.execute(
                select(PolicyChunk.id, PolicyChunk.meta).where(
                    PolicyChunk.policy_id == policy.id, PolicyChunk.embedding.is_(None)
                )
            )
        ).all()
        if pending:
            if self.embedder is None:
                raise RuntimeError("an embedder is required when chunks need embeddings")
            embedded = await embed_texts(
                [meta["embed_text"] for _, meta in pending],
                self.embedder,
                self.settings,
                tokens=self.tokens,
            )
            await self.session.execute(
                update(PolicyChunk),
                [
                    {"id": chunk_id, "embedding": vector}
                    for (chunk_id, _), vector in zip(pending, embedded.vectors, strict=True)
                ],
            )
            result.chunks_embedded = len(pending)
            result.tokens_used = embedded.tokens_used
            result.estimated_cost_usd = embedded.estimated_cost_usd

        result.chunks_deleted = await self.chunk_repo.delete_stale(
            policy.id, {c.content_hash for c in chunks}
        )

    async def _finish(self, run: IngestionRun, policy: Policy, result: IngestResult) -> None:
        run.status = (
            IngestionStatus.FAILED if result.status == "failed" else IngestionStatus.SUCCESS
        )
        run.finished_at = datetime.now(UTC)
        run.chunks_total = result.chunks_total
        run.chunks_new = result.chunks_new
        run.chunks_deleted = result.chunks_deleted
        run.tokens_used = result.tokens_used
        run.error = result.error
        await self.session.flush()

        details = {
            "policy_code": policy.policy_code,
            "status": result.status,
            "chunks_total": result.chunks_total,
            "chunks_new": result.chunks_new,
            "chunks_deleted": result.chunks_deleted,
            "tokens_used": result.tokens_used,
            "estimated_cost_usd": round(result.estimated_cost_usd, 6),
            "file_sha256": run.file_sha256,
        }
        if result.error:
            details["error"] = result.error
        await self.audit.log(
            actor_type=ActorType.SYSTEM,
            action="ingestion.failed" if result.status == "failed" else "ingestion.completed",
            entity_type="ingestion_run",
            entity_id=run.id,
            details=details,
        )
        logger.info("ingest_finished", extra={"policy": policy.policy_code, **details})
