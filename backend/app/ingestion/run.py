"""CLI: python -m app.ingestion.run --policy POL-AS | --all  [--force] [--dry-run]"""

import argparse
import asyncio
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings, has_real_key
from app.core.db import create_engine
from app.core.llm import get_embeddings
from app.core.logging import setup_logging
from app.core.redis import create_redis
from app.ingestion.config import get_ingestion_settings
from app.ingestion.pipeline import IngestionPipeline, IngestResult
from app.rag.cache import SemanticCache
from app.rag.config import get_rag_settings
from app.repositories import PolicyRepository


def format_result(result: IngestResult) -> str:
    lines = [f"{result.policy_code}: {result.status.upper()}"]
    if result.error:
        lines.append(f"  error: {result.error}")
    if stats := result.stats:
        lines += [
            f"  chunks: {stats.chunks}  tokens min/avg/max: "
            f"{stats.min_tokens}/{stats.avg_tokens:.0f}/{stats.max_tokens}",
            f"  chunks with tables: {stats.with_tables}",
            f"  pages skipped as scanned: {stats.skipped_pages or 'none'}",
        ]
    if result.status in {"success", "skipped"}:
        lines += [
            f"  total {result.chunks_total}  new {result.chunks_new}  "
            f"updated {result.chunks_updated}  deleted {result.chunks_deleted}",
            f"  embedded {result.chunks_embedded}  tokens {result.tokens_used}  "
            f"est. cost ${result.estimated_cost_usd:.5f}",
        ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion.run", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--policy", metavar="CODE", help="policy_code, e.g. POL-AS")
    target.add_argument("--all", action="store_true", help="every active policy")
    parser.add_argument("--force", action="store_true", help="ignore the unchanged-file check")
    parser.add_argument("--dry-run", action="store_true", help="parse + chunk only, no DB, no API")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)
    if not args.dry_run and not has_real_key(settings.openai_api_key):
        sys.stderr.write("OPENAI_API_KEY is not set in .env (needed to embed chunks).\n")
        return 2

    engine = create_engine(settings.database_url)
    redis = create_redis(settings.redis_url)
    cache = SemanticCache(redis, get_rag_settings())
    exit_code = 0
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            repo = PolicyRepository(session)
            if args.all:
                policies = await repo.list_active()
            else:
                policy = await repo.get_by_code(args.policy)
                if policy is None:
                    sys.stderr.write(f"Unknown policy {args.policy!r}. Run `make seed` first?\n")
                    return 2
                policies = [policy]

            pipeline = IngestionPipeline(
                session,
                None if args.dry_run else get_embeddings(),
                get_ingestion_settings(),
                on_policy_changed=cache.invalidate_policy,
            )
            for policy in policies:
                result = await pipeline.ingest(policy, force=args.force, dry_run=args.dry_run)
                if not args.dry_run:
                    await session.commit()  # one transaction per policy
                    await pipeline.after_commit(result)
                sys.stdout.write(format_result(result) + "\n")
                if result.status == "failed":
                    exit_code = 1
    finally:
        await redis.aclose()
        await engine.dispose()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
