"""arq connection settings and the helpers the API uses to enqueue work."""

from typing import Any

from arq.connections import ArqRedis, RedisSettings, create_pool

from app.core.config import Settings

QUEUE_NAME = "claimpilot:jobs"  # keeps our jobs apart from any other arq user of the same Redis
HEALTH_KEY = "claimpilot:worker:health"
JOB_RUN = "run_agent"
JOB_EXTRACT = "extract_claim"
JOB_INGEST = "ingest_policies"


def redis_settings(settings: Settings) -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def create_queue(settings: Settings) -> ArqRedis:
    return await create_pool(redis_settings(settings), default_queue_name=QUEUE_NAME)


async def enqueue_extraction(queue: Any, claim_id: str, prefer: str) -> bool:
    """False when an extraction of this claim is already queued or running."""
    locked = await queue.set(f"lock:extract:{claim_id}", "1", nx=True, ex=900)
    if not locked:
        return False
    job = await queue.enqueue_job(JOB_EXTRACT, claim_id, prefer)
    if job is None:  # pragma: no cover - arq generates a unique id
        await queue.delete(f"lock:extract:{claim_id}")
    return job is not None


async def enqueue_ingestion(queue: Any, runs: list[tuple[str, str]], force: bool) -> None:
    await queue.enqueue_job(JOB_INGEST, [list(r) for r in runs], force)
