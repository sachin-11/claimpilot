"""Getting back the runs of a worker that died.

arq keeps a job queued until it finishes and marks it "in progress" for `job_timeout + 10` seconds.
A worker killed mid-job (OOM, SIGKILL, a node going away) therefore leaves the job locked for about
20 minutes here. Each worker refreshes a short-lived `alive` key; a run whose worker has no such
key any more is an orphan: its in-progress lock is dropped, so a live worker takes the job again
and `AgentRuntime.execute` continues it from the last checkpoint."""

import logging
from typing import Any

from arq.constants import in_progress_key_prefix

from app.worker.bus import RunBus
from app.worker.queue import QUEUE_NAME

logger = logging.getLogger(__name__)

ALIVE_PREFIX = "claimpilot:worker:alive:"
ALIVE_TTL_SECONDS = 60  # refreshed every 5 s: a long blocking call must not look like a death


async def beat(ctx: dict[str, Any]) -> None:
    await ctx["redis"].set(ALIVE_PREFIX + ctx["worker_id"], "1", ex=ALIVE_TTL_SECONDS)


async def retire(ctx: dict[str, Any]) -> None:
    """A worker that stops cleanly says so, so its unfinished jobs are picked up at once."""
    await ctx["redis"].delete(ALIVE_PREFIX + ctx["worker_id"])


async def recover_orphans(ctx: dict[str, Any]) -> int:
    redis = ctx["redis"]
    bus = RunBus(redis)
    recovered = 0
    async for raw in redis.scan_iter(match="run:*:meta", count=200):
        key = raw.decode() if isinstance(raw, bytes) else raw
        run_id = key.split(":")[1]
        meta = await bus.get_meta(run_id)
        worker = meta.get("worker", "")
        if meta.get("state") != "running" or not worker:
            continue
        if await redis.exists(ALIVE_PREFIX + worker):
            continue  # its worker is alive
        job_id = f"{run_id}:{int(meta.get('epoch') or 0)}"
        if await redis.zscore(ctx.get("queue_name") or QUEUE_NAME, job_id) is None:
            # not queued any more: the job ended without marking the run, nothing to take again
            await bus.set_meta(run_id, state="idle", status="stalled")
            logger.warning("orphan_run_without_job", extra={"run_id": run_id})
            continue
        await redis.delete(in_progress_key_prefix + job_id)
        await bus.set_meta(run_id, state="queued")
        recovered += 1
        logger.warning("orphan_run_requeued", extra={"run_id": run_id, "dead_worker": worker})
    return recovered
