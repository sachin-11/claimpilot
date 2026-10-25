"""The jobs the worker runs. Every one is safe to run twice: a retried or re-delivered job checks
what already happened before doing it again."""

import logging
from typing import Any

import httpx
from arq import Retry
from sqlalchemy.exc import DBAPIError, InterfaceError

from app.core.llm import RETRYABLE_ERRORS
from app.observability.metrics import record_job
from app.worker.bus import RunBus
from app.worker.queue import JOB_EXTRACT, JOB_INGEST, JOB_RUN

logger = logging.getLogger(__name__)

# Network and database hiccups are worth another try; a bug is not.
TRANSIENT = (
    ConnectionError,
    TimeoutError,
    httpx.TransportError,
    InterfaceError,
    DBAPIError,
    *RETRYABLE_ERRORS,
)
RETRY_BASE_SECONDS = 5.0  # first retry after this, then x2 each time (tests shorten it)


def backoff(job_try: int) -> float:
    return RETRY_BASE_SECONDS * (2 ** max(job_try - 1, 0))


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, DBAPIError):
        return bool(exc.connection_invalidated)
    return isinstance(exc, TRANSIENT)


async def run_agent(
    ctx: dict[str, Any], kind: str, run_id: str, payload: dict[str, Any], meta: dict[str, Any]
) -> str:
    """One phase of an agent run (`start` until the human approval, `resume` after it)."""
    bus: RunBus = ctx["bus"]
    epoch = 0 if kind == "start" else 1
    try:
        await bus.set_meta(run_id, state="running", worker=ctx.get("worker_id", ""))

        async def sink(event: str, data: dict[str, Any]) -> None:
            await bus.publish(run_id, epoch, event, data)

        status = await ctx["runtime"].execute(kind, run_id, payload, meta, sink=sink)
    except Exception as exc:  # noqa: BLE001
        if is_transient(exc) and ctx.get("job_try", 1) < ctx["max_tries"]:
            delay = backoff(ctx.get("job_try", 1))
            logger.warning(
                "run_agent_retry", extra={"run_id": run_id, "in_s": delay, "error": repr(exc)}
            )
            await bus.set_meta(run_id, state="queued")
            record_job(JOB_RUN, "retry")
            raise Retry(defer=delay) from exc
        record_job(JOB_RUN, "error")
        await bus.set_meta(run_id, state="idle", status="failed")
        raise
    await bus.set_meta(run_id, state="idle", status=status)
    record_job(JOB_RUN, "ok")
    return status


async def extract_claim(ctx: dict[str, Any], claim_id: str, prefer: str) -> str:
    """Module 6 intake for one claim (`IntakeService.process` records its own failures)."""
    from uuid import UUID  # noqa: PLC0415

    try:
        result = await ctx["intake"].process(UUID(claim_id), prefer=prefer)
        record_job(JOB_EXTRACT, "ok")
        return str(getattr(result, "status", "done"))
    except Exception as exc:  # noqa: BLE001
        if is_transient(exc) and ctx.get("job_try", 1) < ctx["max_tries"]:
            record_job(JOB_EXTRACT, "retry")
            raise Retry(defer=backoff(ctx.get("job_try", 1))) from exc
        record_job(JOB_EXTRACT, "error")
        raise
    finally:
        await ctx["redis"].delete(f"lock:extract:{claim_id}")  # lets the next extraction be queued


async def ingest_policies(ctx: dict[str, Any], runs: list[list[str]], force: bool) -> int:
    """Policy ingestion (the run rows are created by the API, which the client polls)."""
    from uuid import UUID  # noqa: PLC0415

    from app.api.policies import run_ingestion_task  # noqa: PLC0415

    try:
        await run_ingestion_task(
            ctx["session_factory"],
            ctx["embedder_factory"],
            ctx["ingestion_settings"],
            [(UUID(run_id), code) for run_id, code in runs],
            force,
            ctx.get("on_policy_changed"),
        )
    except Exception as exc:  # noqa: BLE001
        if is_transient(exc) and ctx.get("job_try", 1) < ctx["max_tries"]:
            record_job(JOB_INGEST, "retry")
            raise Retry(defer=backoff(ctx.get("job_try", 1))) from exc
        record_job(JOB_INGEST, "error")
        raise
    record_job(JOB_INGEST, "ok")
    return len(runs)
