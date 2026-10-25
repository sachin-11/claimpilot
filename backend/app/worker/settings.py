"""`arq app.worker.settings.WorkerSettings`: the worker process.

Startup builds what the jobs need once (database, agent runtime with the Postgres checkpointer, MCP
tool provider, intake service, ingestion factories). On SIGTERM arq stops taking jobs and waits
`worker_shutdown_wait_seconds` for the running ones; a job still running after that is cancelled and
re-queued, and it continues from its last checkpoint when a worker picks it up again."""

import logging
import os
import socket
from typing import Any

from prometheus_client import start_http_server
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.deps import AgentDeps
from app.agents.runner import AgentRuntime
from app.agents.tools import McpToolProvider
from app.core.config import get_settings, validate_for_production
from app.core.db import create_engine
from app.core.llm import get_breaker, get_chat_model, get_embeddings
from app.core.logging import setup_logging
from app.ingestion.config import get_ingestion_settings
from app.intake.service import IntakeService
from app.observability.metrics import bind_breaker
from app.observability.tracing import flush_tracing, init_tracing, shutdown_tracing
from app.rag.cache import SemanticCache
from app.rag.config import get_rag_settings
from app.worker.bus import RunBus
from app.worker.queue import HEALTH_KEY, QUEUE_NAME, redis_settings
from app.worker.tasks import extract_claim, ingest_policies, run_agent

logger = logging.getLogger(__name__)
settings = get_settings()


def checkpoint_conninfo() -> str:
    from app.agents.config import get_agent_settings  # noqa: PLC0415

    explicit = get_agent_settings().checkpoint_database_url
    return (explicit or settings.database_url).replace("postgresql+asyncpg://", "postgresql://", 1)


async def on_startup(ctx: dict[str, Any]) -> None:
    setup_logging(settings.log_level)
    validate_for_production(settings)
    init_tracing(settings)
    bind_breaker(get_breaker(settings).state_code)
    engine = create_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    intake = IntakeService(
        session_factory, lambda: get_chat_model(temperature=0), data_dir=settings.data_dir
    )
    deps = AgentDeps(
        tools=McpToolProvider(),
        chat_model=lambda: get_chat_model(temperature=0),
        session_factory=session_factory,
        data_dir=settings.data_dir,
        intake=intake,
    )
    rag_cache = SemanticCache(ctx["redis"], get_rag_settings())
    ctx.update(
        engine=engine,
        session_factory=session_factory,
        intake=intake,
        bus=RunBus(ctx["redis"]),
        runtime=await AgentRuntime.create(deps, checkpoint_conninfo()),
        embedder_factory=get_embeddings,
        ingestion_settings=get_ingestion_settings(),
        on_policy_changed=rag_cache.invalidate_policy,
        max_tries=settings.worker_max_tries,
        worker_id=f"{socket.gethostname()}:{os.getpid()}",
    )
    start_http_server(settings.worker_metrics_port)  # this process' own /metrics
    logger.info("worker_started", extra={"worker": ctx["worker_id"], "queue": QUEUE_NAME})


async def on_shutdown(ctx: dict[str, Any]) -> None:
    logger.info("worker_stopping")
    await ctx["runtime"].close()
    await ctx["engine"].dispose()
    flush_tracing()
    shutdown_tracing()


class WorkerSettings:
    functions = [run_agent, extract_claim, ingest_policies]
    queue_name = QUEUE_NAME
    redis_settings = redis_settings(settings)
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = settings.worker_max_jobs
    job_timeout = settings.worker_job_timeout_seconds
    max_tries = settings.worker_max_tries
    retry_jobs = True  # a job lost with a dead worker is picked up again
    keep_result = 3600
    health_check_interval = 5
    health_check_key = HEALTH_KEY
    job_completion_wait = settings.worker_shutdown_wait_seconds
    allow_abort_jobs = False
    handle_signals = True
