import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import admin, agent_runs, auth, claims, health, metrics, policies, rag
from app.auth.limits import client_ip, hit, limits_active, too_many
from app.core.config import Settings, get_settings, validate_for_production
from app.core.db import create_engine
from app.core.errors import error_response, register_exception_handlers
from app.core.llm import get_breaker, get_embeddings
from app.core.logging import request_id_ctx, setup_logging
from app.core.redis import create_redis
from app.observability.metrics import RequestTimer, bind_breaker, route_template
from app.observability.tracing import init_tracing, shutdown_tracing
from app.worker.queue import create_queue

logger = logging.getLogger(__name__)

# An API that returns JSON only: nothing may be framed, sniffed or embedded.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "same-site",
    "Cache-Control": "no-store",
}


async def warm_up_agents(app: FastAPI) -> None:
    """Create the checkpointer tables (setup()) at startup. Best effort: if the database is not
    reachable yet, the first agent request builds the runtime instead."""
    try:
        app.state.agent_runtime = await agent_runs.build_runtime(app)
    except Exception:  # noqa: BLE001
        logger.warning("agent_runtime_warmup_failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(settings.log_level)
    validate_for_production(settings)  # APP_ENV=production: no start with dev defaults
    init_tracing(settings)  # no-op without Langfuse keys
    bind_breaker(get_breaker(settings).state_code)
    app.state.engine = create_engine(settings.database_url)
    app.state.session_factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.redis = create_redis(settings.redis_url)
    app.state.embedder_factory = get_embeddings  # tests swap in a fake
    app.state.agent_runtime = None  # built on first use (see api/agent_runs.py)
    app.state.queue = await create_queue(settings) if settings.use_worker else None
    if settings.app_env != "test":
        await warm_up_agents(app)
    logger.info("app_started", extra={"env": settings.app_env})
    try:
        yield
    finally:
        if app.state.agent_runtime is not None:
            await app.state.agent_runtime.close()
        if app.state.queue is not None:
            await app.state.queue.aclose()
        await app.state.redis.aclose()
        await app.state.engine.dispose()
        shutdown_tracing()  # flushes pending traces
        logger.info("app_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="ClaimPilot", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = (request.headers.get("X-Request-ID") or uuid4().hex)[:64]
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        timer = RequestTimer(request.method)
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            if trace_id := getattr(request.state, "trace_id", None):
                response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            if request.url.path != "/metrics":
                timer.observe(route_template(request.scope), status_code)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_ctx.reset(token)

    # Added after the middleware above, so CORS is outermost and also covers error responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Trace-ID"],
    )

    @app.middleware("http")
    async def ip_rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Per-IP limit on everything but liveness and the scraper (also covers /auth/*)."""
        path = request.url.path
        if path in {"/health", "/health/ready", "/metrics"} or not limits_active(settings):
            return await call_next(request)
        wait = await hit(
            request.app.state.redis, "ip", client_ip(request), settings.rate_limit_ip_per_minute
        )
        if wait is not None:
            logger.warning("rate_limited", extra={"scope": "ip", "retry_after": wait})
            error = too_many(wait, "ip")
            return error_response(
                request, 429, "rate_limited", str(error.detail), headers=error.headers
            )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        docs = request.url.path in {"/docs", "/redoc", "/openapi.json"}
        for name, value in SECURITY_HEADERS.items():
            if not (docs and name == "Content-Security-Policy"):  # Swagger UI needs its scripts
                response.headers.setdefault(name, value)
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(policies.router)
    app.include_router(rag.router)
    app.include_router(claims.router)
    app.include_router(agent_runs.router)
    return app


app = create_app()
