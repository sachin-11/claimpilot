import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import health
from app.core.config import Settings, get_settings
from app.core.db import create_engine
from app.core.errors import register_exception_handlers
from app.core.logging import request_id_ctx, setup_logging
from app.core.redis import create_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    setup_logging(settings.log_level)
    app.state.engine = create_engine(settings.database_url)
    app.state.session_factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.redis = create_redis(settings.redis_url)
    logger.info("app_started", extra={"env": settings.app_env})
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.engine.dispose()
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
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    return app


app = create_app()
