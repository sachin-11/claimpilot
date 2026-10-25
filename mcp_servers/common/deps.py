"""What a server needs at runtime, injectable so tests can swap the database and RAG service."""

import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.db import create_engine
from app.core.redis import create_redis
from mcp_servers.common.settings import McpSettings, get_mcp_settings

if TYPE_CHECKING:
    from app.rag.service import RetrievalService

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass
class ServerDeps:
    session_factory: SessionFactory
    settings: McpSettings = field(default_factory=get_mcp_settings)
    # Built on first use so servers that never search do not need LLM keys or Redis.
    rag_factory: "Callable[[], RetrievalService] | None" = None
    _rag: "RetrievalService | None" = None

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        return self.session_factory()

    def rag(self) -> "RetrievalService":
        if self._rag is None:
            if self.rag_factory is None:
                raise RuntimeError("no RAG service configured")
            self._rag = self.rag_factory()
        return self._rag


def default_deps() -> ServerDeps:
    """Production wiring: own engine / Redis from the backend settings."""
    app_settings = get_settings()
    engine = create_engine(app_settings.database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    redis: Redis = create_redis(app_settings.redis_url)

    def build_rag() -> "RetrievalService":
        from app.rag.service import build_rag_service

        return build_rag_service(factory, redis)

    @contextlib.asynccontextmanager
    async def session() -> AsyncIterator[AsyncSession]:
        async with factory() as s:
            yield s

    return ServerDeps(session_factory=session, rag_factory=build_rag)
