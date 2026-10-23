from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all models (Module 2 onwards). Alembic reads its metadata."""


def create_engine(database_url: str) -> AsyncEngine:
    # pool_pre_ping drops stale connections (e.g. after a Postgres restart) instead of erroring.
    return create_async_engine(database_url, pool_pre_ping=True)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one AsyncSession per request, closed afterwards."""
    async with request.app.state.session_factory() as session:
        yield session
