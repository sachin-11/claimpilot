import asyncio

from alembic import command
from conftest import alembic_config, recreate_database
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

import app.models  # noqa: F401
from app.core.db import Base

EXPECTED_TABLES = set(Base.metadata.tables)


async def _tables(url: str) -> set[str]:
    engine = create_async_engine(url, poolclass=NullPool)
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        names = {r[0] for r in rows}
    await engine.dispose()
    return names


async def test_upgrade_and_downgrade_cleanly(migration_urls: tuple[str, str]) -> None:
    admin_url, scratch_url = migration_urls
    await recreate_database(admin_url, make_url(scratch_url).database, drop_first=True)
    cfg = alembic_config(scratch_url)

    # Alembic's async env runs its own event loop, so call it from a worker thread.
    await asyncio.to_thread(command.upgrade, cfg, "head")
    assert EXPECTED_TABLES <= await _tables(scratch_url)

    await asyncio.to_thread(command.downgrade, cfg, "base")
    assert await _tables(scratch_url) <= {"alembic_version"}

    # ...and it can go up again after a full downgrade (enum types and trigger were removed).
    await asyncio.to_thread(command.upgrade, cfg, "head")
    assert EXPECTED_TABLES <= await _tables(scratch_url)
