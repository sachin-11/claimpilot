import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Test settings: ignore any local .env and use dummy keys (no real API calls)."""
    return Settings(
        _env_file=None,
        app_env="test",
        openai_api_key="sk-test",
        groq_api_key="gsk-test",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # Entering the context runs the lifespan. Engine and Redis clients are lazy, so no
    # database or redis server is needed.
    with TestClient(create_app(settings)) as c:
        yield c


# ---------------------------------------------------------------------------------------
# Database fixtures: a separate test database, migrated once per session, with every test
# running inside a transaction that is rolled back.
# ---------------------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]


def alembic_config(database_url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.attributes["database_url"] = database_url
    return cfg


def database_urls() -> tuple[str, str, str]:
    """(admin url, test url, scratch url for the migration test), all on the same server."""
    base = Settings()
    main = make_url(base.database_url)
    if base.test_database_url:
        test = make_url(base.test_database_url)
    else:
        test = main.set(database=f"{main.database}_test")
    scratch = test.set(database=f"{test.database}_migrations")
    render = lambda u: u.render_as_string(hide_password=False)  # noqa: E731
    return render(main), render(test), render(scratch)


async def recreate_database(admin_url: str, name: str, *, drop_first: bool) -> None:
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with engine.connect() as conn:
        exists = (
            await conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name})
        ).scalar()
        if exists and drop_first:
            await conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
            exists = None
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
    await engine.dispose()


@pytest.fixture(scope="session")
def migration_urls() -> tuple[str, str]:
    """(admin url, scratch url). The scratch database is rebuilt for the migration test."""
    admin, _, scratch = database_urls()
    return admin, scratch


@pytest.fixture(scope="session")
def test_database_url() -> str:
    admin, test, _ = database_urls()
    asyncio.run(recreate_database(admin, make_url(test).database, drop_first=False))
    command.upgrade(alembic_config(test), "head")
    return test


@pytest.fixture
async def session(test_database_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    async with engine.connect() as conn:
        outer = await conn.begin()
        # Commits inside the code under test only release a SAVEPOINT; the rollback below
        # discards everything.
        db = AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield db
        finally:
            await db.close()
            await outer.rollback()
    await engine.dispose()
