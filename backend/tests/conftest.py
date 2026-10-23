from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

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
