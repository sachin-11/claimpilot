import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import FakeEmbedder, ingestion_settings, make_policy_pdf
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.main import create_app
from app.repositories import PolicyRepository


async def _run_sql(url: str, statement: str | None = None, *, create_policy: bool = False) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession

    engine = create_async_engine(url, poolclass=NullPool)
    async with AsyncSession(engine) as session:
        if create_policy:
            await PolicyRepository(session).upsert_from_rules(
                {"policy_id": "POL-API", "name": "API Policy", "source_pdf": "policy.pdf"}
            )
        if statement:
            await session.execute(text(statement))
        await session.commit()
    await engine.dispose()


@pytest.fixture
def api(test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App wired to the test database (data is committed, so clean up afterwards)."""
    cfg = ingestion_settings(tmp_path, chunk_max_tokens=200, chunk_min_tokens=20)
    monkeypatch.setattr("app.api.policies.get_ingestion_settings", lambda: cfg)
    make_policy_pdf(tmp_path / "policy.pdf")
    asyncio.run(_run_sql(test_database_url, create_policy=True))
    settings = Settings(
        _env_file=None, app_env="test", database_url=test_database_url, openai_api_key="sk-test"
    )
    app = create_app(settings)
    embedder = FakeEmbedder()
    with TestClient(app) as client:
        app.state.embedder_factory = lambda: embedder
        client.embedder = embedder  # type: ignore[attr-defined]
        yield client
    asyncio.run(
        _run_sql(test_database_url, "TRUNCATE policies, ingestion_runs, audit_logs CASCADE")
    )


def test_ingest_endpoint_flow(api: TestClient) -> None:
    started = api.post("/policies/ingest", json={"policy_code": "POL-API"})
    assert started.status_code == 202
    body = started.json()
    assert body["run_id"] == body["runs"][0]["run_id"]

    # TestClient runs the background task before returning, so the run is finished here.
    run = api.get(f"/policies/ingest/{body['run_id']}").json()
    assert run["status"] == "success" and run["chunks_total"] > 0 and run["error"] is None
    assert api.embedder.texts_embedded == run["chunks_total"]  # type: ignore[attr-defined]

    listing = api.get("/policies").json()
    api_policy = next(p for p in listing if p["policy_code"] == "POL-API")
    assert api_policy["chunk_count"] == run["chunks_total"]
    assert api_policy["last_run"]["status"] == "success"

    chunks = api.get("/policies/POL-API/chunks", params={"limit": 2, "offset": 1}).json()
    assert len(chunks) == 2 and chunks[0]["chunk_index"] == 1
    assert "embedding" not in chunks[0] and "embed_text" not in chunks[0]["meta"]
    assert chunks[0]["meta"]["policy_code"] == "POL-API"

    # second request: file unchanged -> success without embedding anything new
    again = api.post("/policies/ingest", json={"all": True}).json()
    assert api.get(f"/policies/ingest/{again['run_id']}").json()["chunks_new"] == 0
    assert api.embedder.texts_embedded == run["chunks_total"]  # type: ignore[attr-defined]


def test_ingest_endpoint_validation(api: TestClient) -> None:
    assert api.post("/policies/ingest", json={}).status_code == 422
    assert api.post("/policies/ingest", json={"policy_code": "X", "all": True}).status_code == 422
    assert api.post("/policies/ingest", json={"policy_code": "NOPE"}).status_code == 404
    assert api.get("/policies/NOPE/chunks").status_code == 404
    assert api.get("/policies/ingest/00000000-0000-0000-0000-000000000000").status_code == 404
