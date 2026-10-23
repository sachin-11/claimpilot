import pytest
from fastapi.testclient import TestClient


def test_health_is_ok_and_sets_request_id(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers["X-Request-ID"]


def test_request_id_is_echoed(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_ready_ok_when_dependencies_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(_: object) -> None:
        return None

    monkeypatch.setattr("app.api.health.check_postgres", ok)
    monkeypatch.setattr("app.api.health.check_redis", ok)

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready",
        "dependencies": {"postgres": {"status": "ok"}, "redis": {"status": "ok"}},
    }


def test_ready_503_when_a_dependency_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def ok(_: object) -> None:
        return None

    async def pgvector_missing(_: object) -> None:
        raise RuntimeError("pgvector extension is not installed")

    monkeypatch.setattr("app.api.health.check_postgres", pgvector_missing)
    monkeypatch.setattr("app.api.health.check_redis", ok)

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    deps = resp.json()["dependencies"]
    assert deps["postgres"]["status"] == "error"
    assert "pgvector" in deps["postgres"]["detail"]
    assert deps["redis"] == {"status": "ok"}


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    resp = client.get("/nope")
    assert resp.status_code == 404
    body = resp.json()["error"]
    assert body["code"] == "http_error"
    assert body["request_id"] == resp.headers["X-Request-ID"]
