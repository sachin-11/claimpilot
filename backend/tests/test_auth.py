"""Module 10, part B: login, lockout, refresh rotation, RBAC, rate limits, the budget guard and the
production checks. A rolled-back database session, a fake Redis and real tokens: no LLM, no network."""

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import httpx
import jwt
import pytest
from auth_helpers import bearer
from fake_redis import FakeRedis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.agent_runs import get_runtime
from app.auth.security import create_access_token, decode_access_token, hash_refresh_token
from app.core.config import (
    InsecureConfigurationError,
    Settings,
    production_problems,
    validate_for_production,
)
from app.core.db import get_session
from app.main import create_app
from app.models import AuditLog, RefreshToken, User, UserRole
from app.observability.usage import spend_key, today

PASSWORD = "correct horse battery"
SETTINGS = dict(_env_file=None, app_env="test", jwt_secret="x" * 40)


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{**SETTINGS, **overrides})


async def make_user(
    session: AsyncSession, email: str, role: UserRole = UserRole.OFFICER, active: bool = True
) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0].title(),
        role=role,
        password_hash=bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode(),
        is_active=active,
    )
    session.add(user)
    await session.flush()
    return user


class StubRuntime:
    """Stands in for the agent runtime: the handlers behind the auth checks never need a graph."""

    async def start_run(self, claim_id: Any, user_id: str | None = None) -> str:
        return "run-1"

    async def get_run_state(self, run_id: str) -> dict[str, Any]:
        from app.agents.runner import RunNotFoundError

        raise RunNotFoundError(run_id)


@contextlib.asynccontextmanager
async def client_for(
    session: AsyncSession,
    settings: Settings | None = None,
    redis: FakeRedis | None = None,
    ip: str = "10.0.0.1",
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeRedis, Settings]]:
    settings = settings or make_settings()
    redis = redis or FakeRedis()
    app = create_app(settings)

    async def shared_session() -> AsyncIterator[AsyncSession]:
        yield session

    async def runtime() -> StubRuntime:
        return StubRuntime()

    app.dependency_overrides[get_session] = shared_session
    app.dependency_overrides[get_runtime] = runtime
    app.state.redis = redis
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False, client=(ip, 5000)),
        base_url="http://t",
    ) as http:
        yield http, redis, settings


async def login(http: httpx.AsyncClient, email: str, password: str = PASSWORD) -> httpx.Response:
    return await http.post("/auth/login", json={"email": email, "password": password})


# ------------------------------------------------------------------ login / refresh / logout


async def test_login_returns_tokens_and_me_works(session: AsyncSession) -> None:
    user = await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, settings):
        resp = await login(http, "asha@example.com")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["token_type"] == "bearer" and body["user"]["role"] == "officer"
        claims = decode_access_token(settings, body["access_token"])
        assert claims.user_id == user.id and claims.role == UserRole.OFFICER
        assert (claims.expires_at - datetime.now(UTC)) <= timedelta(minutes=15, seconds=5)
        me = await http.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
        assert me.status_code == 200 and me.json()["email"] == "asha@example.com"
    stored = (await session.execute(select(RefreshToken))).scalar_one()
    assert stored.token_hash == hash_refresh_token(body["refresh_token"])  # only the hash is stored
    assert body["refresh_token"] not in {stored.token_hash}
    assert (stored.expires_at - datetime.now(UTC)) > timedelta(days=6)
    audit = (await session.execute(select(AuditLog.action))).scalars().all()
    assert "login" in audit


async def test_wrong_password_unknown_user_and_disabled_account_look_the_same(
    session: AsyncSession,
) -> None:
    await make_user(session, "asha@example.com")
    await make_user(session, "gone@example.com", active=False)
    async with client_for(session) as (http, _, _):
        answers = [
            await login(http, "asha@example.com", "wrong password"),
            await login(http, "nobody@example.com"),
            await login(http, "gone@example.com"),
        ]
    assert {r.status_code for r in answers} == {401}
    assert (
        len({r.json()["error"]["message"] if "error" in r.json() else r.text for r in answers}) == 1
    )
    actions = (await session.execute(select(AuditLog.action))).scalars().all()
    assert actions.count("login_failed") == 3


async def test_five_failures_lock_the_email_and_ip_for_the_window(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, redis, settings):
        for _ in range(settings.login_max_failures):
            assert (await login(http, "asha@example.com", "nope")).status_code == 401
        locked = await login(http, "asha@example.com")  # even the right password
        assert locked.status_code == 429
        assert 0 < int(locked.headers["Retry-After"]) <= settings.login_window_minutes * 60
        other_email = await login(http, "nobody@example.com", "nope")
        assert other_email.status_code == 401  # a different e-mail is not locked
        redis.advance(settings.login_window_minutes * 60 + 1)
        assert (await login(http, "asha@example.com")).status_code == 200  # the window passed

    async with client_for(session, ip="10.9.9.9") as (http, redis, _):
        redis.values.update({})  # a fresh Redis: another address has its own counter
        assert (await login(http, "asha@example.com")).status_code == 200


async def test_lockout_is_per_ip_not_global(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    redis = FakeRedis()
    async with client_for(session, redis=redis, ip="10.0.0.1") as (attacker, _, settings):
        for _ in range(settings.login_max_failures):
            await login(attacker, "asha@example.com", "nope")
        assert (await login(attacker, "asha@example.com")).status_code == 429
    async with client_for(session, redis=redis, ip="10.0.0.2") as (elsewhere, _, _):
        assert (await login(elsewhere, "asha@example.com")).status_code == 200


async def test_refresh_rotates_and_a_reused_token_revokes_everything(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, _):
        first = (await login(http, "asha@example.com")).json()
        second = (
            await http.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
        ).json()
        assert second["refresh_token"] != first["refresh_token"]
        assert second["access_token"] != first["access_token"]
        # the old token again: theft. 401, and the new one dies too
        reuse = await http.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
        assert reuse.status_code == 401
        after = await http.post("/auth/refresh", json={"refresh_token": second["refresh_token"]})
        assert after.status_code == 401
        assert (
            await http.post("/auth/refresh", json={"refresh_token": "x" * 40})
        ).status_code == 401
    rows = (await session.execute(select(RefreshToken))).scalars().all()
    assert len(rows) == 2 and all(r.revoked_at is not None for r in rows)
    assert "refresh_token_reuse" in (await session.execute(select(AuditLog.action))).scalars().all()


async def test_expired_refresh_token_is_rejected(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, _):
        token = (await login(http, "asha@example.com")).json()["refresh_token"]
        row = (await session.execute(select(RefreshToken))).scalar_one()
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.flush()
        assert (await http.post("/auth/refresh", json={"refresh_token": token})).status_code == 401


async def test_logout_revokes_the_refresh_token_and_is_idempotent(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, _):
        tokens = (await login(http, "asha@example.com")).json()
        assert (
            await http.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
        ).status_code == 204
        assert (
            await http.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
        ).status_code == 204
        assert (
            await http.post("/auth/logout", json={"refresh_token": "y" * 40})
        ).status_code == 204
        assert (
            await http.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        ).status_code == 401


async def test_logout_everywhere_revokes_every_session(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, _):
        a = (await login(http, "asha@example.com")).json()
        b = (await login(http, "asha@example.com")).json()
        await http.post(
            "/auth/logout", json={"refresh_token": a["refresh_token"], "everywhere": True}
        )
        assert (
            await http.post("/auth/refresh", json={"refresh_token": b["refresh_token"]})
        ).status_code == 401


# ------------------------------------------------------------------ access tokens


async def test_bad_access_tokens_are_rejected(session: AsyncSession) -> None:
    user = await make_user(session, "asha@example.com")
    settings = make_settings()
    past = datetime.now(UTC) - timedelta(hours=1)
    expired, _ = create_access_token(
        settings, user_id=user.id, role=user.role, email=user.email, now=past
    )
    other_secret, _ = create_access_token(
        make_settings(jwt_secret="y" * 40), user_id=user.id, role=user.role, email=user.email
    )
    unsigned = jwt.encode(
        {"sub": str(user.id), "role": "admin", "type": "access"}, None, algorithm="none"
    )
    refresh_as_access = jwt.encode(
        {"sub": str(user.id), "role": "officer", "type": "refresh", "iat": 1, "exp": 4_000_000_000},
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    async with client_for(session, settings) as (http, _, _):
        assert (await http.get("/auth/me")).status_code == 401  # no token
        for bad in (expired, other_secret, unsigned, refresh_as_access, "not-a-token"):
            resp = await http.get("/auth/me", headers={"Authorization": f"Bearer {bad}"})
            assert resp.status_code == 401, bad[:30]
            assert resp.headers["WWW-Authenticate"] == "Bearer"
        ok = await http.get("/auth/me", headers=bearer(settings, user))
        assert ok.status_code == 200


async def test_a_disabled_account_and_a_demotion_apply_at_once(session: AsyncSession) -> None:
    admin = await make_user(session, "boss@example.com", UserRole.ADMIN)
    settings = make_settings()
    headers = bearer(settings, admin)  # a token minted while the user was an admin
    async with client_for(session, settings) as (http, _, _):
        assert (await http.get("/admin/users", headers=headers)).status_code == 200
        admin.role = UserRole.OFFICER  # demoted after the token was issued
        await session.flush()
        assert (await http.get("/admin/users", headers=headers)).status_code == 403
        admin.is_active = False
        await session.flush()
        assert (await http.get("/auth/me", headers=headers)).status_code == 401


# ------------------------------------------------------------------ RBAC matrix

CLAIM = "00000000-0000-0000-0000-000000000001"
RUN = f"/claims/{CLAIM}/runs/r1"
# (method, path, json body, files, who may call: "officer" = officers and admins, "admin" = admins only)
MATRIX = [
    ("POST", "/policies/search", {"query": "waiting period"}, "officer"),
    ("POST", "/policies/query", {"query": "waiting period"}, "officer"),
    ("POST", "/policies/query/stream", {"query": "waiting period"}, "officer"),
    ("GET", "/policies", None, "officer"),
    ("GET", "/policies/POL-AS/chunks", None, "officer"),
    ("POST", "/policies/ingest", {"policy_code": "POL-AS"}, "admin"),
    ("GET", "/policies/ingest/00000000-0000-0000-0000-000000000009", None, "admin"),
    ("GET", f"/claims/{CLAIM}/extraction", None, "officer"),
    ("POST", f"/claims/{CLAIM}/extract", None, "officer"),
    ("POST", f"/claims/{CLAIM}/process", None, "officer"),
    ("GET", RUN, None, "officer"),
    ("GET", f"{RUN}/stream", None, "officer"),
    ("POST", f"{RUN}/approve", {"action": "approved"}, "officer"),
    ("GET", "/admin/audit-logs", None, "admin"),
    ("GET", "/admin/usage", None, "admin"),
    ("GET", "/admin/users", None, "admin"),
    (
        "POST",
        "/admin/users",
        {"email": "n@example.com", "full_name": "N", "password": "x" * 14},
        "admin",
    ),
    ("GET", "/metrics", None, "admin"),
    ("GET", "/auth/me", None, "officer"),
]
OPEN = [("GET", "/health"), ("GET", "/health/ready")]


@pytest.mark.parametrize(("method", "path", "payload", "needs"), MATRIX)
async def test_rbac_matrix(
    session: AsyncSession, method: str, path: str, payload: Any, needs: str
) -> None:
    officer = await make_user(session, "officer@example.com", UserRole.OFFICER)
    admin = await make_user(session, "admin@example.com", UserRole.ADMIN)
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        anon = await http.request(method, path, json=payload)
        assert anon.status_code == 401, (path, anon.status_code)
        as_officer = await http.request(
            method, path, json=payload, headers=bearer(settings, officer)
        )
        as_admin = await http.request(method, path, json=payload, headers=bearer(settings, admin))
    if needs == "admin":
        assert as_officer.status_code == 403, (path, as_officer.status_code)
    else:
        assert as_officer.status_code not in (401, 403), (path, as_officer.status_code)
    assert as_admin.status_code not in (401, 403), (path, as_admin.status_code)


@pytest.mark.parametrize(("method", "path"), OPEN)
async def test_health_endpoints_stay_open(session: AsyncSession, method: str, path: str) -> None:
    async with client_for(session) as (http, _, _):
        resp = await http.request(method, path)
    assert resp.status_code != 401


async def test_upload_needs_a_token(session: AsyncSession) -> None:
    async with client_for(session) as (http, _, _):
        resp = await http.post(
            "/claims", data={"policy_number": "P-1"}, files=[("files", ("a.pdf", b"x"))]
        )
    assert resp.status_code == 401


def test_the_approve_body_has_no_reviewer_field() -> None:
    from app.api.agent_runs import ApproveRequest

    assert "reviewer_id" not in ApproveRequest.model_fields


async def test_metrics_scraper_token_works_without_a_user(session: AsyncSession) -> None:
    settings = make_settings(metrics_token="scrape-token-123456")
    async with client_for(session, settings) as (http, _, _):
        ok = await http.get("/metrics", headers={"Authorization": "Bearer scrape-token-123456"})
        bad = await http.get("/metrics", headers={"Authorization": "Bearer scrape-token-000000"})
    assert ok.status_code == 200 and "http_requests_total" in ok.text
    assert bad.status_code == 401


# ------------------------------------------------------------------ rate limits and the budget guard


async def test_rate_limit_returns_429_with_retry_after(session: AsyncSession) -> None:
    officer = await make_user(session, "officer@example.com")
    settings = make_settings(
        rate_limit_in_tests=True, rate_limit_user_per_minute=3, rate_limit_ip_per_minute=100
    )
    async with client_for(session, settings) as (http, redis, _):
        headers = bearer(settings, officer)
        codes = [(await http.get("/auth/me", headers=headers)).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        blocked = await http.get("/auth/me", headers=headers)
        assert 1 <= int(blocked.headers["Retry-After"]) <= 60
        redis.advance(61)  # a new window
        assert (await http.get("/auth/me", headers=headers)).status_code == 200


async def test_llm_endpoints_have_a_stricter_limit(session: AsyncSession) -> None:
    officer = await make_user(session, "officer@example.com")
    settings = make_settings(
        rate_limit_in_tests=True, rate_limit_llm_user_per_minute=2, daily_llm_budget_inr=0
    )
    async with client_for(session, settings) as (http, _, _):
        headers = bearer(settings, officer)
        codes = [
            (await http.post(f"/claims/{CLAIM}/process", headers=headers)).status_code
            for _ in range(4)
        ]
        # the first two reach the handler, the rest are stopped before it
        assert codes[:2] != [429, 429] and codes[2:] == [429, 429]
        assert (await http.get("/auth/me", headers=headers)).status_code == 200  # others unaffected


async def test_ip_limit_applies_before_login(session: AsyncSession) -> None:
    settings = make_settings(rate_limit_in_tests=True, rate_limit_ip_per_minute=3)
    async with client_for(session, settings) as (http, _, _):
        codes = [(await login(http, "x@example.com", "p")).status_code for _ in range(5)]
    assert codes[-1] == 429 and codes[0] == 401


async def test_health_is_never_rate_limited(session: AsyncSession) -> None:
    settings = make_settings(rate_limit_in_tests=True, rate_limit_ip_per_minute=1)
    async with client_for(session, settings) as (http, _, _):
        assert {(await http.get("/health")).status_code for _ in range(5)} == {200}


async def test_limiter_fails_open_when_redis_is_down(session: AsyncSession) -> None:
    officer = await make_user(session, "officer@example.com")

    class Broken:
        async def incr(self, *_: Any) -> int:
            raise ConnectionError("redis is down")

    settings = make_settings(rate_limit_in_tests=True, rate_limit_user_per_minute=1)
    async with client_for(session, settings, redis=Broken()) as (http, _, _):  # type: ignore[arg-type]
        codes = {
            (await http.get("/auth/me", headers=bearer(settings, officer))).status_code
            for _ in range(3)
        }
    assert codes == {200}


async def test_budget_guard_blocks_new_runs_above_the_limit(session: AsyncSession, caplog) -> None:
    officer = await make_user(session, "officer@example.com")
    admin = await make_user(session, "admin@example.com", UserRole.ADMIN)
    settings = make_settings(rate_limit_in_tests=True, daily_llm_budget_inr=100.0)
    redis = FakeRedis()
    async with client_for(session, settings, redis=redis) as (http, _, _):
        h_officer, h_admin = bearer(settings, officer), bearer(settings, admin)
        under = await http.post(f"/claims/{CLAIM}/process", headers=h_officer)
        assert under.status_code != 429  # nothing spent yet: reaches the handler
        redis.hashes[spend_key(today())] = {
            "inr": "100.5", "calls": "40", "tokens_in": "90000", "tokens_out": "9000",
            "inr:openai:gpt-4o-mini": "100.5",
        }  # fmt: skip
        with caplog.at_level("ERROR"):
            over = await http.post(f"/claims/{CLAIM}/process", headers=h_officer)
        assert over.status_code == 429 and "daily LLM budget" in over.json()["error"]["message"]
        assert int(over.headers["Retry-After"]) > 0
        assert any(r.message == "llm_budget_exceeded" for r in caplog.records)  # the alert line
        usage = (await http.get("/admin/usage", headers=h_admin)).json()
    assert usage["today"]["inr"] == 100.5 and usage["today"]["budget_inr"] == 100.0
    assert usage["today"]["remaining_inr"] == 0 and usage["today"]["by_model"] == {
        "openai:gpt-4o-mini": 100.5
    }
    assert usage["today"]["budget_used_percent"] == 100.5 and len(usage["history"]) == 7


async def test_budget_of_zero_switches_the_guard_off(session: AsyncSession) -> None:
    officer = await make_user(session, "officer@example.com")
    settings = make_settings(rate_limit_in_tests=True, daily_llm_budget_inr=0)
    redis = FakeRedis()
    redis.hashes[spend_key(today())] = {"inr": "99999"}
    async with client_for(session, settings, redis=redis) as (http, _, _):
        resp = await http.post(f"/claims/{CLAIM}/process", headers=bearer(settings, officer))
    assert resp.status_code != 429


# ------------------------------------------------------------------ admin


async def test_audit_log_filters(session: AsyncSession) -> None:
    admin = await make_user(session, "admin@example.com", UserRole.ADMIN)
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        await login(http, "admin@example.com")
        await login(http, "admin@example.com", "wrong password")
        h = bearer(settings, admin)
        everything = (await http.get("/admin/audit-logs", headers=h)).json()
        assert everything["total"] >= 2
        failed = (
            await http.get("/admin/audit-logs", params={"action": "login_failed"}, headers=h)
        ).json()
        assert failed["total"] == 1 and failed["items"][0]["details"]["ip"] == "10.0.0.1"
        mine = (
            await http.get("/admin/audit-logs", params={"actor_id": str(admin.id)}, headers=h)
        ).json()
        assert {i["action"] for i in mine["items"]} == {"login"}
        future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        none = (await http.get("/admin/audit-logs", params={"since": future}, headers=h)).json()
        assert none["total"] == 0
        paged = (await http.get("/admin/audit-logs", params={"limit": 1}, headers=h)).json()
        assert len(paged["items"]) == 1 and paged["total"] == everything["total"]


async def test_admin_manages_users_and_cannot_lock_themselves_out(session: AsyncSession) -> None:
    admin = await make_user(session, "admin@example.com", UserRole.ADMIN)
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        h = bearer(settings, admin)
        created = await http.post(
            "/admin/users",
            json={
                "email": "New.Officer@Example.com",
                "full_name": "New Officer",
                "password": PASSWORD,
            },
            headers=h,
        )
        assert created.status_code == 201 and created.json()["email"] == "new.officer@example.com"
        assert created.json()["role"] == "officer"
        again = await http.post(
            "/admin/users",
            json={"email": "new.officer@example.com", "full_name": "X", "password": PASSWORD},
            headers=h,
        )
        assert again.status_code == 409
        weak = await http.post(
            "/admin/users",
            json={"email": "w@example.com", "full_name": "W", "password": "short"},
            headers=h,
        )
        assert weak.status_code == 422
        uid = created.json()["id"]
        assert (
            await login(http, "new.officer@example.com")
        ).status_code == 200  # the password works
        off = await http.patch(f"/admin/users/{uid}", json={"is_active": False}, headers=h)
        assert off.status_code == 200 and off.json()["is_active"] is False
        assert (await login(http, "new.officer@example.com")).status_code == 401
        self_disable = await http.patch(
            f"/admin/users/{admin.id}", json={"is_active": False}, headers=h
        )
        assert self_disable.status_code == 409
        assert (
            await http.patch(f"/admin/users/{CLAIM}", json={"is_active": False}, headers=h)
        ).status_code == 404
    actions = (await session.execute(select(AuditLog.action))).scalars().all()
    assert "user_created" in actions and "user_updated" in actions


# ------------------------------------------------------------------ hardening


async def test_security_headers_and_cors_expose_trace_id(session: AsyncSession) -> None:
    async with client_for(session) as (http, _, _):
        resp = await http.get("/health")
        bad = await http.get("/auth/me")  # error responses carry them too
        docs = await http.get("/docs")
    for r in (resp, bad):
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "no-referrer"
        assert "default-src 'none'" in r.headers["Content-Security-Policy"]
    assert "Content-Security-Policy" not in docs.headers  # Swagger UI needs its scripts


def test_production_refuses_development_defaults() -> None:
    dev = Settings(
        _env_file=None,
        app_env="production",
        openai_api_key=None,
        database_url="postgresql+asyncpg://claimpilot:claimpilot@localhost:5432/claimpilot",
    )
    problems = production_problems(dev)
    joined = " | ".join(problems)
    assert "JWT_SECRET" in joined and "OPENAI_API_KEY" in joined
    assert "DATABASE_URL" in joined and "CORS_ORIGINS" in joined
    with pytest.raises(InsecureConfigurationError, match="refusing to start"):
        validate_for_production(dev)


def test_production_accepts_a_real_configuration_and_other_envs_are_not_checked() -> None:
    good = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret="s" * 48,
        openai_api_key="sk-real-key-123",
        database_url="postgresql+asyncpg://app:Zx83kQ@db.internal:5432/claimpilot",
        cors_origins="https://claims.example.com",
    )
    assert production_problems(good) == []
    validate_for_production(good)
    validate_for_production(Settings(_env_file=None, app_env="development"))  # no checks


def test_well_known_seed_passwords_are_refused_in_production() -> None:
    cfg = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret="s" * 48,
        openai_api_key="sk-real-key-123",
        database_url="postgresql+asyncpg://app:Zx83kQ@db.internal:5432/claimpilot",
        cors_origins="https://claims.example.com",
        seed_admin_password="changeme",
    )
    assert any("SEED_ADMIN_PASSWORD" in p for p in production_problems(cfg))


async def test_refresh_tokens_are_not_stored_in_clear(session: AsyncSession) -> None:
    await make_user(session, "asha@example.com")
    async with client_for(session) as (http, _, _):
        token = (await login(http, "asha@example.com")).json()["refresh_token"]
    stored = (await session.execute(select(func.count()).select_from(RefreshToken))).scalar_one()
    rows = (await session.execute(select(RefreshToken.token_hash))).scalars().all()
    assert stored == 1 and token not in rows and len(rows[0]) == 64


# ------------------------------------------------------------------ spend recording


def test_record_spend_writes_one_hash_per_day(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.observability import usage

    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Pipe:
        def __getattr__(self, name: str):  # records hincrbyfloat / hincrby / expire
            return lambda *args: calls.append((name, args))

        def execute(self) -> None:
            calls.append(("execute", ()))

    class Client:
        def pipeline(self) -> Pipe:
            return Pipe()

    monkeypatch.setattr(usage, "_tracking", True)
    monkeypatch.setattr(usage, "_sync_client", lambda: Client())
    usage.record_spend(
        provider="openai", model="gpt-4o-mini", inr=0.5, tokens_in=100, tokens_out=20
    )
    key = usage.spend_key(usage.today())
    assert ("hincrbyfloat", (key, "inr", 0.5)) in calls
    assert ("hincrbyfloat", (key, "inr:openai:gpt-4o-mini", 0.5)) in calls
    assert ("hincrby", (key, "tokens_in", 100)) in calls and calls[-1] == ("execute", ())

    monkeypatch.setattr(usage, "_tracking", False)  # evaluation runs switch it off
    calls.clear()
    usage.record_spend(
        provider="anthropic", model="claude-haiku-5-5", inr=1, tokens_in=1, tokens_out=1
    )
    assert calls == []


def test_record_spend_never_raises_when_redis_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.observability import usage

    def boom() -> Any:
        raise ConnectionError("redis is down")

    monkeypatch.setattr(usage, "_tracking", True)
    monkeypatch.setattr(usage, "_sync_client", boom)
    usage.record_spend(provider="openai", model="m", inr=1, tokens_in=1, tokens_out=1)  # no error


def test_the_day_rolls_over_at_indian_midnight() -> None:
    from app.observability.usage import IST, seconds_until_tomorrow, today

    late = datetime(2026, 10, 25, 23, 30, tzinfo=IST)
    assert today(late) == "2026-10-25" and seconds_until_tomorrow(late) == 30 * 60
    assert today(late + timedelta(hours=1)) == "2026-10-26"
    utc_evening = datetime(2026, 10, 25, 20, 0, tzinfo=UTC)  # already the 26th in India
    assert today(utc_evening) == "2026-10-26"


def test_placeholder_and_short_mcp_tokens_are_flagged() -> None:
    from app.mcp_client.tokens import weak_identities

    env = {
        "MCP_NOTIFY_TOKENS": "notify-agent=replace-me-7,good-agent=Zk3pQ9vLm2XeR8tYb5Nw,short=abc123"
    }
    assert sorted(weak_identities("notify", env)) == ["notify-agent", "short"]
    assert weak_identities("claims_db", env) == []
