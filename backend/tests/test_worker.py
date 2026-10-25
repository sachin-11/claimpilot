"""Module 10, part C: the Redis run bus, the queued runtime and the arq jobs.

Real Redis on database 15 (flushed around every test; skipped when Redis is unreachable) and arq's
own burst worker, so the queue, retries and the job ids are the real thing. No LLM, no network."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
import redis.asyncio as aioredis
from agent_helpers import agent_deps, make_agent_world, mcp_world, standard_llm
from arq import Retry
from arq.connections import ArqRedis, RedisSettings, create_pool
from arq.worker import Worker
from fake_redis import FakeRedis
from langgraph.checkpoint.memory import InMemorySaver
from mcp_helpers import FakeRag
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.runner import AgentRuntime, RunNotFoundError
from app.agents.state import HumanApproval
from app.core.config import Settings
from app.models import ClaimDecision
from app.worker import tasks
from app.worker.bus import TERMINAL_EVENTS, RunBus
from app.worker.queue import JOB_EXTRACT, JOB_RUN, QUEUE_NAME, enqueue_extraction
from app.worker.runtime import QueuedAgentRuntime

TEST_QUEUE = "claimpilot:test-jobs"


def _dsn() -> str:
    base = Settings(_env_file=None).redis_url.rsplit("/", 1)[0]
    return f"{base}/15"  # never the database the app and the dev stack use


@pytest.fixture
async def pool() -> AsyncIterator[ArqRedis]:
    settings = RedisSettings.from_dsn(_dsn())
    settings.conn_retries, settings.conn_timeout = 0, 1
    try:
        arq_pool = await create_pool(settings, default_queue_name=TEST_QUEUE)
        await arq_pool.flushdb()
    except Exception:  # noqa: BLE001
        pytest.skip("Redis is not reachable")
    yield arq_pool
    await arq_pool.flushdb()
    await arq_pool.aclose()


@pytest.fixture
async def bus(pool: ArqRedis) -> RunBus:
    client = aioredis.Redis.from_url(_dsn(), decode_responses=True)
    yield RunBus(client)  # type: ignore[misc]
    await client.aclose()


async def burst(pool: ArqRedis, functions: list[Any], ctx: dict[str, Any], **kw: Any) -> None:
    """Run every queued job (including deferred retries) and stop."""
    worker = Worker(
        functions=functions,
        redis_pool=pool,
        queue_name=TEST_QUEUE,
        burst=True,
        poll_delay=0.05,
        handle_signals=False,
        max_tries=ctx.get("max_tries", 4),
        retry_jobs=True,
        ctx=ctx,
        **kw,
    )
    await asyncio.wait_for(worker.main(), timeout=30)
    await worker.close()


# ------------------------------------------------------------------ the bus


async def test_events_are_kept_per_phase_and_replayable(bus: RunBus) -> None:
    await bus.publish("r1", 0, "node_started", {"node": "intake"})
    await bus.publish("r1", 0, "awaiting_approval", {"claim_number": "CLM-1"})
    await bus.publish("r1", 1, "completed", {"run_id": "r1"})
    first = await bus.read("r1", 0, block_ms=10)
    assert [e for _, e, _ in first] == ["node_started", "awaiting_approval"]
    again = await bus.read("r1", 0, first[0][0], block_ms=10)  # a reconnecting client
    assert [e for _, e, _ in again] == ["awaiting_approval"]
    assert [e for _, e, _ in await bus.read("r1", 1, block_ms=10)] == ["completed"]
    assert await bus.read("r1", 2, block_ms=10) == []
    assert TERMINAL_EVENTS == {"awaiting_approval", "completed", "error"}


async def test_meta_roundtrip_and_expiry_is_set(bus: RunBus) -> None:
    await bus.set_meta("r2", state="queued", epoch=0, claim_number="CLM-2", user_id=None)
    meta = await bus.get_meta("r2")
    assert meta == {"state": "queued", "epoch": "0", "claim_number": "CLM-2", "user_id": ""}
    assert 0 < await bus.redis.ttl(bus.meta_key("r2")) <= 7 * 86400
    assert await bus.get_meta("missing") == {}


# ------------------------------------------------------------------ the queued runtime end to end


async def test_a_run_goes_through_the_queue_pauses_and_resumes(
    session: AsyncSession, pool: ArqRedis, bus: RunBus
) -> None:
    world = await make_agent_world(session)
    saver = InMemorySaver()  # stands in for the Postgres checkpoints: shared by API and worker
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, standard_llm(str(world.chunk.id)))
        api = QueuedAgentRuntime(deps, saver, queue=pool, bus=bus)
        worker_runtime = AgentRuntime(deps, saver)  # the worker's own runtime, same checkpoints
        ctx = {"runtime": worker_runtime, "bus": bus, "max_tries": 3, "worker_id": "test"}

        run_id = await api.start_run(world.claim.id, user_id=str(world.reviewer.id))

        # queued: the API answers without having run anything
        state = await api.get_run_state(run_id)
        assert state["status"] == "queued" and state["claim_number"] == world.claim.claim_number
        jobs = await pool.queued_jobs(queue_name=TEST_QUEUE)
        assert [(j.function, j.args[0], j.args[1]) for j in jobs] == [(JOB_RUN, "start", run_id)]
        await api._dispatch("start", run_id, {}, {})  # the same phase again: not queued twice
        assert len(await pool.queued_jobs(queue_name=TEST_QUEUE)) == 1

        await burst(pool, [tasks.run_agent], ctx)

        state = await api.get_run_state(run_id)  # read from the checkpoint by the API
        assert state["status"] == "awaiting_approval"
        assert (await bus.get_meta(run_id))["state"] == "idle"
        streamed = [item async for item in api.stream_run(run_id)]
        events = [item["event"] for item in streamed]
        assert events[0] == "node_started" and events[-1] == "awaiting_approval"
        assert [item["seq"] for item in streamed] == list(range(len(streamed)))
        assert "routing" in events and "tool_called" in events

        # a client that connects again gets the same events: nothing is lost with the stream
        assert [i["event"] async for i in api.stream_run(run_id)] == events

        approval = HumanApproval(action="approved", reviewer=str(world.reviewer.id))
        await api.resume_run(run_id, approval)
        assert (await api.get_run_state(run_id))["status"] == "queued"
        with pytest.raises(Exception, match="not awaiting approval"):  # not resumed twice
            await api.resume_run(run_id, approval)
        await burst(pool, [tasks.run_agent], ctx)

        done = await api.get_run_state(run_id)
        assert done["status"] == "completed"
        resumed = [item["event"] async for item in api.stream_run(run_id)]
        assert resumed[-1] == "completed" and "awaiting_approval" not in resumed  # its own phase


async def test_stream_waits_for_a_run_that_is_still_queued(
    session: AsyncSession, pool: ArqRedis, bus: RunBus
) -> None:
    world = await make_agent_world(session)
    saver = InMemorySaver()
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, standard_llm(str(world.chunk.id)))
        api = QueuedAgentRuntime(deps, saver, queue=pool, bus=bus)
        ctx = {"runtime": AgentRuntime(deps, saver), "bus": bus, "max_tries": 3}
        run_id = await api.start_run(world.claim.id)

        async def client() -> list[str]:  # connects BEFORE the worker has picked the job up
            return [item["event"] async for item in api.stream_run(run_id)]

        reader = asyncio.create_task(client())
        await asyncio.sleep(0.3)
        assert not reader.done()
        await burst(pool, [tasks.run_agent], ctx)
        events = await asyncio.wait_for(reader, timeout=10)
    assert events[-1] == "awaiting_approval"


async def test_unknown_run_is_not_found(session: AsyncSession, pool: ArqRedis, bus: RunBus) -> None:
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, standard_llm(str(world.chunk.id)))
        api = QueuedAgentRuntime(deps, InMemorySaver(), queue=pool, bus=bus)
        with pytest.raises(RunNotFoundError):
            await api.get_run_state("nope")


# ------------------------------------------------------------------ safe to run twice


async def test_a_redelivered_job_does_not_run_the_graph_again(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    saver = InMemorySaver()
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, standard_llm(str(world.chunk.id)))
        runtime = AgentRuntime(deps, saver)
        meta = {"claim_id": str(world.claim.id), "claim_number": world.claim.claim_number}
        payload = {
            "claim_id": str(world.claim.id),
            "claim_number": world.claim.claim_number,
            "run_id": "r-redeliver",
            "completed_steps": [],
            "errors": [],
            "models_used": {},
        }
        assert await runtime.execute("start", "r-redeliver", payload, meta) == "awaiting_approval"
        decisions = (
            await session.execute(select(func.count()).select_from(ClaimDecision))
        ).scalar_one()
        # the same job again (an at-least-once queue): the run already paused, nothing happens
        assert await runtime.execute("start", "r-redeliver", payload, meta) == "awaiting_approval"
        again = (
            await session.execute(select(func.count()).select_from(ClaimDecision))
        ).scalar_one()
        assert again == decisions == 1

        # a worker restart between pause and resume: a NEW runtime on the same checkpoints
        restarted = AgentRuntime(deps, saver)
        approval = HumanApproval(action="approved", reviewer=str(world.reviewer.id))
        status = await restarted.execute(
            "resume", "r-redeliver", approval.model_dump(mode="json"), meta
        )
        assert status == "completed"
        # ... and the resume job delivered once more finds nothing left to do
        assert (
            await restarted.execute("resume", "r-redeliver", approval.model_dump(mode="json"), meta)
            == "completed"
        )


# ------------------------------------------------------------------ retries


class FlakyRuntime:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        self.calls = 0

    async def execute(self, kind: str, run_id: str, payload: Any, meta: Any, *, sink: Any) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        await sink("completed", {"run_id": run_id})
        return "completed"


async def test_a_transient_failure_is_retried_with_backoff(
    pool: ArqRedis, bus: RunBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "RETRY_BASE_SECONDS", 0.05)
    runtime = FlakyRuntime([ConnectionError("db gone"), TimeoutError("slow")])
    await pool.enqueue_job(
        JOB_RUN, "start", "r-flaky", {}, {}, _job_id="r-flaky:0", _queue_name=TEST_QUEUE
    )
    await burst(pool, [tasks.run_agent], {"runtime": runtime, "bus": bus, "max_tries": 4})
    assert runtime.calls == 3  # two failures, then the run
    assert (await bus.get_meta("r-flaky"))["state"] == "idle"
    assert [e for _, e, _ in await bus.read("r-flaky", 0, block_ms=10)] == ["completed"]


async def test_a_bug_is_not_retried(pool: ArqRedis, bus: RunBus) -> None:
    runtime = FlakyRuntime([ValueError("a bug")])
    await pool.enqueue_job(
        JOB_RUN, "start", "r-bug", {}, {}, _job_id="r-bug:0", _queue_name=TEST_QUEUE
    )
    await burst(pool, [tasks.run_agent], {"runtime": runtime, "bus": bus, "max_tries": 4})
    assert runtime.calls == 1
    meta = await bus.get_meta("r-bug")
    assert meta["state"] == "idle" and meta["status"] == "failed"


async def test_retries_stop_at_max_tries(
    pool: ArqRedis, bus: RunBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "RETRY_BASE_SECONDS", 0.02)
    runtime = FlakyRuntime([ConnectionError("down")] * 10)
    await pool.enqueue_job(
        JOB_RUN, "start", "r-down", {}, {}, _job_id="r-down:0", _queue_name=TEST_QUEUE
    )
    await burst(pool, [tasks.run_agent], {"runtime": runtime, "bus": bus, "max_tries": 3})
    assert runtime.calls == 3
    assert (await bus.get_meta("r-down"))["status"] == "failed"


def test_backoff_doubles_and_what_counts_as_transient() -> None:
    import httpx
    from sqlalchemy.exc import DBAPIError, OperationalError

    assert [tasks.backoff(n) for n in (1, 2, 3)] == [5.0, 10.0, 20.0]
    assert tasks.is_transient(ConnectionError()) and tasks.is_transient(httpx.ConnectTimeout("t"))
    assert not tasks.is_transient(ValueError("bug")) and not tasks.is_transient(KeyError("k"))
    dropped = OperationalError("select 1", {}, Exception("server closed the connection"))
    dropped.connection_invalidated = True
    assert tasks.is_transient(dropped)
    plain = DBAPIError("select 1", {}, Exception("syntax error"))
    assert not tasks.is_transient(plain)


# ------------------------------------------------------------------ extraction jobs


async def test_one_extraction_per_claim_at_a_time(pool: ArqRedis) -> None:
    assert await enqueue_extraction(pool, "claim-1", "digital") is True
    assert await enqueue_extraction(pool, "claim-1", "scanned") is False  # already queued
    assert await enqueue_extraction(pool, "claim-2", "digital") is True
    assert len(await pool.queued_jobs(queue_name=TEST_QUEUE)) == 2
    await pool.delete("lock:extract:claim-1", "lock:extract:claim-2")


async def test_extraction_job_releases_the_lock_and_retries_transient_errors(
    pool: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "RETRY_BASE_SECONDS", 0.02)

    class Intake:
        def __init__(self) -> None:
            self.calls = 0

        async def process(self, claim_id: Any, prefer: str = "digital") -> Any:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("blip")

            class Result:
                status = "completed"

            return Result()

    intake = Intake()
    claim = "11111111-1111-1111-1111-111111111111"
    assert await pool.set(f"lock:extract:{claim}", "1", nx=True, ex=900)
    await pool.enqueue_job(JOB_EXTRACT, claim, "digital", _queue_name=TEST_QUEUE)
    await burst(pool, [tasks.extract_claim], {"intake": intake, "max_tries": 3})
    assert intake.calls == 2
    assert await pool.exists(f"lock:extract:{claim}") == 0


# ------------------------------------------------------------------ readiness


async def test_worker_heartbeat_check() -> None:
    from app.api.health import check_worker
    from app.worker.queue import HEALTH_KEY

    redis = FakeRedis()
    with pytest.raises(RuntimeError, match="no worker heartbeat"):
        await check_worker(redis)  # type: ignore[arg-type]
    redis.values[HEALTH_KEY] = "j_complete=1"
    await check_worker(redis)  # type: ignore[arg-type]


def test_readiness_reports_everything_but_only_requires_the_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    async def ok(*_: object) -> None:
        return None

    async def worker_down(*_: object) -> None:
        raise RuntimeError("no worker heartbeat")

    async def langfuse_down() -> dict[str, Any]:
        return {"status": "error", "detail": "unreachable", "optional": True}

    for name in ("check_postgres", "check_redis", "check_mcp"):
        monkeypatch.setattr(f"app.api.health.{name}", ok)
    monkeypatch.setattr("app.api.health.check_worker", ok)
    monkeypatch.setattr("app.api.health.check_langfuse", langfuse_down)
    settings = Settings(
        _env_file=None, app_env="test", use_worker=False, readiness_extra_checks=True
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/health/ready").json()
        assert body["status"] == "ready"  # Langfuse down does NOT fail readiness
        assert body["dependencies"]["langfuse"]["status"] == "error"
        assert {"postgres", "redis", "mcp_policy_docs", "mcp_claims_db", "mcp_notify"} <= set(
            body["dependencies"]
        )
        assert "worker" not in body["dependencies"]  # no worker configured

    monkeypatch.setattr("app.api.health.check_worker", worker_down)
    settings = Settings(
        _env_file=None,
        app_env="test",
        use_worker=True,
        readiness_extra_checks=True,
        redis_url=_dsn(),
    )
    with TestClient(create_app(settings)) as client:
        resp = client.get("/health/ready")
        assert (
            resp.status_code == 503 and resp.json()["dependencies"]["worker"]["status"] == "error"
        )


def test_retry_is_arq_retry() -> None:
    assert issubclass(Retry, Exception)  # the exception the jobs raise to be tried again
    assert QUEUE_NAME == "claimpilot:jobs"
