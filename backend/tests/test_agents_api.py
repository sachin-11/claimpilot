"""The agent-run API: start, stream, inspect and the human approval (Module 8)."""

import contextlib
from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
from agent_helpers import agent_deps, make_agent_world, mcp_world, standard_llm
from auth_helpers import bypass_auth, current_user_for
from langgraph.checkpoint.memory import InMemorySaver
from mcp_helpers import FakeRag
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.approvals import record_approval
from app.agents.runner import AgentRuntime
from app.agents.state import HumanApproval
from app.api.agent_runs import get_runtime
from app.core.db import get_session
from app.main import create_app
from app.models import (
    ActorType,
    AuditLog,
    Claim,
    ClaimApproval,
    ClaimStatus,
    Notification,
)


@contextlib.asynccontextmanager
async def api_client(
    session: AsyncSession, runtime_holder: dict, user=None
) -> AsyncIterator[httpx.AsyncClient]:
    from app.core.config import Settings

    app = create_app(Settings(_env_file=None, app_env="test", openai_api_key="sk-test"))

    async def shared_session() -> AsyncIterator[AsyncSession]:
        yield session

    async def runtime() -> AgentRuntime:
        return runtime_holder["runtime"]

    bypass_auth(app, user)
    app.dependency_overrides[get_session] = shared_session
    app.dependency_overrides[get_runtime] = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as http:
        yield http


async def start(http: httpx.AsyncClient, runtime: AgentRuntime, claim_id) -> str:
    resp = await http.post(f"/claims/{claim_id}/process")
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    await runtime.channels[run_id].task
    return run_id


def body(world, **kw) -> dict:
    return {"action": "approved", **kw}


async def test_process_stream_inspect_and_approve(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(agent_deps(session, provider, llm), InMemorySaver())
        async with api_client(
            session, {"runtime": runtime}, current_user_for(world.reviewer)
        ) as http:
            run_id = await start(http, runtime, world.claim.id)
            base = f"/claims/{world.claim.id}/runs/{run_id}"

            snapshot = (await http.get(base)).json()
            assert snapshot["status"] == "awaiting_approval"
            assert snapshot["decision"]["decision"] == "partial_approve"
            assert snapshot["awaiting_approval"]["decision"]["decision_id"]
            assert snapshot["coverage"][0]["cited_clauses"][0]["section_path"]

            stream = (await http.get(f"{base}/stream")).text
            for event in (
                "node_started",
                "routing",
                "tool_called",
                "node_completed",
                "awaiting_approval",
            ):
                assert f"event: {event}" in stream
            assert stream.index("event: node_started") < stream.index("event: awaiting_approval")
            assert "search_policy" in stream and "idempotency_key" not in stream

            # validation: nothing is written for a bad request
            bad = [
                ({"action": "edited", "final_amount": "50000"}, 422),  # no comment
                ({"action": "edited", "comment": "x"}, 422),  # no amount
                (
                    {"action": "edited", "final_amount": "99999", "comment": "more than claimed"},
                    422,
                ),
            ]
            for extra, code in bad:
                resp = await http.post(f"{base}/approve", json={**body(world), **extra})
                assert resp.status_code == code, (extra, resp.text)
            assert (await session.execute(select(ClaimApproval))).first() is None
            other = await http.get(f"/claims/{world.reviewer.id}/runs/{run_id}")
            assert other.status_code == 404  # run belongs to another claim

            edit = {"action": "edited", "final_amount": "50000", "comment": "goodwill"}
            resp = await http.post(f"{base}/approve", json=body(world, **edit))
            assert resp.status_code == 202, resp.text
            assert resp.json()["final_status"] == "partially_approved"
            assert Decimal(resp.json()["approved_amount"]) == 50000
            await runtime.channels[run_id].task

            done = (await http.get(base)).json()
            assert done["status"] == "completed" and done["notification"]["status"] == "sent_mock"
            again = await http.post(f"{base}/approve", json=body(world))
            assert again.status_code == 409  # a finished run takes no second decision

    claim = (await session.execute(select(Claim))).scalar_one()
    assert claim.status == ClaimStatus.PARTIALLY_APPROVED and claim.approved_amount == Decimal(
        "50000"
    )
    approval = (await session.execute(select(ClaimApproval))).scalar_one()
    assert (approval.action.value, approval.final_amount, approval.comment) == (
        "edited",
        Decimal("50000.00"),
        "goodwill",
    )
    audit = (
        await session.execute(select(AuditLog).where(AuditLog.action == "claim.approval_recorded"))
    ).scalar_one()
    assert audit.actor_type == ActorType.USER and audit.actor_id == str(world.reviewer.id)
    assert (
        await session.execute(select(Notification.template))
    ).scalar_one() == "partially_approved"


async def test_rejection_by_the_human(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(
            session, {"runtime": runtime}, current_user_for(world.reviewer)
        ) as http:
            run_id = await start(http, runtime, world.claim.id)
            resp = await http.post(
                f"/claims/{world.claim.id}/runs/{run_id}/approve",
                json=body(world, action="rejected", comment="duplicate claim"),
            )
            assert resp.status_code == 202 and resp.json()["final_status"] == "rejected"
            await runtime.channels[run_id].task
    assert (await session.execute(select(Claim.status))).scalar_one() == ClaimStatus.REJECTED
    assert (await session.execute(select(Notification.template))).scalar_one() == "rejected"


async def test_a_paused_run_survives_a_backend_restart(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    saver = InMemorySaver()  # stands in for the Postgres checkpointer: state outlives the runtime
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        before = AgentRuntime(agent_deps(session, provider, llm), saver)
        holder = {"runtime": before}
        async with api_client(session, holder, current_user_for(world.reviewer)) as http:
            run_id = await start(http, before, world.claim.id)

            after = AgentRuntime(agent_deps(session, provider, llm), saver)  # "restarted" process
            assert after.channels == {}
            holder["runtime"] = after
            base = f"/claims/{world.claim.id}/runs/{run_id}"
            assert (await http.get(base)).json()["status"] == "awaiting_approval"
            replay = (await http.get(f"{base}/stream")).text
            assert "event: awaiting_approval" in replay  # rebuilt from the checkpoint

            resp = await http.post(f"{base}/approve", json=body(world))
            assert resp.status_code == 202 and resp.json()["resumed"] is True
            await after.channels[run_id].task
            assert (await http.get(base)).json()["status"] == "completed"


async def test_a_failed_resume_can_be_retried_without_a_second_approval(
    session: AsyncSession,
) -> None:
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(
            session, {"runtime": runtime}, current_user_for(world.reviewer)
        ) as http:
            run_id = await start(http, runtime, world.claim.id)
            state = (await runtime.get_run_state(run_id))["state"]
            # the API wrote the approval, then died before resuming the graph
            await record_approval(
                session,
                claim_id=world.claim.id,
                run_id=run_id,
                decision=state["decision"],
                approval=HumanApproval(action="approved", reviewer=str(world.reviewer.id)),
            )
            resp = await http.post(
                f"/claims/{world.claim.id}/runs/{run_id}/approve", json=body(world)
            )
            assert resp.status_code == 202
            assert resp.json()["already_recorded"] is True and resp.json()["resumed"] is True
            await runtime.channels[run_id].task
    rows = (await session.execute(select(ClaimApproval))).scalars().all()
    assert len(rows) == 1
    assert (
        await session.execute(select(Notification.template))
    ).scalar_one() == "partially_approved"


async def test_unknown_claim_and_run(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(
            session, {"runtime": runtime}, current_user_for(world.reviewer)
        ) as http:
            missing = "00000000-0000-0000-0000-000000000001"
            assert (await http.post(f"/claims/{missing}/process")).status_code == 404
            assert (await http.get(f"/claims/{world.claim.id}/runs/nope")).status_code == 404


async def test_a_client_can_attach_right_after_process(session: AsyncSession) -> None:
    """Before the first checkpoint exists the run is `running`, not 404 (regression)."""
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(
            session, {"runtime": runtime}, current_user_for(world.reviewer)
        ) as http:
            resp = await http.post(f"/claims/{world.claim.id}/process")
            run_id = resp.json()["run_id"]
            base = f"/claims/{world.claim.id}/runs/{run_id}"
            assert (await http.get(base)).status_code == 200  # no checkpoint written yet
            stream = await http.get(f"{base}/stream")
            assert stream.status_code == 200 and "event: awaiting_approval" in stream.text


# ------------------------------------------------------------------ separation of duties


async def _run_as(session, world, who, comment=None, action="approved"):
    """Start a run (as a neutral admin) and try to approve it as `who`."""
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(session, {"runtime": runtime}) as starter:
            run_id = await start(starter, runtime, world.claim.id)
        async with api_client(session, {"runtime": runtime}, who) as http:
            payload = {"action": action, **({"comment": comment} if comment else {})}
            resp = await http.post(f"/claims/{world.claim.id}/runs/{run_id}/approve", json=payload)
        if resp.status_code == 202:
            await runtime.channels[run_id].task
    return resp


async def _second_user(session, role):
    import uuid

    from app.models import User

    user = User(
        email=f"u-{uuid.uuid4().hex[:6]}@example.com",
        full_name="Other",
        role=role,
        password_hash="x",
    )
    session.add(user)
    await session.flush()
    return user


async def test_an_officer_cannot_approve_the_claim_they_uploaded(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    world.claim.created_by = world.reviewer.id  # the reviewer uploaded it
    await session.flush()
    resp = await _run_as(session, world, current_user_for(world.reviewer))
    assert resp.status_code == 403 and "separation of duties" in resp.json()["error"]["message"]
    assert (await session.execute(select(ClaimApproval))).first() is None  # nothing was written
    assert (
        await session.execute(select(Claim.status))
    ).scalar_one() == ClaimStatus.AWAITING_APPROVAL


async def test_another_officer_can_approve_it(session: AsyncSession) -> None:
    from app.models import UserRole

    world = await make_agent_world(session)
    world.claim.created_by = world.reviewer.id
    other = await _second_user(session, UserRole.OFFICER)
    resp = await _run_as(session, world, current_user_for(other))
    assert resp.status_code == 202, resp.text
    approval = (await session.execute(select(ClaimApproval))).scalar_one()
    assert approval.reviewer_id == other.id  # the reviewer is the authenticated user


async def test_an_admin_override_needs_a_comment_and_is_audited(session: AsyncSession) -> None:
    from app.models import UserRole

    world = await make_agent_world(session)
    admin = await _second_user(session, UserRole.ADMIN)
    world.claim.created_by = admin.id  # the admin uploaded it
    await session.flush()
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        runtime = AgentRuntime(
            agent_deps(session, provider, standard_llm(str(world.chunk.id))), InMemorySaver()
        )
        async with api_client(session, {"runtime": runtime}) as starter:
            run_id = await start(starter, runtime, world.claim.id)
        url = f"/claims/{world.claim.id}/runs/{run_id}/approve"
        async with api_client(session, {"runtime": runtime}, current_user_for(admin)) as http:
            no_comment = await http.post(url, json={"action": "approved"})
            assert no_comment.status_code == 422
            assert "override" in no_comment.json()["error"]["message"]
            assert (
                await http.post(url, json={"action": "approved", "comment": "ok"})
            ).status_code == 422
            assert (await session.execute(select(ClaimApproval))).first() is None
            done = await http.post(
                url,
                json={"action": "approved", "comment": "reviewer on leave, urgent hospital case"},
            )
            assert done.status_code == 202, done.text
        await runtime.channels[run_id].task
    override = (
        await session.execute(
            select(AuditLog).where(AuditLog.action == "separation_of_duties_override")
        )
    ).scalar_one()
    assert override.actor_id == str(admin.id) and "on leave" in override.details["comment"]


async def test_an_admin_who_did_not_upload_needs_no_override(session: AsyncSession) -> None:
    from app.models import UserRole

    world = await make_agent_world(session)
    admin = await _second_user(session, UserRole.ADMIN)
    resp = await _run_as(session, world, current_user_for(admin))
    assert resp.status_code == 202
    assert (
        await session.execute(
            select(AuditLog).where(AuditLog.action == "separation_of_duties_override")
        )
    ).first() is None
