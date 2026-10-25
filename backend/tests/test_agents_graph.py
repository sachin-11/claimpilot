"""The LangGraph workflow end to end, on the REAL MCP servers (in memory) and the test database,
with a scripted fake LLM. Covers routing, least privilege, the interrupt / resume cycle,
checkpoint persistence, failure paths and idempotency."""

import asyncio
import contextlib
import uuid
from decimal import Decimal

import httpx
import pytest
from agent_helpers import (
    ROOM_RENT_EXTRACTION,
    ScriptedLLM,
    agent_deps,
    initial_state,
    make_agent_world,
    mcp_world,
    passing_coverage,
    standard_llm,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp_helpers import FakeRag
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.agents.nodes.coverage import CoverageCheckLLM, CoverageLLMOutput
from app.agents.nodes.decision import make_decision_node
from app.agents.nodes.finalize import make_finalize_node
from app.agents.runner import AgentRuntime
from app.agents.supervisor import SupervisorChoice
from app.agents.tools import AGENT_TOOLS
from app.models import (
    ActorType,
    AuditLog,
    Claim,
    ClaimApproval,
    ClaimDecision,
    ClaimStatus,
    Notification,
)
from app.repositories.claim import ClaimRepository

RUN = "run-0001-abcdefgh"


def config(run_id: str = RUN, limit: int = 40) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": limit}


async def status_of(session: AsyncSession, claim_id) -> ClaimStatus:
    await session.commit()  # release the savepoint so a fresh read sees the latest row
    claim = (await session.execute(select(Claim).where(Claim.id == claim_id))).scalar_one()
    await session.refresh(claim)
    return claim.status


async def settle(session: AsyncSession, claim, status: ClaimStatus, amount=None) -> None:
    """What the approval API does to the claim, so finalize / notify can run."""
    claim.approved_amount = amount
    await ClaimRepository(session).update_status(claim, status, actor_type=ActorType.USER)


# ------------------------------------------------------------------------------ happy path


async def test_partial_claim_runs_pauses_then_finalizes(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())

        updates: list[dict] = []
        async for chunk in graph.astream(initial_state(world), config(), stream_mode="updates"):
            updates.append(chunk)
        names = [list(u)[0] for u in updates]
        assert names[:2] == ["intake", "supervisor"]
        # coverage and risk ran in the same super-step: both sit between two supervisor visits
        assert set(names[2:4]) == {"coverage", "risk"}
        assert names[4:6] == ["supervisor", "decision"]
        assert names[-1] == "__interrupt__"  # paused for the human

        state = (await graph.aget_state(config())).values
        decision = state["decision"]
        assert decision["decision"] == "partial_approve"
        # 67,200 claimed; room-rent cut 20,700; 5% co-pay on 46,500 = 2,325 -> 44,175
        assert Decimal(decision["payable_amount"]) == Decimal("44175.00")
        assert [d["rule"] for d in decision["deductions"]] == ["room_rent_limit", "co_pay_percent"]
        assert decision["cited_clauses"][0]["section_path"].startswith("6 Waiting Period")
        assert state["models_used"]["coverage"] == "fake-model"
        assert state["risk"]["score"] == 0
        assert await status_of(session, world.claim.id) == ClaimStatus.AWAITING_APPROVAL

        row = (await session.execute(select(ClaimDecision))).scalar_one()
        assert row.agent_run_id == RUN and str(row.id) == decision["decision_id"]
        assert row.cited_clauses[0]["chunk_id"] == str(world.chunk.id)

        # the approval API settles the claim, then the graph resumes
        await settle(session, world.claim, ClaimStatus.PARTIALLY_APPROVED, Decimal("44175"))
        approval = {"action": "approved", "reviewer": str(world.reviewer.id)}
        final = await graph.ainvoke(Command(resume=approval), config())
        assert final["completed_steps"][-1] == "finalize"
        assert final["notification"] == {"status": "sent_mock", "template": "partially_approved"}

    sent = (await session.execute(select(Notification))).scalar_one()
    assert sent.template == "partially_approved" and "44,175.00" in sent.rendered_message
    assert sent.idempotency_key == f"{RUN}-notify"


async def test_each_agent_gets_only_its_own_tools(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        for agent, servers in AGENT_TOOLS.items():
            loaded = {t.name for t in await provider.for_agent(agent)}
            assert loaded == {n for names in servers.values() for n in names}
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())

    # the coverage model was bound to the policy_docs tools only
    assert llm.bound_tool_names == [["get_policy_rules", "search_policy"]]
    risk_tools = {t.name for t in await _tools(session, world, "risk")}
    assert risk_tools == {
        "get_claim",
        "get_claim_extraction",
        "get_customer_history",
        "list_claims",
    }
    assert not {"update_claim_status", "save_decision_draft"} & risk_tools


async def _tools(session, world, agent):
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        return await provider.for_agent(agent)


# ------------------------------------------------------------------------------ routing


async def test_missing_documents_skip_coverage_and_go_to_decision(session: AsyncSession) -> None:
    world = await make_agent_world(session, missing_documents=["discharge_summary"])
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        steps = [
            list(u)[0]
            async for u in graph.astream(initial_state(world), config(), stream_mode="updates")
        ]
    assert "coverage" not in steps and "risk" not in steps
    assert steps == ["intake", "supervisor", "decision", "__interrupt__"]
    assert "react" not in llm.calls and "CoverageLLMOutput" not in llm.calls
    state = (await graph.aget_state(config())).values
    assert state["decision"]["decision"] == "request_documents"
    assert state["decision"]["payable_amount"] is None
    assert await status_of(session, world.claim.id) == ClaimStatus.AWAITING_APPROVAL


async def test_invalid_supervisor_choice_uses_the_default_order(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(
        str(world.chunk.id),
        {SupervisorChoice: SupervisorChoice(next_steps=["decision"], reason="skip ahead")},
    )
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        steps = [
            list(u)
            async for u in graph.astream(initial_state(world), config(), stream_mode="updates")
        ]
    # "decision" was not allowed after intake, so coverage and risk still ran first
    flat = [n for group in steps for n in group]
    assert flat.index("coverage") < flat.index("decision") and "risk" in flat


# ------------------------------------------------------------------------------ decisions


async def test_waiting_period_claim_is_rejected_when_rules_and_wording_agree(
    session: AsyncSession,
) -> None:
    from datetime import date

    world = await make_agent_world(session, start=date(2025, 2, 10))  # admitted 19 days later
    cid = str(world.chunk.id)
    coverage = CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(
                check="initial_waiting",
                applies=True,
                result="fail",
                explanation="Within the 30-day initial waiting period.",
                chunk_ids=[cid],
            )
        ]
    )
    llm = standard_llm(cid, {CoverageLLMOutput: coverage})
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
    decision = (await graph.aget_state(config())).values["decision"]
    assert decision["decision"] == "reject" and Decimal(decision["payable_amount"]) == 0
    assert decision["cited_clauses"][0]["chunk_id"] == cid


async def test_uncited_or_invented_citations_make_the_check_unclear(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    coverage = CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(
                check="initial_waiting",
                applies=True,
                result="pass",
                explanation="Fine.",
                chunk_ids=["00000000-0000-0000-0000-000000000000"],
            )
        ]
    )
    llm = standard_llm(str(world.chunk.id), {CoverageLLMOutput: coverage})
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
    state = (await graph.aget_state(config())).values
    assert (
        state["coverage"][0]["result"] == "unclear" and state["coverage"][0]["cited_clauses"] == []
    )
    assert state["decision"]["decision"] == "flag_for_review"


async def test_the_model_can_cite_clauses_by_short_ref(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(
        str(world.chunk.id), {CoverageLLMOutput: passing_coverage("C1")}
    )  # the first clause search_policy returns is C1
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
    state = (await graph.aget_state(config())).values
    cited = state["coverage"][0]["cited_clauses"]
    assert [c["chunk_id"] for c in cited] == [str(world.chunk.id)]
    assert state["decision"]["decision"] == "partial_approve"


async def test_risk_flag_forces_review(session: AsyncSession) -> None:
    ext = {**ROOM_RENT_EXTRACTION, "admission_date": "2024-01-20"}  # 19 days after policy start
    world = await make_agent_world(session, extraction=ext)
    llm = standard_llm(
        str(world.chunk.id),
        {CoverageLLMOutput: passing_coverage(str(world.chunk.id))},
    )
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
    state = (await graph.aget_state(config())).values
    codes = [f["code"] for f in state["risk"]["flags"]]
    assert "admission_soon_after_policy_start" in codes and state["risk"]["score"] >= 50
    assert state["decision"]["decision"] == "flag_for_review"
    assert "admission_soon_after_policy_start" in state["decision"]["risk_flags"]


# ------------------------------------------------------------------------------ approval cycle


@pytest.mark.parametrize(
    ("action", "amount", "comment", "status", "template"),
    [
        ("approved", None, None, ClaimStatus.PARTIALLY_APPROVED, "partially_approved"),
        (
            "edited",
            "50000",
            "goodwill adjustment",
            ClaimStatus.PARTIALLY_APPROVED,
            "partially_approved",
        ),
        ("rejected", None, "not payable", ClaimStatus.REJECTED, "rejected"),
    ],
)
async def test_resume_with_each_human_action(
    session: AsyncSession, action, amount, comment, status, template
) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
        await settle(session, world.claim, status, Decimal(amount) if amount else None)
        resume = {
            "action": action,
            "final_amount": amount,
            "comment": comment,
            "reviewer": str(world.reviewer.id),
        }
        final = await graph.ainvoke(Command(resume=resume), config())
    assert final["approval"]["action"] == action
    assert final["notification"]["template"] == template
    assert (await session.execute(select(Notification.template))).scalar_one() == template


async def test_resume_with_an_incomplete_edit_goes_to_the_error_path(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
        bad = {"action": "edited", "final_amount": "100", "comment": None, "reviewer": "r"}
        final = await graph.ainvoke(Command(resume=bad), config())
    assert final["errors"][0]["node"] == "approval" and "comment" in final["errors"][0]["error"]
    assert "finalize" not in final["completed_steps"]
    assert await status_of(session, world.claim.id) == ClaimStatus.FAILED


async def test_finalize_failure_to_notify_does_not_fail_the_run(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
        # claim never settled by the API: notify refuses 'partially_approved' for this status
        final = await graph.ainvoke(
            Command(resume={"action": "approved", "reviewer": str(world.reviewer.id)}), config()
        )
    assert final["notification"]["status"] == "failed" and not final.get("errors")
    assert final["completed_steps"][-1] == "finalize"


# ------------------------------------------------------------------------------ persistence


@contextlib.asynccontextmanager
async def postgres_saver(conninfo: str):  # type: ignore[no-untyped-def]
    """AsyncPostgresSaver on its own connection pool: the same wiring as AgentRuntime.create."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver
    finally:
        await pool.close()


async def test_a_fresh_graph_instance_resumes_from_the_checkpoint(
    session: AsyncSession, test_database_url: str
) -> None:
    conninfo = test_database_url.replace("postgresql+asyncpg://", "postgresql://")
    run_id = "persist-" + uuid.uuid4().hex  # checkpoints are committed: never reuse a thread
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        async with postgres_saver(conninfo) as saver:
            first = build_graph(agent_deps(session, provider, llm), saver)
            await first.ainvoke(initial_state(world, run_id), config(run_id))
        # "restart": new pool, new saver, new graph object; nothing is shared in memory
        async with postgres_saver(conninfo) as saver:
            second = build_graph(agent_deps(session, provider, llm), saver)
            snapshot = await second.aget_state(config(run_id))
            assert snapshot.next == ("approval",) and snapshot.values["decision"]["decision"]
            await settle(session, world.claim, ClaimStatus.PARTIALLY_APPROVED, Decimal("44175"))
            final = await second.ainvoke(
                Command(resume={"action": "approved", "reviewer": str(world.reviewer.id)}),
                config(run_id),
            )
    assert final["completed_steps"][-1] == "finalize"


# ------------------------------------------------------------------------------ failures


async def test_node_failure_routes_to_error_and_marks_the_claim_failed(
    session: AsyncSession,
) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    del llm.structured[CoverageLLMOutput]  # coverage's final answer raises (not transient)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        graph = build_graph(agent_deps(session, provider, llm), InMemorySaver())
        final = await graph.ainvoke(initial_state(world), config())
    assert [e["node"] for e in final["errors"]] == ["coverage"]
    assert "decision" not in final["completed_steps"] and final["completed_steps"][-1] == "error"
    assert await status_of(session, world.claim.id) == ClaimStatus.FAILED
    changes = (
        await session.execute(select(AuditLog).where(AuditLog.action == "claim.status_changed"))
    ).scalars()
    last = [c for c in changes if c.details.get("to") == "failed"][0]
    assert last.actor_type == ActorType.AGENT and last.actor_id == "agent-under-test"


async def test_transient_errors_are_retried_then_routed_to_error(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    attempts = {"n": 0}

    def flaky(_messages):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("provider unreachable")
        return passing_coverage(str(world.chunk.id))

    llm = standard_llm(str(world.chunk.id), {CoverageLLMOutput: flaky})
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, llm, node_retry_initial_interval=0.01)
        final = await build_graph(deps, InMemorySaver()).ainvoke(initial_state(world), config())
    assert attempts["n"] == 3 and not final.get("errors")  # succeeded on the third attempt

    # a transient error that never clears: retries run out, the run ends on the error path
    world2 = await make_agent_world(session, number="CLM-A2", policy_code="POL-T2")
    always = standard_llm(
        str(world2.chunk.id),
        {CoverageLLMOutput: lambda _m: (_ for _ in ()).throw(httpx.ConnectError("down"))},
    )
    async with mcp_world(session, FakeRag(world2.chunk)) as provider:
        deps = agent_deps(session, provider, always, node_retry_initial_interval=0.01)
        runtime = AgentRuntime(deps, InMemorySaver())
        run_id = await runtime.start_run(world2.claim.id)
        await runtime.channels[run_id].task
        snapshot = await runtime.get_run_state(run_id)
    assert snapshot["status"] == "failed"
    assert "ConnectError" in snapshot["state"]["errors"][0]["error"]
    assert await status_of(session, world2.claim.id) == ClaimStatus.FAILED


async def test_recursion_limit_ends_the_run_on_the_error_path(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, llm, agent_recursion_limit=3)
        runtime = AgentRuntime(deps, InMemorySaver())
        run_id = await runtime.start_run(world.claim.id)
        await runtime.channels[run_id].task
        snapshot = await runtime.get_run_state(run_id)
        events = [e["event"] for e in runtime.channels[run_id].events]
    assert (
        snapshot["status"] == "failed" and "step limit" in snapshot["state"]["errors"][0]["error"]
    )
    assert events[-1] == "error"
    assert await status_of(session, world.claim.id) == ClaimStatus.FAILED


async def test_agent_tool_budget_is_not_an_error(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    endless = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_policy",
                    "id": f"c{i}",
                    "type": "tool_call",
                    "args": {"query": "waiting", "policy_code": "POL-T"},
                }
            ],
        )
        for i in range(10)
    ]
    llm = standard_llm(str(world.chunk.id))
    llm.react = endless
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, llm, coverage_max_tool_calls=2)
        final = await build_graph(deps, InMemorySaver()).ainvoke(initial_state(world), config())
    assert not final.get("errors") and "decision" in final["completed_steps"]
    assert llm.calls.count("react") <= 4  # the budget stopped the loop


# ------------------------------------------------------------------------------ idempotency


async def test_rerunning_decision_and_notify_does_not_duplicate_rows(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    llm = standard_llm(str(world.chunk.id))
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, llm)
        graph = build_graph(deps, InMemorySaver())
        await graph.ainvoke(initial_state(world), config())
        state = dict((await graph.aget_state(config())).values)

        first = state["decision"]
        again = await make_decision_node(deps)(state)  # a retry of the whole node
        assert again["decision"]["decision_id"] == first["decision_id"]

        await settle(session, world.claim, ClaimStatus.PARTIALLY_APPROVED, Decimal("44175"))
        state["approval"] = {"action": "approved", "reviewer": str(world.reviewer.id)}
        await make_finalize_node(deps)(state)
        await make_finalize_node(deps)(state)

    decisions = (
        await session.execute(select(func.count()).select_from(ClaimDecision))
    ).scalar_one()
    sent = (await session.execute(select(func.count()).select_from(Notification))).scalar_one()
    awaiting = (
        await session.execute(
            select(func.count())
            .select_from(AuditLog)
            .where(
                AuditLog.action == "claim.status_changed",
                AuditLog.details["to"].astext == "awaiting_approval",
            )
        )
    ).scalar_one()
    assert (decisions, sent, awaiting) == (1, 1, 1)


async def test_unused_import_guard() -> None:
    assert ScriptedLLM and ClaimApproval and asyncio  # keep helper imports referenced
