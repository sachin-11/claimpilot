"""MCP servers (Module 7): tools called through the SDK's in-memory client; no real LLM."""

import json
from decimal import Decimal

import httpx
import pytest
from mcp_helpers import FakeRag, call, connected, deps_for, make_claim, make_world
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp_client.tokens import identities, token_for, token_map
from app.models import (
    ActorType,
    AuditLog,
    ClaimDecision,
    ClaimExtraction,
    ClaimStatus,
    Notification,
    NotificationChannel,
    ToolIdempotency,
)
from app.repositories.claim import ClaimRepository
from mcp_servers.claims_db.server import create_server as create_claims_db
from mcp_servers.claims_db.state_machine import TransitionError, check_agent_transition
from mcp_servers.common.auth import BearerAuthMiddleware
from mcp_servers.notify.server import create_server as create_notify
from mcp_servers.notify.server import mask_recipient, sanitize_extra
from mcp_servers.policy_docs.server import create_server as create_policy_docs

S = ClaimStatus
KEY = "00000000-aaaa-bbbb-cccc-000000000001"


async def tool_names(server) -> set[str]:
    async with connected(server) as client:
        return {t.name for t in (await client.list_tools()).tools}


async def audit_for(session: AsyncSession, action: str) -> list[AuditLog]:
    rows = await session.execute(select(AuditLog).where(AuditLog.action == action))
    return list(rows.scalars())


# ------------------------------------------------------------------------------ listing


async def test_tool_listing_and_descriptions(session: AsyncSession) -> None:
    expected = {
        "policy_docs": (
            create_policy_docs,
            {"list_policies", "search_policy", "answer_policy_question", "get_policy_rules"},
        ),
        "claims_db": (
            create_claims_db,
            {
                "get_claim",
                "get_claim_extraction",
                "get_customer_history",
                "list_claims",
                "update_claim_status",
                "save_decision_draft",
            },
        ),
        "notify": (create_notify, {"send_status_update"}),
    }
    for _name, (factory, names) in expected.items():
        server = factory(deps_for(session, write_enabled=True))
        async with connected(server) as client:
            tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == names
        assert all(t.description and len(t.description) > 40 for t in tools)


async def test_policy_docs_resources_and_constraints(session: AsyncSession) -> None:
    server = create_policy_docs(deps_for(session))
    async with connected(server) as client:
        templates = {
            t.uriTemplate for t in (await client.list_resource_templates()).resourceTemplates
        }
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert templates == {
        "policy://{policy_code}/sections",
        "policy://{policy_code}/chunk/{chunk_id}",
    }
    top_n = tools["search_policy"].inputSchema["properties"]["top_n"]
    assert (top_n["minimum"], top_n["maximum"]) == (1, 10)


async def test_write_tools_absent_when_write_disabled(session: AsyncSession) -> None:
    claims = await tool_names(create_claims_db(deps_for(session, write_enabled=False)))
    assert claims == {"get_claim", "get_claim_extraction", "get_customer_history", "list_claims"}
    assert await tool_names(create_notify(deps_for(session, write_enabled=False))) == set()
    assert "update_claim_status" in await tool_names(
        create_claims_db(deps_for(session, write_enabled=True))
    )


# ------------------------------------------------------------------------------ policy_docs


async def test_search_policy_returns_clauses_with_section_paths(session: AsyncSession) -> None:
    world = await make_world(session)
    rag = FakeRag(world.chunk)
    server = create_policy_docs(deps_for(session, rag=rag, max_clause_chars=300))
    async with connected(server) as client:
        err, clauses = await call(
            client,
            "search_policy",
            {"query": "cataract waiting", "policy_code": "POL-T", "top_n": 3},
        )
        assert not err
        assert clauses[0]["section_path"].startswith("6 Waiting Period")
        assert clauses[0]["chunk_id"] == str(world.chunk.id)
        assert clauses[0]["page_start"] == 7
        assert clauses[0]["score"] == 0.91
        assert len(clauses[0]["content"]) <= 300  # truncated
        assert rag.search_calls == [
            {"query": "cataract waiting", "policy_codes": ["POL-T"], "top_n": 3}
        ]

        err, msg = await call(client, "search_policy", {"query": "cataract", "top_n": 11})
        assert err  # constraint: top_n <= 10
        err, msg = await call(client, "search_policy", {"query": "cataract", "policy_code": "NOPE"})
        assert err and "list_policies" in msg


async def test_other_policy_tools_and_resources(session: AsyncSession) -> None:
    world = await make_world(session)
    server = create_policy_docs(deps_for(session, rag=FakeRag(world.chunk)))
    async with connected(server) as client:
        err, policies = await call(client, "list_policies")
        assert policies == [
            {
                "policy_code": "POL-T",
                "name": "Test Policy",
                "insurer": "Test Insurer",
                "chunk_count": 1,
            }
        ]
        err, rules = await call(client, "get_policy_rules", {"policy_code": "POL-T"})
        assert rules["rules"] == {"waiting": {"ped": 36}}
        err, answer = await call(client, "answer_policy_question", {"question": "cataract wait?"})
        assert answer["insufficient_context"] is False and answer["confidence"] == "high"
        assert answer["citations"][0]["section_path"] == "6.3"

        sections = await client.read_resource("policy://POL-T/sections")
        assert json.loads(sections.contents[0].text) == [world.chunk.section_path]
        chunk = await client.read_resource(f"policy://POL-T/chunk/{world.chunk.id}")
        body = json.loads(chunk.contents[0].text)
        assert body["metadata"] == {"clauses": ["6.3"]}  # embed_text is not exposed


# ------------------------------------------------------------------------------ claims_db reads


async def test_claim_reads_hide_contact_details(session: AsyncSession) -> None:
    world = await make_world(session)
    claim = await make_claim(session, world)
    prior = await make_claim(session, world, "CLM-T0")
    prior.status, prior.approved_amount = S.APPROVED, Decimal("100000")
    session.add(
        ClaimExtraction(
            claim_id=claim.id,
            extracted_data={
                "patient_name": "Asha Verma",
                "procedure": "Cataract surgery",
                "consistency_issues": [{"type": "date_mismatch", "severity": "warning"}],
                "missing_documents": ["discharge_summary"],
                "field_sources": {"x": {}},
                "intake": {},
            },
            confidence=0.9,
            validation_errors=[],
            version=1,
        )
    )
    await session.flush()
    server = create_claims_db(deps_for(session))
    async with connected(server) as client:
        err, summary = await call(client, "get_claim", {"claim_number": "CLM-T1"})
        assert summary["policy_code"] == "POL-T" and summary["status"] == "submitted"
        assert "asha" not in json.dumps(summary).lower() and "9876" not in json.dumps(summary)

        err, extraction = await call(client, "get_claim_extraction", {"claim_number": "CLM-T1"})
        assert extraction["consistency_issues"][0]["type"] == "date_mismatch"
        assert extraction["missing_documents"] == ["discharge_summary"]
        assert extraction["extracted"] == {"procedure": "Cataract surgery"}
        assert extraction["patient_name_present"] is True
        assert "Asha" not in json.dumps(extraction)

        err, history = await call(client, "get_customer_history", {"claim_number": "CLM-T1"})
        assert history["previous_claim_count"] == 1
        assert history["remaining_sum_insured"] == 400000.0
        assert history["days_from_policy_start_to_admission"] == 425

        err, listed = await call(client, "list_claims", {"status": "approved", "limit": 5})
        assert [c["claim_number"] for c in listed] == ["CLM-T0"]

        err, msg = await call(client, "get_claim", {"claim_number": "CLM-NOPE"})
        assert err and "not found" in msg and "Traceback" not in msg
        err, msg = await call(client, "get_claim", {"claim_number": "bad number!"})
        assert err  # pattern constraint


# ------------------------------------------------------------------------------ state machine


def test_state_machine_pure() -> None:
    check_agent_transition(S.SUBMITTED, S.PROCESSING)
    check_agent_transition(S.PROCESSING, S.AWAITING_APPROVAL)
    check_agent_transition(S.DOCUMENTS_REQUESTED, S.PROCESSING)
    check_agent_transition(S.AWAITING_APPROVAL, S.FAILED)
    for old, new in [
        (S.SUBMITTED, S.AWAITING_APPROVAL),
        (S.APPROVED, S.FAILED),
        (S.FAILED, S.PROCESSING),
    ]:
        with pytest.raises(TransitionError):
            check_agent_transition(old, new)
    for old in S:
        for final in (S.APPROVED, S.PARTIALLY_APPROVED, S.REJECTED):
            with pytest.raises(TransitionError, match="human"):
                check_agent_transition(old, final)


async def test_update_status_rejects_bad_and_human_only_transitions(session: AsyncSession) -> None:
    world = await make_world(session)
    claim = await make_claim(session, world)
    server = create_claims_db(deps_for(session, write_enabled=True, mcp_caller_id="decision-agent"))
    async with connected(server) as client:
        err, msg = await call(
            client,
            "update_claim_status",
            {
                "claim_number": "CLM-T1",
                "new_status": "awaiting_approval",
                "reason": "skip",
                "idempotency_key": KEY + "1",
            },
        )
        assert err and "from 'submitted' to 'awaiting_approval'" in msg and "processing" in msg

        await ClaimRepository(session).update_status(
            claim, S.AWAITING_APPROVAL, actor_type=ActorType.SYSTEM
        )
        for i, status in enumerate(["approved", "partially_approved", "rejected"]):
            err, msg = await call(
                client,
                "update_claim_status",
                {
                    "claim_number": "CLM-T1",
                    "new_status": status,
                    "reason": "try",
                    "idempotency_key": f"{KEY}-h{i}",
                },
            )
            assert err and "human reviewer" in msg
        assert claim.status == S.AWAITING_APPROVAL
        changes = await audit_for(session, "claim.status_changed")
        assert all(r.actor_type == ActorType.SYSTEM for r in changes)


async def test_update_status_is_idempotent_and_audited(session: AsyncSession) -> None:
    world = await make_world(session)
    claim = await make_claim(session, world)
    server = create_claims_db(deps_for(session, write_enabled=True, mcp_caller_id="intake-agent"))
    args = {
        "claim_number": "CLM-T1",
        "new_status": "processing",
        "reason": "intake started",
        "idempotency_key": KEY,
    }
    async with connected(server) as client:
        err, first = await call(client, "update_claim_status", args)
        err2, second = await call(client, "update_claim_status", args)
        assert not err and not err2
        assert first["idempotent_replay"] is False and second["idempotent_replay"] is True
        assert {k: v for k, v in first.items() if k != "idempotent_replay"} == {
            k: v for k, v in second.items() if k != "idempotent_replay"
        }
        # same key, different arguments: refused
        err, msg = await call(client, "update_claim_status", {**args, "reason": "something else"})
        assert err and "different arguments" in msg

    assert claim.status == S.PROCESSING
    changes = await audit_for(session, "claim.status_changed")
    assert len(changes) == 1  # not applied twice
    assert changes[0].actor_type == ActorType.AGENT and changes[0].actor_id == "intake-agent"
    assert changes[0].details["reason"] == "intake started"
    rows = (await session.execute(select(func.count()).select_from(ToolIdempotency))).scalar_one()
    assert rows == 1


async def test_save_decision_draft(session: AsyncSession) -> None:
    world = await make_world(session)
    claim = await make_claim(session, world)
    server = create_claims_db(deps_for(session, write_enabled=True, mcp_caller_id="decision-agent"))
    args = {
        "claim_number": "CLM-T1",
        "decision": "partial_approve",
        "payable_amount": "65000.50",
        "deductions": [{"description": "Room rent above cap", "amount": "15000", "clause": "4.2"}],
        "reasons": ["Room rent capped"],
        "cited_clauses": [{"chunk_id": str(world.chunk.id)}],
        "risk_flags": [],
        "agent_run_id": "run-1",
        "idempotency_key": KEY,
    }
    async with connected(server) as client:
        err, msg = await call(client, "save_decision_draft", args)
        assert err and "'processing'" in msg  # claim is still 'submitted'

        await ClaimRepository(session).update_status(
            claim, S.PROCESSING, actor_type=ActorType.SYSTEM
        )
        err, first = await call(client, "save_decision_draft", args)
        err2, second = await call(client, "save_decision_draft", args)
        assert not err and not err2
        assert first["decision_id"] == second["decision_id"]
        assert second["idempotent_replay"] is True

        bad = {**args, "idempotency_key": KEY + "x", "cited_clauses": [{"chunk_id": "not-a-uuid"}]}
        err, msg = await call(client, "save_decision_draft", bad)
        assert err and "valid id" in msg
        err, msg = await call(
            client,
            "save_decision_draft",
            {**args, "idempotency_key": KEY + "y", "payable_amount": "-5"},
        )
        assert err

    decisions = (await session.execute(select(ClaimDecision))).scalars().all()
    assert len(decisions) == 1
    assert (
        decisions[0].cited_clauses[0]["section_path"] == world.chunk.section_path
    )  # from our records
    assert decisions[0].payable_amount == Decimal("65000.50")
    assert claim.status == S.PROCESSING  # a draft changes no status
    drafts = await audit_for(session, "decision.draft_saved")
    assert len(drafts) == 1 and drafts[0].actor_type == ActorType.AGENT
    assert drafts[0].actor_id == "decision-agent"


# ------------------------------------------------------------------------------ notify


def test_notify_helpers() -> None:
    assert sanitize_extra({"to": "x@y.com", "email": "x", "reason": "r" * 500}) == {
        "reason": "r" * 300
    }
    assert mask_recipient(NotificationChannel.EMAIL, "asha@example.com") == "a***@example.com"


async def test_notify_uses_customer_record_not_arguments(session: AsyncSession) -> None:
    world = await make_world(session)
    await make_claim(session, world)
    server = create_notify(deps_for(session, write_enabled=True, mcp_caller_id="notify-agent"))
    async with connected(server) as client:
        err, out = await call(
            client,
            "send_status_update",
            {
                "claim_number": "CLM-T1",
                "channel": "email",
                "template": "received",
                "idempotency_key": KEY,
                "extra": {
                    "to": "evil@attacker.com",
                    "recipient": "evil@attacker.com",
                    "phone": "1",
                },
            },
        )
        assert not err
        assert out["recipient"] == "a***@example.com"  # the customer on record, masked
        err, replay = await call(
            client,
            "send_status_update",
            {
                "claim_number": "CLM-T1",
                "channel": "email",
                "template": "received",
                "idempotency_key": KEY,
            },
        )
        assert (
            replay["idempotent_replay"] is True
            and replay["notification_id"] == out["notification_id"]
        )

        err, sms = await call(
            client,
            "send_status_update",
            {
                "claim_number": "CLM-T1",
                "channel": "sms",
                "template": "received",
                "idempotency_key": KEY + "2",
            },
        )
        assert not err and sms["recipient"].endswith("3210")

    rows = (
        (await session.execute(select(Notification).order_by(Notification.created_at)))
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert all("evil" not in r.rendered_message for r in rows)
    assert "Asha" in rows[0].rendered_message and "CLM-T1" in rows[0].rendered_message
    assert rows[0].status.value == "sent_mock"
    sent = await audit_for(session, "notification.sent_mock")
    assert len(sent) == 2 and {r.actor_id for r in sent} == {"notify-agent"}
    assert all(r.actor_type == ActorType.AGENT for r in sent)


async def test_notify_status_gate_and_rate_limit(session: AsyncSession) -> None:
    world = await make_world(session)
    claim = await make_claim(session, world)
    server = create_notify(deps_for(session, write_enabled=True, notify_max_per_claim_per_hour=2))

    def args(template: str, n: int, **extra):
        return {
            "claim_number": "CLM-T1",
            "channel": "email",
            "template": template,
            "idempotency_key": f"{KEY}-{n}",
            **extra,
        }

    async with connected(server) as client:
        err, msg = await call(client, "send_status_update", args("approved", 0))
        assert err and "not 'approved'" in msg  # no "approved" message before a human approves

        claim.status, claim.approved_amount = S.DOCUMENTS_REQUESTED, None
        err, _ = await call(
            client,
            "send_status_update",
            args("documents_requested", 1, extra={"missing_documents": ["Discharge summary"]}),
        )
        assert not err
        err, _ = await call(client, "send_status_update", args("received", 2))
        assert not err
        err, msg = await call(client, "send_status_update", args("received", 3))
        assert err and "Rate limit" in msg

    body = (
        await session.execute(
            select(Notification).where(Notification.template == "documents_requested")
        )
    ).scalar_one()
    assert "Discharge summary" in body.rendered_message


# ------------------------------------------------------------------------------ HTTP auth


def test_token_parsing() -> None:
    env = {
        "MCP_CLAIMS_DB_TOKENS": "risk-agent=aaa, decision-agent=bbb",
        "MCP_CLAIMS_DB_TOKEN": "ccc",
    }
    assert token_map("claims_db", env) == {
        "ccc": "claims_db-client",
        "aaa": "risk-agent",
        "bbb": "decision-agent",
    }
    assert identities("notify", {}) == {}
    assert token_for("claims_db", "risk-agent", env) == "aaa"
    with pytest.raises(KeyError):
        token_for("claims_db", "nobody", env)


INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _http_app(session: AsyncSession, tokens: dict[str, str]):
    app = create_claims_db(
        deps_for(session, write_enabled=True), host="0.0.0.0"
    ).streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, tokens=tokens)
    return app


async def test_http_rejects_missing_or_wrong_token(session: AsyncSession) -> None:
    app = _http_app(session, {"good-token": "risk-agent"})
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mcp"
        ) as http:
            for headers in (
                HEADERS,
                {**HEADERS, "Authorization": "Bearer wrong"},
                {**HEADERS, "Authorization": "Basic good-token"},
                {**HEADERS, "Authorization": "Bearer "},
            ):
                resp = await http.post("/mcp", json=INIT, headers=headers)
                assert resp.status_code == 401
            ok = await http.post(
                "/mcp", json=INIT, headers={**HEADERS, "Authorization": "Bearer good-token"}
            )
            assert ok.status_code == 200
            assert (await http.get("/healthz")).status_code == 200


async def test_http_token_identity_reaches_audit_log(session: AsyncSession) -> None:
    world = await make_world(session)
    await make_claim(session, world)
    app = _http_app(session, {"tok-risk": "risk-agent"})
    auth = {**HEADERS, "Authorization": "Bearer tok-risk"}

    def rpc_body(resp: httpx.Response) -> dict:
        text = resp.text
        data = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        return json.loads(data[-1] if data else text)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mcp"
        ) as http:
            init = await http.post("/mcp", json=INIT, headers=auth)
            sid = {"mcp-session-id": init.headers["mcp-session-id"], **auth}
            await http.post(
                "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=sid
            )
            resp = await http.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "update_claim_status",
                        "arguments": {
                            "claim_number": "CLM-T1",
                            "new_status": "processing",
                            "reason": "risk check",
                            "idempotency_key": KEY,
                        },
                    },
                },
                headers=sid,
            )
            assert rpc_body(resp)["result"]["isError"] is False

    change = (await audit_for(session, "claim.status_changed"))[0]
    assert (change.actor_type, change.actor_id) == (ActorType.AGENT, "risk-agent")
