"""Helpers for the MCP server tests: rows in the rolled-back test DB, an in-memory MCP client, and a
fake RetrievalService (no LLM / embedding / network)."""

import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Customer, CustomerPolicy, Policy, PolicyChunk
from app.rag.schemas import Citation, PolicyAnswerResult, RetrievedChunk
from app.repositories.claim import ClaimRepository
from app.schemas.claim import ClaimCreate
from mcp_servers.common.deps import ServerDeps
from mcp_servers.common.settings import McpSettings


def mcp_settings(**overrides: Any) -> McpSettings:
    return McpSettings(_env_file=None, **overrides)


def deps_for(session: AsyncSession, rag: Any = None, **settings: Any) -> ServerDeps:
    """Deps whose DB session is the shared, rolled-back test session (never closed here)."""

    @contextlib.asynccontextmanager
    async def shared() -> AsyncIterator[AsyncSession]:
        yield session

    return ServerDeps(
        session_factory=shared,
        settings=mcp_settings(**settings),
        rag_factory=(lambda: rag) if rag is not None else None,
    )


@dataclass
class World:
    policy: Policy
    chunk: PolicyChunk
    customer: Customer
    customer_policy: CustomerPolicy


async def make_world(session: AsyncSession, *, code: str = "POL-T") -> World:
    policy = Policy(
        policy_code=code, name="Test Policy", insurer="Test Insurer", rules={"waiting": {"ped": 36}}
    )
    session.add(policy)
    await session.flush()
    chunk = PolicyChunk(
        policy_id=policy.id,
        chunk_index=0,
        content="6.3 Cataract is covered after a waiting period of 24 months.",
        section_path="6 Waiting Period > 6.3. Specific Waiting Period",
        page_start=7,
        page_end=7,
        content_hash=uuid.uuid4().hex,
        meta={"clauses": ["6.3"], "embed_text": "x" * 50},
    )
    customer = Customer(full_name="Asha Verma", email="asha.verma@example.com", phone="9876543210")
    session.add_all([chunk, customer])
    await session.flush()
    cp = CustomerPolicy(
        customer_id=customer.id,
        policy_id=policy.id,
        policy_number=f"PN-{uuid.uuid4().hex[:8]}",
        sum_insured=Decimal("500000"),
        start_date=date(2024, 1, 1),
    )
    session.add(cp)
    await session.flush()
    return World(policy, chunk, customer, cp)


async def make_claim(session: AsyncSession, world: World, number: str = "CLM-T1", **fields: Any):
    return await ClaimRepository(session).create(
        ClaimCreate(
            claim_number=number,
            customer_policy_id=world.customer_policy.id,
            hospital_name="City Hospital",
            admission_date=date(2025, 3, 1),
            discharge_date=date(2025, 3, 4),
            claimed_amount=Decimal("80000"),
            **fields,
        )
    )


@contextlib.asynccontextmanager
async def connected(server: FastMCP):
    """An MCP client session wired to `server` in memory."""
    async with create_connected_server_and_client_session(server) as client:
        yield client


async def call(client: Any, name: str, args: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """(is_error, data). Data is the structured result, or the error text."""
    result = await client.call_tool(name, args or {})
    if result.isError:
        return True, " ".join(getattr(c, "text", "") for c in result.content)
    if result.structuredContent is not None:
        data = result.structuredContent
        return False, data["result"] if set(data) == {"result"} else data
    return False, [json.loads(c.text) for c in result.content]


class FakeRag:
    """Stands in for RetrievalService."""

    def __init__(self, chunk: PolicyChunk | None = None) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.chunk = chunk

    async def search(self, query, policy_codes=None, top_n=None, section_prefix=None):
        self.search_calls.append({"query": query, "policy_codes": policy_codes, "top_n": top_n})
        return [
            RetrievedChunk(
                chunk_id=self.chunk.id if self.chunk else uuid.uuid4(),
                policy_code="POL-T",
                section_path="6 Waiting Period > 6.3. Specific Waiting Period",
                page_start=7,
                page_end=7,
                content="6.3 Cataract waiting period is 24 months. " + "x" * 5000,
                fused_score=0.03,
                rerank_score=0.91,
            )
        ]

    async def answer(self, query, policy_codes=None, chat_history=None, use_cache=True):
        return PolicyAnswerResult(
            answer="Cataract has a 24 month waiting period [1].",
            citations=[
                Citation(chunk_id=uuid.uuid4(), section_path="6.3", supporting_text="24 months")
            ],
            insufficient_context=False,
            confidence="high",
            chunks_used=[],
        )
