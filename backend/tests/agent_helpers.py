"""Helpers for the agent tests: the REAL MCP servers (claims_db, policy_docs, notify) connected
in memory to the rolled-back test DB, a scripted fake LLM, and claim data builders. No real LLM,
network or MCP process."""

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp_helpers import FakeRag, connected, deps_for, shared_session_factory
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.config import AgentSettings
from app.agents.deps import AgentDeps
from app.agents.nodes.coverage import CoverageCheckLLM, CoverageLLMOutput
from app.agents.nodes.decision import ReasonsLLM
from app.agents.nodes.risk import RiskLLMOutput
from app.agents.supervisor import SupervisorChoice
from app.agents.tools import AGENT_TOOLS
from app.models import (
    ClaimExtraction,
    Customer,
    CustomerPolicy,
    Policy,
    PolicyChunk,
    User,
    UserRole,
)
from app.repositories.claim import ClaimRepository
from app.schemas.claim import ClaimCreate
from mcp_servers.claims_db.server import create_server as create_claims_db
from mcp_servers.notify.server import create_server as create_notify
from mcp_servers.policy_docs.server import create_server as create_policy_docs

# The Arogya Sanjeevani numbers (data_gen/policy_rules.yaml), as stored in policies.rules.
AS_RULES: dict[str, Any] = {
    "room_rent_limit": {"percent_of_si_per_day": 2.0, "max_per_day": 5000},
    "icu_limit": {"percent_of_si_per_day": 5.0, "max_per_day": 10000},
    "co_pay_percent": 5,
    "initial_waiting_days": 30,
    "specific_disease_waiting": {"cataract": 24, "hernia": 24},
    "pre_existing_waiting_months": 48,
    "sub_limits": {"cataract": {"amount": 40000, "unit": "per_eye"}},
    "exclusions": ["cosmetic_surgery", "dental_unless_accident", "obesity_treatment"],
    "maternity": {"covered": False, "waiting_months": 0},
    "proportionate_categories": ["room_rent", "professional_fees", "ot_charges"],
}

# An appendectomy with a Rs 8,000/day room under a Rs 5,000/day limit (see test_agents_pure).
ROOM_RENT_EXTRACTION: dict[str, Any] = {
    "diagnosis_text": "Acute Appendicitis",
    "icd10_code": "K35.80",
    "procedure": "Laparoscopic Appendectomy",
    "admission_date": "2025-03-01",
    "discharge_date": "2025-03-04",
    "room_type": "Deluxe",
    "room_rent_per_day": "8000.00",
    "icu_days": None,
    "line_items": [
        {"description": "Room rent", "category": "room", "amount": "24000.00"},
        {"description": "Surgeon", "category": "surgeon", "amount": "20000.00"},
        {"description": "OT", "category": "ot", "amount": "10000.00"},
        {"description": "Medicines", "category": "medicines", "amount": "12000.00"},
    ],
    "taxes": "1200.00",
    "bill_total": "67200.00",
    "claimed_amount": "67200.00",
}


@dataclass
class AgentWorld:
    policy: Policy
    chunk: PolicyChunk
    customer: Customer
    cp: CustomerPolicy
    claim: Any
    reviewer: User


async def make_agent_world(
    session: AsyncSession,
    *,
    extraction: dict[str, Any] | None = None,
    missing_documents: list[str] | None = None,
    issues: list[dict[str, Any]] | None = None,
    rules: dict[str, Any] | None = None,
    start: date = date(2024, 1, 1),
    number: str = "CLM-A1",
    with_policy: bool = True,
    policy_code: str = "POL-T",
) -> AgentWorld:
    policy = Policy(
        policy_code=policy_code, name="Test Policy", insurer="Test", rules=rules or AS_RULES
    )
    chunk = PolicyChunk(
        policy_id=None,
        chunk_index=0,
        content="6.3 Cataract and hernia have a waiting period of 24 months.",
        section_path="6 Waiting Period > 6.3. Specific Waiting Period",
        page_start=8,
        page_end=9,
        content_hash=uuid.uuid4().hex,
        meta={"clauses": ["6.3"]},
    )
    customer = Customer(full_name="Asha Verma", email="asha.verma@example.com", phone="9876543210")
    reviewer = User(
        email=f"rev-{uuid.uuid4().hex[:6]}@example.com",
        full_name="Reviewer",
        role=UserRole.OFFICER,
        password_hash="x",
    )
    session.add_all([policy, customer, reviewer])
    await session.flush()
    chunk.policy_id = policy.id
    session.add(chunk)
    cp = CustomerPolicy(
        customer_id=customer.id,
        policy_id=policy.id,
        policy_number=f"PN-{uuid.uuid4().hex[:8]}",
        sum_insured=Decimal("500000"),
        start_date=start,
    )
    session.add(cp)
    await session.flush()
    ext = extraction or ROOM_RENT_EXTRACTION
    claim = await ClaimRepository(session).create(
        ClaimCreate(
            claim_number=number,
            customer_policy_id=cp.id if with_policy else None,
            hospital_name="City Hospital",
            admission_date=date.fromisoformat(ext["admission_date"]),
            claimed_amount=Decimal(ext["claimed_amount"]),
        )
    )
    data = {
        **ext,
        "missing_documents": missing_documents or [],
        "consistency_issues": issues or [],
        "patient_name": "Asha Verma",
    }
    session.add(
        ClaimExtraction(
            claim_id=claim.id, extracted_data=data, confidence=0.9, validation_errors=[], version=1
        )
    )
    await session.flush()
    return AgentWorld(policy, chunk, customer, cp, claim, reviewer)


# ----------------------------------------------------------------------------- fake LLM


class _Structured:
    def __init__(self, llm: "ScriptedLLM", schema: type) -> None:
        self.llm, self.schema = llm, schema

    async def ainvoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self.llm.calls.append(self.schema.__name__)
        spec = self.llm.structured.get(self.schema)
        if spec is None:
            raise RuntimeError(f"no scripted answer for {self.schema.__name__}")
        parsed = spec(messages) if callable(spec) else spec
        raw = AIMessage(content="", response_metadata={"model_name": self.llm.model_name})
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class _WithTools:
    def __init__(self, llm: "ScriptedLLM", tools: list[BaseTool]) -> None:
        self.llm = llm
        llm.bound_tool_names.append(sorted(t.name for t in tools))

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.llm.calls.append("react")
        if not self.llm.react:
            return AIMessage(content="done")
        step = self.llm.react.pop(0)
        return step(messages) if callable(step) else step


@dataclass
class ScriptedLLM:
    structured: dict[type, Any] = field(default_factory=dict)
    react: list[Any] = field(default_factory=list)
    model_name: str = "fake-model"
    calls: list[str] = field(default_factory=list)
    bound_tool_names: list[list[str]] = field(default_factory=list)

    def with_structured_output(self, schema: type, **_: Any) -> _Structured:
        return _Structured(self, schema)

    def bind_tools(self, tools: list[BaseTool]) -> _WithTools:
        return _WithTools(self, tools)


def search_call(query: str = "waiting period", call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_policy",
                "args": {"query": query, "policy_code": "POL-T"},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def passing_coverage(
    chunk_id: str, *, extra: list[CoverageCheckLLM] | None = None
) -> CoverageLLMOutput:
    """The checks relevant to ROOM_RENT_EXTRACTION under AS_RULES: initial waiting, room rent, co-pay."""
    cited = [chunk_id]
    return CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(
                check="initial_waiting", applies=True, result="pass",
                explanation="Admitted long after the 30-day initial period.", chunk_ids=cited,
            ),
            CoverageCheckLLM(
                check="room_rent", applies=True, result="partial",
                explanation="Room rent is above the daily limit.", chunk_ids=cited,
            ),
            CoverageCheckLLM(
                check="co_pay", applies=True, result="pass",
                explanation="A 5% co-payment applies.", chunk_ids=cited,
            ),
            *(extra or []),
        ]
    )  # fmt: skip


def standard_llm(chunk_id: str, overrides: dict[type, Any] | None = None) -> ScriptedLLM:
    structured: dict[type, Any] = {
        SupervisorChoice: lambda messages: SupervisorChoice(
            next_steps=["coverage", "risk"], reason="both are ready"
        ),
        CoverageLLMOutput: passing_coverage(chunk_id),
        RiskLLMOutput: RiskLLMOutput(summary="No risk signals.", extra_flags=[]),
        ReasonsLLM: ReasonsLLM(reasons=["Room rent is above the policy limit."]),
        **(overrides or {}),
    }
    return ScriptedLLM(structured=structured, react=[search_call(), AIMessage(content="done")])


# ----------------------------------------------------------------------------- MCP tools


class McpProvider:
    """ToolProvider over the real MCP servers connected in memory. Each agent is handed only the
    tools AGENT_TOOLS allows, exactly like the HTTP provider."""

    def __init__(self, sessions: dict[str, Any]) -> None:
        self.sessions = sessions
        self.requested: list[str] = []

    async def for_agent(self, agent: str) -> list[BaseTool]:
        self.requested.append(agent)
        tools: list[BaseTool] = []
        for server, names in AGENT_TOOLS[agent].items():
            loaded = await load_mcp_tools(self.sessions[server])
            tools.extend(t for t in loaded if t.name in names)
        return tools


@contextlib.asynccontextmanager
async def mcp_world(
    session: AsyncSession, rag: FakeRag, *, write_enabled: bool = True
) -> AsyncIterator[McpProvider]:
    """policy_docs + claims_db + notify, in memory, sharing the test DB session."""
    deps = deps_for(session, rag=rag, write_enabled=write_enabled, mcp_caller_id="agent-under-test")
    async with (
        connected(create_policy_docs(deps)) as policy_docs,
        connected(create_claims_db(deps)) as claims_db,
        connected(create_notify(deps)) as notify,
    ):
        yield McpProvider({"policy_docs": policy_docs, "claims_db": claims_db, "notify": notify})


def agent_deps(
    session: AsyncSession,
    provider: Any,
    llm: ScriptedLLM,
    **settings: Any,
) -> AgentDeps:
    return AgentDeps(
        tools=provider,
        chat_model=lambda: llm,
        session_factory=shared_session_factory(session),
        data_dir=Path("/nonexistent"),
        settings=AgentSettings(_env_file=None, **settings),
    )


def initial_state(world: AgentWorld, run_id: str = "run-0001-abcdefgh") -> dict[str, Any]:
    return {
        "claim_id": str(world.claim.id),
        "claim_number": world.claim.claim_number,
        "run_id": run_id,
        "completed_steps": [],
        "errors": [],
        "models_used": {},
    }


_ = (Callable,)  # keep the import for type hints in tests
