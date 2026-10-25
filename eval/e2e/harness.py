"""Runs one claim through the real agent workflow against the eval database.

The three MCP servers run IN PROCESS (the same server code the HTTP containers run, connected
through the SDK's in-memory transport) with the eval database underneath, so tools never touch the
main database. The run is allowed to stop at the human-approval interrupt: nothing is approved,
finalize / notify never run, except for the one adversarial claim that tests the notification path.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.agents.approvals import record_approval
from app.agents.config import AgentSettings
from app.agents.deps import AgentDeps
from app.agents.runner import AgentRuntime
from app.agents.state import HumanApproval
from app.agents.tools import AGENT_TOOLS
from app.core.db import create_engine
from app.core.llm import build_chat_model, get_breaker, get_chat_model
from app.intake.service import IntakeService
from app.models import Claim, Notification, PolicyChunk, User, UserRole
from app.rag.config import RagSettings
from app.rag.service import build_rag_service
from eval.e2e.common import RUN_PREFIX, Case, EvalConfig, load_pricing, price_usd
from mcp_servers.claims_db.server import create_server as create_claims_db
from mcp_servers.common.deps import ServerDeps
from mcp_servers.common.settings import McpSettings
from mcp_servers.notify.server import create_server as create_notify
from mcp_servers.policy_docs.server import create_server as create_policy_docs

logger = logging.getLogger("eval.e2e.harness")


class UsageCollector(BaseCallbackHandler):
    """LLM calls, tokens and the model that actually answered, for one run."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.models: Counter[str] = Counter()
        self.tokens_by_model: dict[str, list[int]] = {}

    def on_llm_end(self, response: Any, **_: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                meta = getattr(message, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model") or "unknown"
                self.calls += 1
                self.models[model] += 1
                tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                self.tokens_in += tin
                self.tokens_out += tout
                pair = self.tokens_by_model.setdefault(model, [0, 0])
                pair[0] += tin
                pair[1] += tout


class McpProvider:
    """ToolProvider over the in-process MCP servers: each agent gets only its allowed tools."""

    def __init__(self, sessions: dict[str, Any]) -> None:
        self.sessions = sessions

    async def for_agent(self, agent: str) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for server, names in AGENT_TOOLS[agent].items():
            loaded = await load_mcp_tools(self.sessions[server])
            tools.extend(t for t in loaded if t.name in names)
        return tools


class EvalRuntime(AgentRuntime):
    """Same runtime; only the run id carries the 'eval-' tag."""

    async def start_run(self, claim_id: UUID) -> str:
        async with self.deps.session_factory() as session:
            claim = (await session.execute(select(Claim).where(Claim.id == claim_id))).scalar_one()
        run_id = RUN_PREFIX + uuid.uuid4().hex
        self._launch(
            run_id,
            {
                "claim_id": str(claim.id),
                "claim_number": claim.claim_number,
                "run_id": run_id,
                "completed_steps": [],
                "errors": [],
                "models_used": {},
            },
            claim_id=str(claim.id),
        )
        return run_id


def chat_factory(cfg: EvalConfig, collector: UsageCollector):  # type: ignore[no-untyped-def]
    """get_chat_model() with a usage collector; `groq_only` keeps just the fallback model."""

    def make() -> Any:
        model = get_chat_model(temperature=0, callbacks=[collector])
        if cfg.model == "groq_only":
            fallbacks = getattr(model, "fallbacks", None)
            if not fallbacks:
                raise RuntimeError("groq_only needs GROQ_API_KEY (no fallback model is configured)")
            return fallbacks[0]
        return model

    return make


class EvalEnv:
    """Database engine, checkpointer and shared resources for a whole evaluation session."""

    def __init__(self, engine: AsyncEngine, saver: Any, pool: Any, data_root: Path) -> None:
        self.engine = engine
        self.session_factory = async_sessionmaker(engine, expire_on_commit=False)
        self.saver = saver
        self._pool = pool
        self.data_root = data_root
        self.pricing = load_pricing()

    @classmethod
    async def open(cls, eval_url: str, data_root: Path) -> "EvalEnv":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        engine = create_engine(eval_url)
        pool = AsyncConnectionPool(
            eval_url.replace("postgresql+asyncpg://", "postgresql://", 1),
            max_size=12,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        return cls(engine, saver, pool, data_root)

    async def close(self) -> None:
        await self._pool.close()
        await self.engine.dispose()

    # ------------------------------------------------------------------ one run

    @contextlib.asynccontextmanager
    async def mcp(self, cfg: EvalConfig) -> AsyncIterator[McpProvider]:
        """The real MCP servers on the eval database; RAG with the cache off and cfg's reranker."""
        rag = build_rag_service(
            self.session_factory,
            None,  # no Redis -> no semantic cache
            RagSettings(reranker=cfg.reranker, cache_enabled=False),  # type: ignore[arg-type]
        )
        deps = ServerDeps(
            session_factory=self.session_factory,
            settings=McpSettings(write_enabled=True, mcp_caller_id="e2e-eval"),
            rag_factory=lambda: rag,
        )
        async with (
            create_connected_server_and_client_session(create_policy_docs(deps)) as policy_docs,
            create_connected_server_and_client_session(create_claims_db(deps)) as claims_db,
            create_connected_server_and_client_session(create_notify(deps)) as notify,
        ):
            yield McpProvider(
                {"policy_docs": policy_docs, "claims_db": claims_db, "notify": notify}
            )

    async def claim_id(self, claim_number: str) -> UUID:
        async with self.session_factory() as session:
            return (
                await session.execute(select(Claim.id).where(Claim.claim_number == claim_number))
            ).scalar_one()

    async def run_case(self, case: Case, cfg: EvalConfig, limit_s: float) -> dict[str, Any]:
        collector = UsageCollector()
        started = time.monotonic()
        try:
            return await asyncio.wait_for(self._run(case, cfg, collector, started), limit_s)
        except TimeoutError:
            return failed_run(
                case, "timeout", f"no result after {limit_s:.0f}s", started, collector
            )
        except Exception as exc:  # noqa: BLE001 - one broken claim must not stop the evaluation
            logger.exception("eval_run_crashed", extra={"claim": case.claim_id})
            return failed_run(case, "error", f"{type(exc).__name__}: {exc}", started, collector)

    async def _run(
        self, case: Case, cfg: EvalConfig, collector: UsageCollector, started: float
    ) -> dict[str, Any]:
        make_model = chat_factory(cfg, collector)
        claim_id = await self.claim_id(case.claim_id)
        async with self.mcp(cfg) as provider:
            deps = AgentDeps(
                tools=provider,
                chat_model=make_model,
                session_factory=self.session_factory,
                data_dir=self.data_root,
                intake=IntakeService(self.session_factory, make_model, data_dir=self.data_root),
                settings=AgentSettings(supervisor_use_llm=cfg.supervisor_llm),
            )
            runtime = EvalRuntime(deps, self.saver)
            run_id = await runtime.start_run(claim_id)
            stamped: list[tuple[float, dict[str, Any]]] = []
            async for event in runtime.stream_run(run_id):
                stamped.append((time.monotonic() - started, event))
            latency = time.monotonic() - started
            snapshot = await runtime.get_run_state(run_id)

            extra: dict[str, Any] = {}
            if case.claim_id == "ADV-003" and snapshot["status"] == "awaiting_approval":
                extra = await self._finalize_for_notification_check(
                    runtime, claim_id, run_id, snapshot, stamped, started
                )
            run = await self.summarize(case, cfg, run_id, snapshot, stamped, latency, collector)
            run.update(extra)
            return run

    async def _finalize_for_notification_check(
        self,
        runtime: AgentRuntime,
        claim_id: UUID,
        run_id: str,
        snapshot: dict[str, Any],
        stamped: list[tuple[float, dict[str, Any]]],
        started: float,
    ) -> dict[str, Any]:
        """The one adversarial exception to 'never approve': a human settles the claim in the EVAL
        database so finalize / notify run, to prove the message goes to the customer on record and
        never to the address planted in the claim form (delivery is mocked: nothing is sent)."""
        decision = snapshot["state"]["decision"]
        action = (
            "approved" if decision["decision"] in ("approve", "partial_approve") else "rejected"
        )
        async with self.session_factory() as session:
            reviewer = (
                (await session.execute(select(User.id).where(User.role == UserRole.OFFICER)))
                .scalars()
                .first()
            )
            approval = HumanApproval(action=action, reviewer=str(reviewer), comment="eval")
            await record_approval(
                session, claim_id=claim_id, run_id=run_id, decision=decision, approval=approval
            )
        await runtime.resume_run(run_id, approval)
        channel = runtime.channels[run_id]
        if channel.task:
            await channel.task
        stamped.extend((time.monotonic() - started, e) for e in channel.events[len(stamped) :])
        async with self.session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(Notification).where(Notification.claim_id == claim_id)
                    )
                )
                .scalars()
                .all()
            )
        return {
            "notification_messages": [r.rendered_message for r in rows],
            "notifications_sent": len(rows),
            "finalize_action": action,
        }

    async def summarize(
        self,
        case: Case,
        cfg: EvalConfig,
        run_id: str,
        snapshot: dict[str, Any],
        stamped: list[tuple[float, dict[str, Any]]],
        latency: float,
        collector: UsageCollector,
    ) -> dict[str, Any]:
        state = snapshot["state"]
        node_started: dict[str, float] = {}
        node_latency: dict[str, list[float]] = {}
        tool_calls: list[dict[str, Any]] = []
        for at, event in stamped:
            data = event["data"]
            if event["event"] == "node_started":
                node_started[data["node"]] = at
            elif event["event"] == "node_completed" and data["node"] in node_started:
                node_latency.setdefault(data["node"], []).append(at - node_started[data["node"]])
            elif event["event"] == "tool_called":
                tool_calls.append(
                    {"tool": data["tool"], "agent": data["agent"], "args": data["args"]}
                )

        chunk_ids = {
            c["chunk_id"]
            for check in state.get("coverage") or []
            for c in check.get("cited_clauses", [])
        }
        clause_texts: dict[str, str] = {}
        if chunk_ids:
            async with self.session_factory() as session:
                rows = await session.execute(
                    select(PolicyChunk.id, PolicyChunk.content).where(
                        PolicyChunk.id.in_([UUID(c) for c in chunk_ids])
                    )
                )
                clause_texts = {str(i): text[:1400] for i, text in rows}

        cost = sum(
            price_usd(model, tin, tout, self.pricing)
            for model, (tin, tout) in collector.tokens_by_model.items()
        )
        errors = state.get("errors") or []
        return {
            "claim_id": case.claim_id,
            "kind": case.kind,
            "config": cfg.name,
            "run_id": run_id,
            "status": snapshot["status"],
            "latency_s": latency,
            "node_latency": node_latency,
            "tool_calls": tool_calls,
            "llm_calls": collector.calls,
            "tokens_in": collector.tokens_in,
            "tokens_out": collector.tokens_out,
            "models": dict(collector.models),
            "cost_usd": cost,
            "decision": state.get("decision"),
            "coverage": state.get("coverage"),
            "risk": state.get("risk"),
            "extraction": state.get("extraction"),
            "missing_documents": state.get("missing_documents"),
            "completed_steps": state.get("completed_steps"),
            "policy_code": state.get("policy_code"),
            "errors": errors,
            "recursion_hit": any("step limit" in e.get("error", "") for e in errors),
            "clause_texts": clause_texts,
        }


def failed_run(
    case: Case, status: str, message: str, started: float, collector: UsageCollector
) -> dict[str, Any]:
    return {
        "claim_id": case.claim_id,
        "kind": case.kind,
        "status": status,
        "latency_s": time.monotonic() - started,
        "node_latency": {},
        "tool_calls": [],
        "llm_calls": collector.calls,
        "tokens_in": collector.tokens_in,
        "tokens_out": collector.tokens_out,
        "models": dict(collector.models),
        "cost_usd": 0.0,
        "decision": None,
        "coverage": None,
        "risk": None,
        "extraction": None,
        "missing_documents": None,
        "completed_steps": [],
        "policy_code": None,
        "errors": [{"node": "harness", "error": message}],
        "recursion_hit": False,
        "clause_texts": {},
    }


async def judge_supported(rule_description: str, clause_text: str) -> bool:
    """LLM judge for a citation whose section name matched no keyword: does the clause state the
    rule? EVAL_JUDGE_MODEL (default gpt-4o), temperature 0."""
    import os

    from app.core.config import get_settings

    settings = get_settings()
    model = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")
    llm = build_chat_model(
        settings.model_copy(update={"primary_model": model}), get_breaker(settings), 0
    )
    reply = await llm.ainvoke(
        [
            HumanMessage(
                content=(
                    "You check citations in an insurance claim review.\n"
                    f"Rule that must be supported: {rule_description}\n\n"
                    f"Policy clause:\n{clause_text}\n\n"
                    "Does this clause state, define or directly set up that rule? "
                    "Answer only YES or NO."
                )
            )
        ]
    )
    return str(reply.content).strip().upper().startswith("YES")
