"""Start / stream / resume / inspect agent runs.

A run executes in a background task and publishes events to a per-run channel, so the SSE
endpoint is just a reader: a client that connects late replays the buffered events, one that never
connects does not stop the run. The checkpoint (thread_id = run_id) is the source of truth: after
a backend restart the channel is gone, but the paused run can still be inspected and resumed, and
the stream endpoint rebuilds its last event from the checkpoint."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from sqlalchemy import select

from app.agents.deps import AgentDeps
from app.agents.graph import build_graph
from app.agents.nodes.base import short
from app.agents.nodes.error import make_error_node
from app.agents.state import ClaimState, HumanApproval
from app.core.config import get_settings
from app.core.logging import bind_log_context
from app.models import Claim, CustomerPolicy, Policy
from app.observability.metrics import record_agent_run
from app.observability.tracing import run_trace, trace_id_for

logger = logging.getLogger(__name__)

STREAM_MODES = ["updates", "custom"]


class RunNotFoundError(LookupError):
    pass


class RunStateError(RuntimeError):
    """The run is not in a state that allows the request (e.g. resume of a run not awaiting)."""


@dataclass
class RunChannel:
    events: list[dict[str, Any]] = field(default_factory=list)
    claim_id: str | None = None
    claim_number: str | None = None
    policy_code: str | None = None
    user_id: str | None = None  # who started / resumed the run (the officer), for the trace
    # Where events also go (the Redis stream the API relays to SSE clients); None = memory only.
    sink: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    done: bool = False
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        async with self.changed:
            self.events.append(
                {
                    "seq": len(self.events),
                    "event": event,
                    "data": data,
                    "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                }
            )
            self.changed.notify_all()
        if self.sink is not None:
            try:
                await self.sink(event, data)
            except Exception:  # noqa: BLE001 - a broken event stream must not kill the run
                logger.warning("event_sink_failed", extra={"event": event}, exc_info=True)

    async def finish(self) -> None:
        async with self.changed:
            self.done = True
            self.changed.notify_all()


def summarize_update(node: str, update: dict[str, Any] | None) -> str:
    """One line for a node_completed event."""
    update = update or {}
    if node == "intake":
        return (
            f"policy {update.get('policy_code')}; "
            f"{len(update.get('missing_documents') or [])} missing document(s)"
        )
    if node == "coverage":
        checks = update.get("coverage") or []
        return (
            ", ".join(f"{c['check']}={c['result']}" for c in checks if c.get("applies"))
            or "no checks"
        )
    if node == "risk":
        risk = update.get("risk") or {}
        return f"score {risk.get('score')}; flags: {[f['code'] for f in risk.get('flags', [])]}"
    if node == "decision":
        d = update.get("decision") or {}
        return f"{d.get('decision')}; payable {d.get('payable_amount')}"
    if node == "supervisor":
        return f"next: {update.get('next_steps')}"
    if node == "approval":
        return f"{(update.get('approval') or {}).get('action')}"
    if node == "finalize":
        return short((update.get("notification") or {}).get("status"))
    return ""


class AgentRuntime:
    def __init__(self, deps: AgentDeps, checkpointer: Any, closer: Any = None) -> None:
        self.deps = deps
        self.checkpointer = checkpointer
        self.graph = build_graph(deps, checkpointer)
        self.channels: dict[str, RunChannel] = {}
        self._closer = closer

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    async def create(cls, deps: AgentDeps, conninfo: str, **kwargs: Any) -> "AgentRuntime":
        """Production wiring: AsyncPostgresSaver on the app database; setup() creates its tables."""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        return cls(deps, saver, closer=pool.close, **kwargs)

    async def close(self) -> None:
        for channel in self.channels.values():
            if channel.task and not channel.task.done():
                channel.task.cancel()
        if self._closer is not None:
            await self._closer()

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": self.deps.settings.agent_recursion_limit,
        }

    # ------------------------------------------------------------------ start / resume

    async def claim_and_policy(self, claim_id: UUID) -> tuple[Claim, str | None]:
        """The claim and the code of its policy (None when it has no customer policy)."""
        async with self.deps.session_factory() as session:
            claim = (
                await session.execute(select(Claim).where(Claim.id == claim_id))
            ).scalar_one_or_none()
            if claim is None:
                raise RunNotFoundError(f"claim {claim_id} not found")
            policy_code = None
            if claim.customer_policy_id is not None:
                policy_code = (
                    await session.execute(
                        select(Policy.policy_code)
                        .join(CustomerPolicy, CustomerPolicy.policy_id == Policy.id)
                        .where(CustomerPolicy.id == claim.customer_policy_id)
                    )
                ).scalar_one_or_none()
        return claim, policy_code

    async def start_run(self, claim_id: UUID, user_id: str | None = None) -> str:
        claim, policy_code = await self.claim_and_policy(claim_id)
        run_id = uuid.uuid4().hex
        initial: ClaimState = {
            "claim_id": str(claim.id),
            "claim_number": claim.claim_number,
            "run_id": run_id,
            "completed_steps": [],
            "errors": [],
            "models_used": {},
        }
        meta = {
            "claim_id": str(claim.id),
            "claim_number": claim.claim_number,
            "policy_code": policy_code,
            "user_id": user_id,
        }
        await self._dispatch("start", run_id, dict(initial), meta)
        return run_id

    async def resume_run(self, run_id: str, approval: HumanApproval) -> None:
        snapshot = await self.get_run_state(run_id)
        if snapshot["status"] != "awaiting_approval":
            raise RunStateError(f"run {run_id} is '{snapshot['status']}', not awaiting approval")
        meta = {
            "claim_id": snapshot["claim_id"],
            "claim_number": snapshot["claim_number"],
            "policy_code": (snapshot["state"] or {}).get("policy_code"),
            "user_id": str(approval.reviewer),
        }
        await self._dispatch("resume", run_id, approval.model_dump(mode="json"), meta)

    async def _dispatch(
        self, kind: str, run_id: str, payload: dict[str, Any], meta: dict[str, Any]
    ) -> None:
        """Hand a start / resume to whatever executes it: a background task in this process here,
        a Redis job for the worker in `QueuedAgentRuntime`."""
        run_input = payload if kind == "start" else Command(resume=payload)
        self._launch(run_id, run_input, **meta)

    async def execute(
        self,
        kind: str,
        run_id: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
        *,
        sink: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Run one job's worth of the graph in THIS task and return the run status (the worker).

        Safe to run twice (a retried or re-delivered job): the checkpoint decides. A run that
        already paused or finished is left alone; one that stopped mid-way continues from its last
        checkpoint instead of starting over."""
        before = await self._status_if_started(run_id)
        if kind == "start" and before in {"awaiting_approval", "completed", "failed"}:
            return before
        run_input = await self._input_for(kind, run_id, payload)
        channel = RunChannel(
            claim_id=meta.get("claim_id"),
            claim_number=meta.get("claim_number"),
            policy_code=meta.get("policy_code"),
            user_id=meta.get("user_id"),
            sink=sink,
        )
        self.channels[run_id] = channel
        await self._drive(run_id, run_input, channel)
        return (await self.get_run_state(run_id, inside_run=True))["status"]

    async def _status_if_started(self, run_id: str) -> str | None:
        try:
            return (await self.get_run_state(run_id, inside_run=True))["status"]
        except RunNotFoundError:
            return None

    async def _input_for(self, kind: str, run_id: str, payload: dict[str, Any]) -> Any:
        snapshot = await self.graph.aget_state(self._config(run_id))
        progressed = bool((snapshot.values or {}).get("claim_id"))
        pending = any(i for task in snapshot.tasks for i in task.interrupts)
        if kind == "start":
            return None if progressed else payload  # None = continue from the checkpoint
        return Command(resume=payload) if pending else None

    def _launch(
        self,
        run_id: str,
        run_input: Any,
        claim_id: str | None = None,
        claim_number: str | None = None,
        policy_code: str | None = None,
        user_id: str | None = None,
    ) -> None:
        channel = self.channels.get(run_id)
        if channel is None or channel.done:
            old = channel.events if channel else []
            claim_id = claim_id or (channel.claim_id if channel else None)
            channel = RunChannel(events=old, claim_id=claim_id)
            self.channels[run_id] = channel
        channel.claim_number = claim_number or channel.claim_number
        channel.policy_code = policy_code or channel.policy_code
        channel.user_id = user_id or channel.user_id
        channel.done = False
        channel.task = asyncio.create_task(self._drive(run_id, run_input, channel))

    # ------------------------------------------------------------------ execution

    async def _drive(self, run_id: str, run_input: Any, channel: RunChannel) -> None:
        config = self._config(run_id)
        scenario = "eval" if run_id.startswith("eval-") else "live"
        tags = [f"scenario:{scenario}"]
        if channel.policy_code:
            tags.append(f"policy:{channel.policy_code}")
        resumed = isinstance(run_input, Command)
        started = time.perf_counter()
        with (
            bind_log_context(
                trace_id=trace_id_for(run_id), run_id=run_id, claim_number=channel.claim_number
            ),
            run_trace(
                "claim_run",
                seed=run_id,
                session_id=channel.claim_number,
                user_id=channel.user_id,
                tags=tags,
                metadata={"run_id": run_id, "scenario": scenario, "resumed": resumed},
                input={"claim_number": channel.claim_number, "resumed": resumed},
            ) as trace,
        ):
            config["callbacks"] = trace.callbacks
            try:
                async for mode, data in self.graph.astream(
                    run_input, config, stream_mode=STREAM_MODES
                ):
                    if mode == "custom":
                        await channel.publish(str(data.get("type", "custom")), data)
                        continue
                    for node, update in data.items():
                        if node == "__interrupt__":
                            payload = data[node][0].value
                            await channel.publish("awaiting_approval", payload)
                        elif node != "__start__":
                            await channel.publish(
                                "node_completed",
                                {"node": node, "summary": summarize_update(node, update)},
                            )
                await self._publish_outcome(run_id, channel)
            except GraphRecursionError:
                logger.error("agent_recursion_limit", extra={"run_id": run_id})
                await self._fail_run(run_id, channel, "step limit reached (possible loop)")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never leave a run silently dead
                logger.exception("agent_run_crashed", extra={"run_id": run_id})
                await self._fail_run(run_id, channel, f"{type(exc).__name__}: {exc}")
            finally:
                status = await self._finish_trace(run_id, trace)
                record_agent_run(status, time.perf_counter() - started)
                await channel.finish()

    async def _finish_trace(self, run_id: str, trace: Any) -> str:
        """Outcome and the models that answered, on the root observation of the trace."""
        try:
            snapshot = await self.get_run_state(run_id, inside_run=True)
            state = snapshot.get("state") or {}
            models = state.get("models_used") or {}
            fallback = get_settings().fallback_model
            decision = state.get("decision") or {}
            status = str(snapshot["status"])
            trace.update(
                output={"status": status, "decision": decision.get("decision")},
                metadata={
                    "models_used": models,
                    "fallback_used": any(m == fallback for m in models.values()),
                },
            )
        except Exception:  # noqa: BLE001 - observability must not break the run
            logger.warning("trace_finish_failed", extra={"run_id": run_id}, exc_info=True)
            return "unknown"
        return status

    async def _publish_outcome(self, run_id: str, channel: RunChannel) -> None:
        snapshot = await self.get_run_state(run_id, inside_run=True)
        if snapshot["status"] == "failed":
            await channel.publish("error", {"errors": snapshot["state"].get("errors", [])})
        elif snapshot["status"] == "completed":
            await channel.publish(
                "completed", {"run_id": run_id, "claim_number": snapshot["claim_number"]}
            )

    async def _fail_run(self, run_id: str, channel: RunChannel, reason: str) -> None:
        """The error path for failures that escape the graph (recursion limit, crash): record the
        error in the checkpoint, mark the claim failed through MCP, tell the stream."""
        config = self._config(run_id)
        try:
            await self.graph.aupdate_state(
                config,
                {"errors": [{"node": "graph", "error": reason[:300]}]},
                as_node="supervisor",
            )
            state = (await self.graph.aget_state(config)).values
            await make_error_node(self.deps)(state)  # type: ignore[arg-type]
            await self.graph.aupdate_state(config, {"completed_steps": ["error"]}, as_node="error")
        except Exception:  # noqa: BLE001
            logger.exception("fail_run_cleanup_failed", extra={"run_id": run_id})
        await channel.publish("error", {"errors": [{"node": "graph", "error": reason[:300]}]})

    # ------------------------------------------------------------------ reading

    async def get_run_state(self, run_id: str, *, inside_run: bool = False) -> dict[str, Any]:
        snapshot = await self.graph.aget_state(self._config(run_id))
        values = dict(snapshot.values or {})
        channel = self.channels.get(run_id)
        if not values.get("claim_id"):  # nothing but empty reducer channels yet
            if channel is not None and not channel.done:  # started, no checkpoint written yet
                return {
                    "run_id": run_id,
                    "claim_id": channel.claim_id,
                    "claim_number": None,
                    "status": "running",
                    "next": [],
                    "interrupt": None,
                    "state": {},
                }
            raise RunNotFoundError(f"run {run_id} not found")
        interrupt_payload = next(
            (i.value for task in snapshot.tasks for i in task.interrupts), None
        )
        running = channel is not None and not channel.done and not inside_run
        errors = values.get("errors") or []
        steps = values.get("completed_steps") or []
        if running:
            status = "running"
        elif interrupt_payload is not None:
            status = "awaiting_approval"
        elif errors:
            status = "failed"
        elif "finalize" in steps:
            status = "completed"
        else:
            status = "stalled"  # e.g. the process died mid-run: not resumable from here
        return {
            "run_id": run_id,
            "claim_id": values.get("claim_id"),
            "claim_number": values.get("claim_number"),
            "status": status,
            "next": list(snapshot.next),
            "interrupt": interrupt_payload,
            "state": values,
        }

    async def stream_run(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Buffered events of this process's run, then live ones until the run pauses or ends.
        Without a channel (backend restarted) the current state is replayed as one event."""
        channel = self.channels.get(run_id)
        if channel is None:
            snapshot = await self.get_run_state(run_id)
            kind = {
                "awaiting_approval": "awaiting_approval",
                "completed": "completed",
                "failed": "error",
            }.get(snapshot["status"], "state")
            data = (
                snapshot["interrupt"]
                if kind == "awaiting_approval"
                else {"status": snapshot["status"]}
            )
            yield {"seq": 0, "event": kind, "data": data}
            return
        index = 0
        while True:
            async with channel.changed:
                await channel.changed.wait_for(
                    lambda i=index: i < len(channel.events) or channel.done
                )
                batch = channel.events[index:]
                finished = channel.done
            for event in batch:
                yield event
            index += len(batch)
            if finished and index >= len(channel.events):
                return
