"""Start / stream / resume / inspect agent runs.

A run executes in a background task and publishes events to a per-run channel, so the SSE
endpoint is just a reader: a client that connects late replays the buffered events, one that never
connects does not stop the run. The checkpoint (thread_id = run_id) is the source of truth: after
a backend restart the channel is gone, but the paused run can still be inspected and resumed, and
the stream endpoint rebuilds its last event from the checkpoint."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
from app.models import Claim

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
    done: bool = False
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        async with self.changed:
            self.events.append({"seq": len(self.events), "event": event, "data": data})
            self.changed.notify_all()

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
    async def create(cls, deps: AgentDeps, conninfo: str) -> "AgentRuntime":
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
        return cls(deps, saver, closer=pool.close)

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

    async def start_run(self, claim_id: UUID) -> str:
        async with self.deps.session_factory() as session:
            claim = (
                await session.execute(select(Claim).where(Claim.id == claim_id))
            ).scalar_one_or_none()
        if claim is None:
            raise RunNotFoundError(f"claim {claim_id} not found")
        run_id = uuid.uuid4().hex
        initial: ClaimState = {
            "claim_id": str(claim.id),
            "claim_number": claim.claim_number,
            "run_id": run_id,
            "completed_steps": [],
            "errors": [],
            "models_used": {},
        }
        self._launch(run_id, initial, claim_id=str(claim.id))
        return run_id

    async def resume_run(self, run_id: str, approval: HumanApproval) -> None:
        snapshot = await self.get_run_state(run_id)
        if snapshot["status"] != "awaiting_approval":
            raise RunStateError(f"run {run_id} is '{snapshot['status']}', not awaiting approval")
        self._launch(run_id, Command(resume=approval.model_dump(mode="json")))

    def _launch(self, run_id: str, run_input: Any, claim_id: str | None = None) -> None:
        channel = self.channels.get(run_id)
        if channel is None or channel.done:
            old = channel.events if channel else []
            claim_id = claim_id or (channel.claim_id if channel else None)
            channel = RunChannel(events=old, claim_id=claim_id)
            self.channels[run_id] = channel
        channel.done = False
        channel.task = asyncio.create_task(self._drive(run_id, run_input, channel))

    # ------------------------------------------------------------------ execution

    async def _drive(self, run_id: str, run_input: Any, channel: RunChannel) -> None:
        config = self._config(run_id)
        try:
            async for mode, data in self.graph.astream(run_input, config, stream_mode=STREAM_MODES):
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
            await channel.finish()

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
