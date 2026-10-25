"""The agent runtime as the API process sees it when a worker does the running.

It reads checkpoints like any `AgentRuntime` (so a run paused at the human approval can be
inspected and resumed from any process), but `start_run` / `resume_run` only enqueue a job, and
status and events come from Redis instead of from an in-process task."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.agents.runner import AgentRuntime, RunNotFoundError
from app.worker.bus import TERMINAL_EVENTS, RunBus

logger = logging.getLogger(__name__)

JOB_START = "run_agent"
START_EPOCH, RESUME_EPOCH = 0, 1
STREAM_WAIT_SECONDS = 600  # how long a client may wait for a queued run to produce events


class QueuedAgentRuntime(AgentRuntime):
    def __init__(self, *args: Any, queue: Any, bus: RunBus, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.queue = queue  # an arq pool
        self.bus = bus

    async def _dispatch(
        self, kind: str, run_id: str, payload: dict[str, Any], meta: dict[str, Any]
    ) -> None:
        epoch = START_EPOCH if kind == "start" else RESUME_EPOCH
        await self.bus.set_meta(run_id, state="queued", epoch=epoch, kind=kind, **meta)
        # One job per run and phase: enqueueing the same one twice is a no-op (None comes back)
        job = await self.queue.enqueue_job(
            JOB_START, kind, run_id, payload, meta, _job_id=f"{run_id}:{epoch}"
        )
        logger.info(
            "run_enqueued",
            extra={"run_id": run_id, "kind": kind, "duplicate": job is None},
        )

    async def get_run_state(self, run_id: str, *, inside_run: bool = False) -> dict[str, Any]:
        meta = await self.bus.get_meta(run_id)
        busy = meta.get("state") in {"queued", "running"} and not inside_run
        try:
            snapshot = await super().get_run_state(run_id, inside_run=True)
        except RunNotFoundError:
            if not meta:
                raise
            return {  # enqueued, the worker has not written a checkpoint yet
                "run_id": run_id,
                "claim_id": meta.get("claim_id"),
                "claim_number": meta.get("claim_number") or None,
                "status": meta["state"] if busy else "stalled",
                "next": [],
                "interrupt": None,
                "state": {},
            }
        if busy and not self._phase_finished(meta, snapshot):
            snapshot["status"] = "running" if meta["state"] == "running" else "queued"
        return snapshot

    @staticmethod
    def _phase_finished(meta: dict[str, str], snapshot: dict[str, Any]) -> bool:
        """The checkpoint already shows the end of the current phase although the worker has not yet
        marked the job idle (the terminal event is published a moment before): the run is paused
        at the approval (phase 0) or done (phase 1). Without this a client that reacts to the
        `awaiting_approval` event would still read "running"."""
        epoch = int(meta.get("epoch") or 0)
        if epoch == START_EPOCH:
            return snapshot["interrupt"] is not None or snapshot["status"] == "failed"
        return snapshot["status"] in {"completed", "failed"}

    async def stream_run(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Events of the current phase from the Redis stream, until it pauses or ends. A run that is
        not busy and has no events (events expired) is replayed from its checkpoint."""
        meta = await self.bus.get_meta(run_id)
        if not meta:
            async for item in super().stream_run(run_id):  # a run from before the worker existed
                yield item
            return
        epoch = int(meta.get("epoch") or 0)
        last_id, seq = "0-0", 0
        deadline = time.monotonic() + STREAM_WAIT_SECONDS
        while time.monotonic() < deadline:
            entries = await self.bus.read(run_id, epoch, last_id, block_ms=1000)
            for entry_id, event, data in entries:
                last_id = entry_id
                yield {"seq": seq, "event": event, "data": data}
                seq += 1
                if event in TERMINAL_EVENTS:
                    return
            if not entries:
                current = await self.bus.get_meta(run_id)
                if current.get("state") == "idle" and seq == 0:
                    async for item in super().stream_run(run_id):  # nothing to replay
                        yield item
                    return
                if current.get("state") == "idle":
                    return  # a phase that ended without a terminal event
                await asyncio.sleep(0)
