"""Run state and events in Redis, shared by the API and the worker.

- `run:{run_id}:meta` (hash): where the run is (queued / running / idle), who started it. It is what
  lets the API answer "is this run running?" without being the process that runs it.
- `run:{run_id}:events:{epoch}` (stream): the SSE events. Epoch 0 is the first job (up to the human
  approval), epoch 1 the resumed job: a client that connects later replays its epoch from the start,
  and a finished pause never replays into the next phase. A stream (not pub/sub) so late and
  reconnecting clients lose nothing.
Everything expires after a week; the checkpoint in Postgres stays the source of truth."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

KEEP_SECONDS = 7 * 86400
MAX_EVENTS = 2000
TERMINAL_EVENTS = frozenset({"awaiting_approval", "completed", "error"})


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class RunBus:
    def __init__(self, redis: Any) -> None:
        self.redis = redis

    @staticmethod
    def meta_key(run_id: str) -> str:
        return f"run:{run_id}:meta"

    @staticmethod
    def stream_key(run_id: str, epoch: int) -> str:
        return f"run:{run_id}:events:{epoch}"

    async def set_meta(self, run_id: str, **fields: Any) -> None:
        key = self.meta_key(run_id)
        clean = {k: ("" if v is None else str(v)) for k, v in fields.items()}
        await self.redis.hset(key, mapping=clean)
        await self.redis.expire(key, KEEP_SECONDS)

    async def get_meta(self, run_id: str) -> dict[str, str]:
        raw = await self.redis.hgetall(self.meta_key(run_id))
        return {_text(k): _text(v) for k, v in raw.items()}

    async def publish(self, run_id: str, epoch: int, event: str, data: dict[str, Any]) -> None:
        key = self.stream_key(run_id, epoch)
        await self.redis.xadd(
            key,
            {"event": event, "data": json.dumps(data, default=str)},
            maxlen=MAX_EVENTS,
            approximate=True,
        )
        await self.redis.expire(key, KEEP_SECONDS)

    async def read(
        self, run_id: str, epoch: int, last_id: str = "0-0", block_ms: int = 1000
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """(entry id, event, data) after `last_id`; waits up to `block_ms` for something new."""
        reply = await self.redis.xread(
            {self.stream_key(run_id, epoch): last_id}, count=100, block=block_ms
        )
        out: list[tuple[str, str, dict[str, Any]]] = []
        for _, entries in reply or []:
            for entry_id, fields in entries:
                values = {_text(k): _text(v) for k, v in fields.items()}
                out.append((_text(entry_id), values["event"], json.loads(values["data"])))
        return out
