"""Estimated LLM spend per day, shared by every process through Redis (API and worker).

`record_spend` runs inside the LangChain callback (synchronous), so it uses a small synchronous
Redis client; the budget guard and the admin usage endpoint read the same hash asynchronously.
A day is an Indian calendar day (the budget is in INR). Everything fails soft: if Redis is
unreachable the call is simply not counted, never an error for the LLM call itself."""

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
KEEP_DAYS = 10
_tracking = True
_client: Any = None


def disable_tracking() -> None:
    """Evaluation runs call this: their spend must not eat the live daily budget."""
    global _tracking
    _tracking = False


def today(now: datetime | None = None) -> str:
    return (now or datetime.now(IST)).astimezone(IST).strftime("%Y-%m-%d")


def seconds_until_tomorrow(now: datetime | None = None) -> int:
    now = (now or datetime.now(IST)).astimezone(IST)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(int((midnight - now).total_seconds()), 1)


def spend_key(day: str) -> str:
    return f"llm_spend:{day}"


def _sync_client() -> Any:
    global _client
    if _client is None:
        import redis  # noqa: PLC0415

        from app.core.config import get_settings  # noqa: PLC0415

        _client = redis.Redis.from_url(
            get_settings().redis_url, socket_timeout=0.5, socket_connect_timeout=0.5
        )
    return _client


def record_spend(*, provider: str, model: str, inr: float, tokens_in: int, tokens_out: int) -> None:
    if not _tracking:
        return
    try:
        client = _sync_client()
        key = spend_key(today())
        pipe = client.pipeline()
        pipe.hincrbyfloat(key, "inr", inr)
        pipe.hincrby(key, "calls", 1)
        pipe.hincrby(key, "tokens_in", tokens_in)
        pipe.hincrby(key, "tokens_out", tokens_out)
        pipe.hincrbyfloat(key, f"inr:{provider}:{model}", inr)
        pipe.hincrby(key, f"calls:{provider}", 1)
        pipe.expire(key, KEEP_DAYS * 86400)
        pipe.execute()
    except Exception:  # noqa: BLE001 - spend tracking must never break an LLM call
        logger.warning("spend_record_failed", exc_info=True)


def runs_key(day: str) -> str:
    return f"agent_runs:{day}"


def record_run(status: str) -> None:
    """One agent run ended in `status` today (shared by the API and the worker)."""
    if not _tracking:
        return
    try:
        key = runs_key(today())
        pipe = _sync_client().pipeline()
        pipe.hincrby(key, status, 1)
        pipe.expire(key, KEEP_DAYS * 86400)
        pipe.execute()
    except Exception:  # noqa: BLE001 - counting must never break a run
        logger.warning("run_record_failed", exc_info=True)


async def read_runs(redis: Any, day: str) -> dict[str, int]:
    raw = await redis.hgetall(runs_key(day))
    return {
        (k.decode() if isinstance(k, bytes) else k): int(v.decode() if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


async def read_spend(redis: Any, day: str) -> dict[str, Any]:
    """The day's totals: inr, calls, tokens_in, tokens_out and the INR spend by provider:model."""
    raw = await redis.hgetall(spend_key(day))
    data = {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}

    def num(name: str, cast: Any) -> Any:
        value = data.get(name)
        value = value.decode() if isinstance(value, bytes) else value
        return cast(value) if value is not None else cast(0)

    by_model = {
        name.removeprefix("inr:"): round(float(v.decode() if isinstance(v, bytes) else v), 4)
        for name, v in data.items()
        if name.startswith("inr:")
    }
    calls_by_provider = {
        name.removeprefix("calls:"): int(v.decode() if isinstance(v, bytes) else v)
        for name, v in data.items()
        if name.startswith("calls:")
    }
    return {
        "day": day,
        "calls_by_provider": calls_by_provider,
        "inr": round(num("inr", float), 4),
        "calls": num("calls", int),
        "tokens_in": num("tokens_in", int),
        "tokens_out": num("tokens_out", int),
        "by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1])),
    }
