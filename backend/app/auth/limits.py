"""Redis rate limits (fixed one-minute window) and the daily LLM budget guard.

Both fail OPEN when Redis is unreachable (a limiter must not take the API down); the login lockout
in `service.py` is the one control that does not. Off when APP_ENV=test unless a test enables it."""

import logging
from typing import Any

from fastapi import HTTPException, Request

from app.core.config import Settings
from app.observability.usage import read_spend, seconds_until_tomorrow, today

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


def limits_active(settings: Settings) -> bool:
    if not settings.rate_limit_enabled:
        return False
    return not (settings.app_env == "test" and not settings.rate_limit_in_tests)


def client_ip(request: Request) -> str:
    """The peer address. Behind a reverse proxy run uvicorn with --proxy-headers and
    --forwarded-allow-ips so this is the real client, not the proxy."""
    return request.client.host if request.client else "unknown"


async def hit(redis: Any, scope: str, ident: str, limit: int) -> int | None:
    """Count one request. None = allowed, else the seconds to wait (Retry-After)."""
    key = f"rl:{scope}:{ident}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, WINDOW_SECONDS)
        if count <= limit:
            return None
        return max(int(await redis.ttl(key)), 1)
    except Exception:  # noqa: BLE001 - fail open
        logger.warning("rate_limit_unavailable", exc_info=True)
        return None


def too_many(retry_after: int, what: str) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f"rate limit exceeded ({what}); retry in {retry_after} s",
        headers={"Retry-After": str(retry_after)},
    )


async def enforce(request: Request, settings: Settings, scope: str, ident: str, limit: int) -> None:
    if not limits_active(settings):
        return
    wait = await hit(request.app.state.redis, scope, ident, limit)
    if wait is not None:
        logger.warning("rate_limited", extra={"scope": scope, "retry_after": wait})
        raise too_many(wait, scope)


async def ensure_budget(request: Request, settings: Settings) -> None:
    """Refuse new LLM work once today's estimated spend reached DAILY_LLM_BUDGET_INR."""
    if settings.daily_llm_budget_inr <= 0 or not limits_active(settings):
        return
    try:
        spent = (await read_spend(request.app.state.redis, today()))["inr"]
    except Exception:  # noqa: BLE001 - fail open
        logger.warning("budget_check_unavailable", exc_info=True)
        return
    if spent >= settings.daily_llm_budget_inr:
        wait = seconds_until_tomorrow()
        logger.error(  # the alert: one line per refused request
            "llm_budget_exceeded",
            extra={"spent_inr": spent, "budget_inr": settings.daily_llm_budget_inr},
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"the daily LLM budget of Rs {settings.daily_llm_budget_inr:g} is used up "
                f"(Rs {spent:.2f} spent today); new runs resume tomorrow or ask an admin"
            ),
            headers={"Retry-After": str(wait)},
        )
