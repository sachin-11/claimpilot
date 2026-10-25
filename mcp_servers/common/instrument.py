"""Caller identity, per-call logging and error mapping for every tool."""

import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

from mcp.server.fastmcp import Context

from mcp_servers.common.auth import CALLER_SCOPE_KEY
from mcp_servers.common.errors import ToolError
from mcp_servers.common.settings import McpSettings

logger = logging.getLogger("mcp.tool")

P = ParamSpec("P")
R = TypeVar("R")


def caller_id(ctx: Context | None, settings: McpSettings) -> str:
    """HTTP: the identity the bearer token maps to. stdio / in-memory: MCP_CALLER_ID."""
    try:
        request = ctx.request_context.request if ctx is not None else None
    except ValueError:  # no active request
        request = None
    if request is not None:
        scope = getattr(request, "scope", {})
        return str(scope.get(CALLER_SCOPE_KEY, "unknown"))
    return settings.mcp_caller_id


def instrumented(
    server: str, settings: McpSettings
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Wrap a tool so each call logs {tool, caller, latency_ms, ok}. Arguments and results are
    never logged (they may hold PII). Unexpected exceptions become a generic ToolError."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            ctx = kwargs.get("ctx")
            caller = caller_id(ctx if isinstance(ctx, Context) else None, settings)
            started = time.perf_counter()
            outcome: dict[str, Any] = {"ok": True}
            try:
                return await fn(*args, **kwargs)
            except ToolError as exc:
                outcome = {"ok": False, "error_kind": "tool_error", "error": str(exc)[:200]}
                raise
            except Exception as exc:
                outcome = {"ok": False, "error_kind": type(exc).__name__}
                logger.exception("tool_crashed", extra={"server": server, "tool": fn.__name__})
                raise ToolError("Internal error while running this tool. Try again later.") from exc
            finally:
                logger.info(
                    "tool_call",
                    extra={
                        "server": server,
                        "tool": fn.__name__,
                        "caller": caller,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        **outcome,
                    },
                )

        return wrapper

    return decorator
