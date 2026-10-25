"""Node plumbing: error guard, retry policy, short summaries for events."""

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, RetryPolicy

from app.agents.config import AgentSettings
from app.agents.events import emit
from app.agents.state import ClaimState
from app.agents.tools import is_transient

logger = logging.getLogger(__name__)

NodeFn = Callable[[ClaimState], Awaitable[dict[str, Any] | Command]]


def retry_policy(settings: AgentSettings) -> RetryPolicy:
    """Transient LLM / MCP errors are retried by LangGraph before the error node sees them."""
    return RetryPolicy(
        max_attempts=settings.node_retry_attempts,
        initial_interval=settings.node_retry_initial_interval,
        backoff_factor=2.0,
        retry_on=is_transient,
    )


def failure_update(node: str, error: str) -> dict[str, Any]:
    return {"errors": [{"node": node, "error": error[:300]}], "completed_steps": [f"{node}:failed"]}


def guarded(name: str, fn: NodeFn) -> NodeFn:
    """Announce the node, and turn a non-transient exception into a hop to the error node.

    Transient errors are re-raised so the node's RetryPolicy can retry them; if they persist the
    exception leaves the graph and the runner ends the run on the error path. GraphBubbleUp
    (interrupt) always passes through untouched."""

    @functools.wraps(fn)
    async def node(state: ClaimState) -> dict[str, Any] | Command:
        emit({"type": "node_started", "node": name})
        try:
            return await fn(state)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if is_transient(exc):
                logger.warning(
                    "node_transient_error", extra={"node": name, "error": type(exc).__name__}
                )
                raise
            logger.exception("node_failed", extra={"node": name})
            return Command(
                goto="error", update=failure_update(name, f"{type(exc).__name__}: {exc}")
            )

    return node


def short(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
