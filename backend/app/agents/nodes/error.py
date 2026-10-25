"""Error node: end the run gracefully. Marks the claim `failed` through MCP (the state machine
refuses this for a claim a human has already settled; that is logged, not raised)."""

import logging
from typing import Any

from app.agents.deps import AgentDeps
from app.agents.events import emit
from app.agents.state import ClaimState
from app.agents.tools import ToolFailure, toolbox_for

logger = logging.getLogger(__name__)


def make_error_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def error(state: ClaimState) -> dict[str, Any]:
        errors = state.get("errors") or []
        reason = "; ".join(f"{e['node']}: {e['error']}" for e in errors)[:480] or "unknown error"
        emit({"type": "node_started", "node": "error"})
        try:
            tools = await toolbox_for(deps.tools, "supervisor")
            await tools.call(
                "update_claim_status",
                {
                    "claim_number": state["claim_number"],
                    "new_status": "failed",
                    "reason": reason,
                    "idempotency_key": f"{state['run_id']}-failed",
                },
            )
        except ToolFailure as exc:
            logger.warning("claim_not_marked_failed", extra={"error": str(exc)[:200]})
        except Exception:  # noqa: BLE001 - the run must still end cleanly
            logger.exception("claim_not_marked_failed")
        return {"completed_steps": ["error"]}

    return error
