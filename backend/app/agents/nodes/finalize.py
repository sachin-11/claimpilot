"""Finalize node. The approval API (not an agent) has already written claim_approvals and the final
claim status. This node only tells the customer and closes the run."""

import logging
from typing import Any

from app.agents.deps import AgentDeps
from app.agents.outcome import final_outcome
from app.agents.state import ClaimState, HumanApproval
from app.agents.tools import is_transient, toolbox_for

logger = logging.getLogger(__name__)


def make_finalize_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def finalize(state: ClaimState) -> dict[str, Any]:
        approval = HumanApproval.model_validate(state["approval"])
        status, _ = final_outcome(state["decision"], approval)
        template = status.value  # received | documents_requested | approved | ... (same names)

        extra: dict[str, Any] = {}
        if template == "documents_requested":
            extra["missing_documents"] = state.get("missing_documents") or []
        # the reviewer's comment is an internal note: it is never sent to the customer

        tools = await toolbox_for(deps.tools, "finalize")
        try:
            sent = await tools.call(
                "send_status_update",
                {
                    "claim_number": state["claim_number"],
                    "channel": deps.settings.notify_channel,
                    "template": template,
                    "extra": extra or None,
                    "idempotency_key": f"{state['run_id']}-notify",
                },
            )
            notification = {"status": sent.get("status"), "template": template}
        except Exception as exc:  # noqa: BLE001
            if is_transient(exc):
                raise  # let the RetryPolicy try again
            # The human decision is already final; a failed message must not undo or fail it.
            logger.warning(
                "notify_failed", extra={"run_id": state["run_id"], "error": type(exc).__name__}
            )
            notification = {"status": "failed", "template": template, "error": str(exc)[:200]}
        return {"notification": notification, "completed_steps": ["finalize"]}

    return finalize
