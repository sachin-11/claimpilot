"""Hybrid supervisor: hard guards decide which steps are ALLOWED; the LLM may only choose among
them. Why hybrid: the model adds flexibility (it can skip or order allowed steps), while the guards
give hard guarantees that no step runs before its inputs exist (intake first, coverage only with
complete documents, decision only after the join). If the model fails or answers with a step that
is not allowed, the deterministic default (run everything allowed) is used. Both are logged."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.deps import AgentDeps
from app.agents.events import emit
from app.agents.llm import ask_structured
from app.agents.prompts import load_prompt
from app.agents.state import ClaimState

logger = logging.getLogger(__name__)

WORKERS = ("intake", "coverage", "risk", "decision")


class SupervisorChoice(BaseModel):
    next_steps: list[str] = Field(description="one or more of the allowed steps")
    reason: str


def allowed_steps(state: ClaimState) -> list[str]:
    """Steps that may run now, in default order. Empty = nothing left for the supervisor."""
    done = {s for s in state.get("completed_steps") or [] if ":" not in s}
    if state.get("errors"):
        return []
    if "intake" not in done:
        return ["intake"]
    if "decision" in done:
        return []
    # Missing documents or no policy: coverage and risk cannot say anything useful.
    if state.get("missing_documents") or not state.get("policy_code"):
        return ["decision"]
    pending = [s for s in ("coverage", "risk") if s not in done]
    return pending or ["decision"]


def resolve_choice(allowed: list[str], choice: SupervisorChoice | None) -> tuple[list[str], str]:
    """The LLM's pick if it is valid (non-empty, only allowed steps), else the default order."""
    if choice is None:
        return list(allowed), "default_order (supervisor LLM unavailable)"
    picked = list(dict.fromkeys(choice.next_steps))
    if picked and all(step in allowed for step in picked):
        return picked, "llm"
    return list(allowed), f"default_order (llm chose {choice.next_steps}, not allowed)"


def make_supervisor_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def supervisor(state: ClaimState) -> Command:
        allowed = allowed_steps(state)
        if not allowed:
            if state.get("errors"):
                # the failing node already routed to the error node; just stop this branch
                return Command(goto=END)
            return Command(goto=END)

        choice: SupervisorChoice | None = None
        model_name: str | None = None
        if len(allowed) > 1 and deps.settings.supervisor_use_llm:
            try:
                choice, model_name = await ask_structured(
                    deps.chat_model(),
                    SupervisorChoice,
                    [
                        SystemMessage(content=load_prompt("supervisor")),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "allowed_steps": allowed,
                                    "completed_steps": state.get("completed_steps") or [],
                                    "missing_documents": state.get("missing_documents") or [],
                                },
                            )
                        ),
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - fall back to the deterministic order
                logger.warning("supervisor_llm_failed", extra={"error": type(exc).__name__})
        if len(allowed) == 1:
            chosen, source = list(allowed), "forced (only one step is allowed)"
        else:
            chosen, source = resolve_choice(allowed, choice)
        logger.info(
            "supervisor_route",
            extra={
                "run_id": state.get("run_id"),
                "allowed": allowed,
                "llm_choice": choice.next_steps if choice else None,
                "chosen": chosen,
                "source": source,
            },
        )
        emit({"type": "routing", "allowed": allowed, "chosen": chosen, "source": source})
        update: dict[str, Any] = {"next_steps": chosen}
        if model_name:
            update["models_used"] = {"supervisor": model_name}
        return Command(goto=chosen, update=update)

    return supervisor
