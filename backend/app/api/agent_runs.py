"""Agent runs on a claim: start, stream (SSE), inspect, and the human approval."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.approvals import ApprovalConflictError, record_approval
from app.agents.deps import AgentDeps
from app.agents.outcome import ApprovalError
from app.agents.runner import AgentRuntime, RunNotFoundError, RunStateError
from app.agents.state import HumanApproval
from app.agents.tools import McpToolProvider
from app.api.claims import get_intake_service
from app.core.config import get_settings
from app.core.db import get_session
from app.core.llm import get_chat_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claims", tags=["agents"])

_build_lock = asyncio.Lock()


def checkpoint_conninfo() -> str:
    from app.agents.config import get_agent_settings

    explicit = get_agent_settings().checkpoint_database_url
    url = explicit or get_settings().database_url
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_runtime(request: Request) -> AgentRuntime:
    """The process-wide runtime. Built on first use (and at startup outside tests), so a missing
    MCP server or database never stops the API from booting."""
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is not None:
        return runtime
    async with _build_lock:
        runtime = getattr(request.app.state, "agent_runtime", None)
        if runtime is None:
            runtime = await build_runtime(request.app)
            request.app.state.agent_runtime = runtime
    return runtime


async def build_runtime(app) -> AgentRuntime:  # type: ignore[no-untyped-def]
    class _Request:  # get_intake_service only needs .app
        pass

    holder = _Request()
    holder.app = app  # type: ignore[attr-defined]
    deps = AgentDeps(
        tools=McpToolProvider(),
        chat_model=lambda: get_chat_model(temperature=0),
        session_factory=app.state.session_factory,
        data_dir=get_settings().data_dir,
        intake=get_intake_service(holder),  # type: ignore[arg-type]
    )
    return await AgentRuntime.create(deps, checkpoint_conninfo())


class ProcessResponse(BaseModel):
    claim_id: UUID
    run_id: str


class ApproveRequest(BaseModel):
    action: Literal["approved", "rejected", "edited"]
    final_amount: Annotated[Decimal | None, Field(ge=0, le=Decimal("100000000"))] = None
    comment: Annotated[str | None, Field(max_length=1000)] = None
    reviewer_id: UUID


class ApproveResponse(BaseModel):
    run_id: str
    final_status: str
    approved_amount: Decimal | None
    already_recorded: bool
    resumed: bool


@router.post("/{claim_id}/process", status_code=202)
async def process_claim(
    claim_id: UUID, runtime: AgentRuntime = Depends(get_runtime)
) -> ProcessResponse:
    try:
        run_id = await runtime.start_run(claim_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return ProcessResponse(claim_id=claim_id, run_id=run_id)


async def _owned_run(runtime: AgentRuntime, claim_id: UUID, run_id: str) -> dict:
    try:
        snapshot = await runtime.get_run_state(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if snapshot["claim_id"] != str(claim_id):
        raise HTTPException(404, f"run {run_id} does not belong to this claim")
    return snapshot


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/{claim_id}/runs/{run_id}/stream")
async def stream_run(
    claim_id: UUID, run_id: str, runtime: AgentRuntime = Depends(get_runtime)
) -> StreamingResponse:
    """Server-sent events: node_started, routing, tool_called, node_completed, awaiting_approval
    (the review payload), completed, error. The stream ends when the run pauses or finishes."""
    await _owned_run(runtime, claim_id, run_id)

    async def events() -> AsyncIterator[str]:
        async for item in runtime.stream_run(run_id):
            yield _sse(item["event"], item["data"])

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{claim_id}/runs/{run_id}")
async def get_run(
    claim_id: UUID, run_id: str, runtime: AgentRuntime = Depends(get_runtime)
) -> dict:
    snapshot = await _owned_run(runtime, claim_id, run_id)
    state = snapshot["state"]
    return {
        "run_id": run_id,
        "claim_number": snapshot["claim_number"],
        "status": snapshot["status"],
        "completed_steps": state.get("completed_steps", []),
        "next": snapshot["next"],
        "awaiting_approval": snapshot["interrupt"],
        "decision": state.get("decision"),
        "coverage": state.get("coverage"),
        "risk": state.get("risk"),
        "approval": state.get("approval"),
        "notification": state.get("notification"),
        "errors": state.get("errors", []),
        "models_used": state.get("models_used", {}),
    }


@router.post("/{claim_id}/runs/{run_id}/approve", status_code=202)
async def approve_run(
    claim_id: UUID,
    run_id: str,
    body: ApproveRequest,
    runtime: AgentRuntime = Depends(get_runtime),
    session: AsyncSession = Depends(get_session),
) -> ApproveResponse:
    """The human decision. Writes claim_approvals, the final claim status and the audit log in one
    transaction, THEN resumes the paused graph (finalize + notify). If the resume fails, calling
    this again with the same body retries the resume without writing a second approval."""
    snapshot = await _owned_run(runtime, claim_id, run_id)
    decision = snapshot["state"].get("decision")
    if snapshot["status"] != "awaiting_approval" or decision is None:
        raise HTTPException(409, f"run is '{snapshot['status']}', not awaiting approval")
    try:
        approval = HumanApproval(
            action=body.action,
            final_amount=body.final_amount,
            comment=body.comment,
            reviewer=str(body.reviewer_id),
        )
    except ValidationError as exc:
        raise HTTPException(422, "; ".join(e["msg"] for e in exc.errors())) from exc

    try:
        status, amount, already = await record_approval(
            session, claim_id=claim_id, run_id=run_id, decision=decision, approval=approval
        )
    except ApprovalError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ApprovalConflictError as exc:
        raise HTTPException(409, str(exc)) from exc

    resumed = False
    try:
        await runtime.resume_run(run_id, approval)
        resumed = True
    except RunStateError:
        logger.info("approve_resume_skipped", extra={"run_id": run_id})
    return ApproveResponse(
        run_id=run_id,
        final_status=status.value,
        approved_amount=amount,
        already_recorded=already,
        resumed=resumed,
    )
