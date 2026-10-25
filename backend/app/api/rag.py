"""Retrieval + grounded answers (officers and admins; the LLM endpoints are rate limited)."""

import json
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.auth.deps import llm_guard, require_officer
from app.core.logging import bind_log_context
from app.observability.tracing import run_trace, trace_id_for
from app.rag.answerer import AnswerGenerationError
from app.rag.schemas import PolicyAnswerResult, QueryRequest, RetrievedChunk, SearchRequest
from app.rag.service import RetrievalService, build_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/policies", tags=["rag"], dependencies=[Depends(require_officer)])


def get_rag_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        try:
            service = build_rag_service(request.app.state.session_factory, request.app.state.redis)
        except Exception as exc:  # e.g. no OPENAI_API_KEY: report, don't crash
            logger.exception("rag_service_unavailable")
            raise HTTPException(503, "retrieval service is not configured") from exc
        request.app.state.rag_service = service
    return service


@router.post("/search")
async def search(
    body: SearchRequest, service: RetrievalService = Depends(get_rag_service)
) -> list[RetrievedChunk]:
    """Hybrid search + rerank, no LLM answer. Shows vector_rank, keyword_rank, fused_score and
    rerank_score so retrieval quality can be inspected."""
    return await service.search(body.query, body.policy_codes, body.top_n)


def _query_trace(body: QueryRequest, stream: bool, seed: str | None = None):  # type: ignore[no-untyped-def]
    """One `policy_query` trace per question (its own trace type, not a claim run)."""
    return run_trace(
        "policy_query",
        seed=seed or uuid4().hex,
        tags=["scenario:live", *(f"policy:{c}" for c in body.policy_codes or [])],
        metadata={"stream": stream, "use_cache": body.use_cache},
        input={"query": body.query},
    )


@router.post("/query", dependencies=[Depends(llm_guard)])
async def query(
    request: Request,
    body: QueryRequest,
    service: RetrievalService = Depends(get_rag_service),
) -> PolicyAnswerResult:
    with _query_trace(body, stream=False) as trace, bind_log_context(trace_id=trace.trace_id):
        request.state.trace_id = trace.trace_id
        try:
            result = await service.answer(
                body.query, body.policy_codes, body.chat_history, use_cache=body.use_cache
            )
        except AnswerGenerationError as exc:
            raise HTTPException(502, f"could not generate an answer: {exc}") from exc
        trace.update(
            output={
                "model": result.model_used,
                "cache_hit": result.cache_hit,
                "citations": len(result.citations),
                "confidence": result.confidence,
            }
        )
        return result


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream", dependencies=[Depends(llm_guard)])
async def query_stream(
    request: Request,
    body: QueryRequest,
    service: RetrievalService = Depends(get_rag_service),
) -> StreamingResponse:
    """Server-sent events: `citations` (retrieved clauses), `token`*, `done` (validated
    citations + timings) or `error`."""

    seed = uuid4().hex

    async def events() -> AsyncIterator[str]:
        with (
            _query_trace(body, stream=True, seed=seed) as trace,
            bind_log_context(trace_id=trace.trace_id),
        ):
            async for item in service.answer_stream(
                body.query, body.policy_codes, body.chat_history, use_cache=body.use_cache
            ):
                if item.event == "done" and isinstance(item.data, dict):
                    trace.update(output={"model": item.data.get("model_used")})
                yield _sse(item.event, item.data)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-ID": trace_id_for(seed),
        },
    )
