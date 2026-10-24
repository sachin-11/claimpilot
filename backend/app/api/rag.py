"""Retrieval + grounded answers. No auth yet (Module 10)."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.rag.answerer import AnswerGenerationError
from app.rag.schemas import PolicyAnswerResult, QueryRequest, RetrievedChunk, SearchRequest
from app.rag.service import RetrievalService, build_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/policies", tags=["rag"])


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


@router.post("/query")
async def query(
    body: QueryRequest, service: RetrievalService = Depends(get_rag_service)
) -> PolicyAnswerResult:
    try:
        return await service.answer(
            body.query, body.policy_codes, body.chat_history, use_cache=body.use_cache
        )
    except AnswerGenerationError as exc:
        raise HTTPException(502, f"could not generate an answer: {exc}") from exc


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query/stream")
async def query_stream(
    body: QueryRequest, service: RetrievalService = Depends(get_rag_service)
) -> StreamingResponse:
    """Server-sent events: `citations` (retrieved clauses), `token`*, `done` (validated
    citations + timings) or `error`."""

    async def events() -> AsyncIterator[str]:
        async for item in service.answer_stream(
            body.query, body.policy_codes, body.chat_history, use_cache=body.use_cache
        ):
            yield _sse(item.event, item.data)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
