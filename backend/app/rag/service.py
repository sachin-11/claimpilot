"""RetrievalService: the one interface later modules (MCP tools, agents, API) call."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.llm import get_chat_model, get_embeddings
from app.rag.answerer import Answerer, AnswerGenerationError
from app.rag.cache import SemanticCache
from app.rag.config import RagSettings, get_rag_settings
from app.rag.reranker import Reranker, build_reranker
from app.rag.retriever import Retriever, merge_ranked
from app.rag.rewriter import QueryRewriter
from app.rag.schemas import ChatTurn, PolicyAnswerResult, RetrievedChunk, StreamEvent

logger = logging.getLogger(__name__)

STREAM_WORDS_PER_EVENT = 6  # only used to replay a cached answer as tokens


class QueryEmbedder(Protocol):
    async def aembed_query(self, text: str) -> list[float]: ...


@dataclass
class _Prepared:
    """Everything known before an answer is generated."""

    chunks: list[RetrievedChunk]
    standalone: str
    rewritten_query: str | None
    embedding: list[float] | None
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    cached: PolicyAnswerResult | None = None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RetrievalService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embeddings: QueryEmbedder,
        chat_model_factory: Callable[[], object],
        reranker: Reranker,
        cache: SemanticCache | None,
        settings: RagSettings,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.reranker = reranker
        self.cache = cache
        self.retriever = Retriever(session_factory, settings)
        self.rewriter = QueryRewriter(chat_model_factory, settings)
        self.answerer = Answerer(chat_model_factory, settings)

    # ------------------------------------------------------------------ public interface

    async def search(
        self,
        query: str,
        policy_codes: Sequence[str] | None = None,
        top_n: int | None = None,
        section_prefix: str | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid search + rerank, no LLM answer. This is what MCP tools and agents call."""
        started = time.perf_counter()
        prepared = await self._prepare(
            query,
            policy_codes,
            history=None,
            top_n=top_n or self.settings.rerank_top_n,
            use_cache=False,
            section_prefix=section_prefix,
        )
        self._log("search", query, policy_codes, prepared, started, cache_hit=False, model=None)
        return prepared.chunks

    async def answer(
        self,
        query: str,
        policy_codes: Sequence[str] | None = None,
        chat_history: list[ChatTurn] | None = None,
        use_cache: bool = True,
    ) -> PolicyAnswerResult:
        started = time.perf_counter()
        prepared = await self._prepare(
            query,
            policy_codes,
            history=chat_history,
            top_n=self.settings.rerank_top_n,
            use_cache=use_cache,
        )
        if prepared.cached is not None:
            result = self._from_cache(prepared)
            self._log("answer", query, policy_codes, prepared, started, cache_hit=True, model=None)
            return result

        generate_start = time.perf_counter()
        outcome = await self.answerer.answer(prepared.standalone, prepared.chunks)
        prepared.timings["generate"] = _ms(generate_start)

        result = self._result(prepared, outcome.answer, outcome.model_used)
        await self._cache_store(policy_codes, prepared, result, use_cache)
        self._log(
            "answer",
            query,
            policy_codes,
            prepared,
            started,
            cache_hit=False,
            model=result.model_used,
        )
        return result

    async def answer_stream(
        self,
        query: str,
        policy_codes: Sequence[str] | None = None,
        chat_history: list[ChatTurn] | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Events: citations (retrieved clauses) -> token* -> done (validated citations, timings).
        Any failure becomes a final `error` event."""
        started = time.perf_counter()
        try:
            prepared = await self._prepare(
                query,
                policy_codes,
                history=chat_history,
                top_n=self.settings.rerank_top_n,
                use_cache=use_cache,
            )
            yield StreamEvent(event="citations", data=self._citation_metadata(prepared))

            if prepared.cached is not None:
                result = self._from_cache(prepared)
                words = result.answer.split(" ")
                for i in range(0, len(words), STREAM_WORDS_PER_EVENT):
                    piece = " ".join(words[i : i + STREAM_WORDS_PER_EVENT])
                    yield StreamEvent(
                        event="token",
                        data=piece + (" " if i + STREAM_WORDS_PER_EVENT < len(words) else ""),
                    )
                self._log(
                    "stream", query, policy_codes, prepared, started, cache_hit=True, model=None
                )
                yield StreamEvent(event="done", data=result.model_dump(mode="json"))
                return

            generate_start = time.perf_counter()
            stream = self.answerer.stream(prepared.standalone, prepared.chunks)
            async for token in stream.tokens():
                if "first_token" not in prepared.timings:
                    prepared.timings["first_token"] = _ms(generate_start)
                yield StreamEvent(event="token", data=token)
            prepared.timings["generate"] = _ms(generate_start)

            outcome = stream.outcome
            assert outcome is not None
            result = self._result(prepared, outcome.answer, outcome.model_used)
            await self._cache_store(policy_codes, prepared, result, use_cache)
            self._log(
                "stream",
                query,
                policy_codes,
                prepared,
                started,
                cache_hit=False,
                model=result.model_used,
            )
            yield StreamEvent(event="done", data=result.model_dump(mode="json"))
        except AnswerGenerationError as exc:
            logger.error("rag_stream_failed", extra={"error": str(exc)})
            yield StreamEvent(event="error", data={"message": str(exc)})
        except Exception:
            logger.exception("rag_stream_failed")
            yield StreamEvent(event="error", data={"message": "internal error while answering"})

    # ------------------------------------------------------------------ pipeline

    async def _prepare(
        self,
        query: str,
        policy_codes: Sequence[str] | None,
        *,
        history: list[ChatTurn] | None,
        top_n: int,
        use_cache: bool,
        section_prefix: str | None = None,
    ) -> _Prepared:
        timings: dict[str, float] = {}
        warnings: list[str] = []

        start = time.perf_counter()
        standalone = await self.rewriter.rewrite(query, history)  # no history: no LLM call
        variants = [standalone]
        if self.settings.multi_query:
            variants += await self.rewriter.alternatives(standalone)
        timings["rewrite"] = _ms(start)

        start = time.perf_counter()
        embeddings = await asyncio.gather(*(self._embed(v, warnings) for v in variants))
        timings["embed"] = _ms(start)
        embedding = embeddings[0]

        prepared = _Prepared(
            chunks=[],
            standalone=standalone,
            rewritten_query=standalone if standalone != query else None,
            embedding=embedding,
            timings=timings,
            warnings=warnings,
        )

        if use_cache and self.cache is not None and embedding is not None:
            start = time.perf_counter()
            prepared.cached = await self._cache_lookup(policy_codes, embedding)
            timings["cache"] = _ms(start)
            if prepared.cached is not None:
                return prepared

        results = await asyncio.gather(
            *(
                self.retriever.hybrid(
                    v, emb, policy_codes=policy_codes, section_prefix=section_prefix
                )
                for v, emb in zip(variants, embeddings, strict=True)
            )
        )
        for key in ("vector", "keyword"):
            values = [r.timings_ms[key] for r in results if key in r.timings_ms]
            if values:
                timings[key] = max(values)
        timings["fuse"] = round(sum(r.timings_ms.get("fuse", 0.0) for r in results), 1)
        for result in results:
            warnings.extend(w for w in result.warnings if w not in warnings)
        fused = merge_ranked(
            [r.chunks for r in results], k=self.settings.rrf_k, top_k=self.settings.fused_top_k
        )

        start = time.perf_counter()
        prepared.chunks = await self._rerank(standalone, fused, top_n, warnings)
        timings["rerank"] = _ms(start)
        return prepared

    async def _embed(self, text: str, warnings: list[str]) -> list[float] | None:
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as exc:  # noqa: BLE001 - degrade to keyword-only instead of failing
            logger.warning("query_embedding_failed", extra={"error": type(exc).__name__})
            message = f"query embedding failed ({type(exc).__name__}); using keyword search only"
            if message not in warnings:
                warnings.append(message)
            return None

    async def _rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int, warnings: list[str]
    ) -> list[RetrievedChunk]:
        try:
            return await self.reranker.rerank(query, chunks, top_n)
        except Exception as exc:  # noqa: BLE001 - never fail the request because of the reranker
            logger.warning(
                "rerank_failed_using_fused_order",
                extra={"reranker": self.reranker.name, "error": type(exc).__name__},
            )
            warnings.append(f"reranker '{self.reranker.name}' failed; fused order used")
            return chunks[:top_n]

    # ------------------------------------------------------------------ cache (never fatal)

    async def _cache_lookup(
        self, policy_codes: Sequence[str] | None, embedding: list[float]
    ) -> PolicyAnswerResult | None:
        try:
            return await self.cache.lookup(policy_codes, embedding)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            logger.warning("cache_lookup_failed", exc_info=True)
            return None

    async def _cache_store(
        self,
        policy_codes: Sequence[str] | None,
        prepared: _Prepared,
        result: PolicyAnswerResult,
        use_cache: bool,
    ) -> None:
        if not use_cache or self.cache is None or prepared.embedding is None or result.warnings:
            return  # degraded answers (failed reranker / no vectors) are not worth caching
        try:
            await self.cache.store(policy_codes, prepared.embedding, result)
        except Exception:  # noqa: BLE001
            logger.warning("cache_store_failed", exc_info=True)

    # ------------------------------------------------------------------ results / logging

    def _result(self, prepared: _Prepared, answer, model_used: str | None) -> PolicyAnswerResult:  # type: ignore[no-untyped-def]
        return PolicyAnswerResult(
            answer=answer.answer,
            citations=answer.citations,
            insufficient_context=answer.insufficient_context,
            confidence=answer.confidence,
            chunks_used=prepared.chunks,
            model_used=model_used,
            cache_hit=False,
            rewritten_query=prepared.rewritten_query,
            warnings=prepared.warnings,
            timings_ms=prepared.timings,
        )

    def _from_cache(self, prepared: _Prepared) -> PolicyAnswerResult:
        assert prepared.cached is not None
        return prepared.cached.model_copy(
            update={
                "cache_hit": True,
                "rewritten_query": prepared.rewritten_query,
                "timings_ms": prepared.timings,
                "warnings": prepared.warnings,
            }
        )

    @staticmethod
    def _citation_metadata(prepared: _Prepared) -> list[dict]:
        source = prepared.cached.chunks_used if prepared.cached else prepared.chunks
        return [
            {
                "chunk_id": str(c.chunk_id),
                "policy_code": c.policy_code,
                "section_path": c.section_path,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "rerank_score": c.rerank_score,
                "fused_score": c.fused_score,
            }
            for c in source
        ]

    def _log(
        self,
        kind: str,
        query: str,
        policy_codes: Sequence[str] | None,
        prepared: _Prepared,
        started: float,
        *,
        cache_hit: bool,
        model: str | None,
    ) -> None:
        chunks = prepared.cached.chunks_used if prepared.cached else prepared.chunks
        logger.info(
            "rag_request",
            extra={
                "kind": kind,
                "query": query[:200],
                "policy_codes": list(policy_codes) if policy_codes else None,
                "chunk_ids": [str(c.chunk_id) for c in chunks],
                "cache_hit": cache_hit,
                "model": model,
                "reranker": self.reranker.name,
                "timings_ms": prepared.timings,
                "total_ms": _ms(started),
                "warnings": prepared.warnings,
            },
        )


def build_rag_service(
    session_factory: async_sessionmaker[AsyncSession],
    redis: Redis | None,
    settings: RagSettings | None = None,
) -> RetrievalService:
    """Production wiring: real embeddings, chat model (OpenAI -> Groq fallback), reranker, cache."""
    rag_settings = settings or get_rag_settings()
    return RetrievalService(
        session_factory=session_factory,
        embeddings=get_embeddings(),
        chat_model_factory=lambda: get_chat_model(temperature=0),
        reranker=build_reranker(rag_settings),
        cache=SemanticCache(redis, rag_settings)
        if redis is not None and rag_settings.cache_enabled
        else None,
        settings=rag_settings,
    )
