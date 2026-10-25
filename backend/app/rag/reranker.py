"""Pluggable rerankers. Selected with RERANKER=none|local|cohere."""

import asyncio
import logging
from typing import Protocol

import httpx

from app.core.config import get_settings, has_real_key
from app.rag.config import RagSettings
from app.rag.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


def _document(chunk: RetrievedChunk, max_chars: int) -> str:
    """Section path + text: the path often contains the words the question is about."""
    return f"{chunk.section_path or ''}\n{chunk.content}"[:max_chars]


def _apply_scores(
    chunks: list[RetrievedChunk], scores: list[float], top_n: int
) -> list[RetrievedChunk]:
    ranked = []
    for chunk, score in zip(chunks, scores, strict=True):
        ranked.append(chunk.model_copy(update={"rerank_score": float(score)}))
    ranked.sort(key=lambda c: -(c.rerank_score or 0.0))
    return ranked[:top_n]


class NoopReranker:
    """Keeps the fused order (RRF)."""

    name = "none"

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_n]


class CohereReranker:
    name = "cohere"

    def __init__(
        self, api_key: str, model: str, max_chars: int, client: httpx.AsyncClient | None = None
    ):
        self.api_key = api_key
        self.model = model
        self.max_chars = max_chars
        self._client = client

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": [_document(c, self.max_chars) for c in chunks],
            "top_n": min(top_n, len(chunks)),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        client = self._client or httpx.AsyncClient(timeout=get_settings().reranker_timeout_seconds)
        try:
            response = await client.post(COHERE_RERANK_URL, json=payload, headers=headers)
            response.raise_for_status()
        finally:
            if self._client is None:
                await client.aclose()
        results = response.json()["results"]
        return [
            chunks[item["index"]].model_copy(
                update={"rerank_score": float(item["relevance_score"])}
            )
            for item in results
        ][:top_n]


class LocalReranker:
    """sentence-transformers CrossEncoder, loaded once, lazily (the model is ~1 GB)."""

    name = "local"

    def __init__(self, model_name: str, max_chars: int):
        self.model_name = model_name
        self.max_chars = max_chars
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        from sentence_transformers import CrossEncoder  # optional dependency

        logger.info("reranker_loading", extra={"model": self.model_name})
        return CrossEncoder(self.model_name)

    async def _get_model(self):  # type: ignore[no-untyped-def]
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
            return self._model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = await self._get_model()
        pairs = [(query, _document(c, self.max_chars)) for c in chunks]
        scores = await asyncio.to_thread(lambda: model.predict(pairs).tolist())
        return _apply_scores(chunks, scores, top_n)


def build_reranker(settings: RagSettings) -> Reranker:
    """The configured reranker. A misconfiguration degrades to 'none' instead of failing."""
    if settings.reranker == "cohere":
        if not has_real_key(settings.cohere_api_key):
            logger.warning("reranker_fallback", extra={"reason": "COHERE_API_KEY not set"})
            return NoopReranker()
        return CohereReranker(
            settings.cohere_api_key.get_secret_value(),  # type: ignore[union-attr]
            settings.cohere_rerank_model,
            settings.rerank_max_chars,
        )
    if settings.reranker == "local":
        return LocalReranker(settings.local_rerank_model, settings.rerank_max_chars)
    return NoopReranker()
