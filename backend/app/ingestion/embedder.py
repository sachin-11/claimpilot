"""Batch embedding with retries. Uses get_embeddings() only (no cross-provider fallback)."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import openai
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ingestion.chunker import TokenCounter, count_tokens
from app.ingestion.config import IngestionSettings
from app.models.policy import EMBEDDING_DIM

logger = logging.getLogger(__name__)

RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    TimeoutError,
)


class AsyncEmbedder(Protocol):
    """The part of LangChain's Embeddings interface that ingestion needs."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class EmbedResult:
    vectors: list[list[float]] = field(default_factory=list)
    tokens_used: int = 0
    batches: int = 0
    estimated_cost_usd: float = 0.0


def make_batches(
    texts: list[str], token_counts: list[int], max_items: int, max_tokens: int
) -> list[list[int]]:
    """Indices grouped into batches of <= max_items texts and <= max_tokens tokens."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index, tokens in enumerate(token_counts):
        if current and (len(current) >= max_items or current_tokens + tokens > max_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(index)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


async def embed_texts(
    texts: list[str],
    embedder: AsyncEmbedder,
    settings: IngestionSettings,
    *,
    tokens: TokenCounter = count_tokens,
    sleep: Callable | None = None,
) -> EmbedResult:
    """Embed texts in batches; retry rate limits / timeouts with exponential backoff.

    Returns one vector per input text, in the same order.
    """
    result = EmbedResult()
    if not texts:
        return result

    counts = [tokens(t) for t in texts]
    vectors: list[list[float] | None] = [None] * len(texts)

    for batch in make_batches(
        texts, counts, settings.embed_batch_size, settings.embed_max_batch_tokens
    ):
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE),
            stop=stop_after_attempt(settings.embed_max_retries),
            wait=wait_exponential(
                multiplier=settings.embed_retry_min_seconds,
                max=settings.embed_retry_max_seconds,
            ),
            before_sleep=lambda state: logger.warning(
                "embed_retry",
                extra={"attempt": state.attempt_number, "error": repr(state.outcome.exception())},
            ),
            reraise=True,
            **({"sleep": sleep} if sleep else {}),
        )
        batch_vectors = await retrying(embedder.aembed_documents, [texts[i] for i in batch])
        if len(batch_vectors) != len(batch):
            raise ValueError(f"expected {len(batch)} vectors, got {len(batch_vectors)}")
        for index, vector in zip(batch, batch_vectors, strict=True):
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"embedding has {len(vector)} dimensions, the column holds {EMBEDDING_DIM}; "
                    "check EMBEDDING_MODEL"
                )
            vectors[index] = vector
        result.batches += 1
        result.tokens_used += sum(counts[i] for i in batch)

    result.vectors = [v for v in vectors if v is not None]
    result.estimated_cost_usd = (
        result.tokens_used / 1_000_000 * settings.embed_cost_per_million_tokens
    )
    logger.info(
        "embedded",
        extra={
            "texts": len(texts),
            "batches": result.batches,
            "tokens": result.tokens_used,
            "cost_usd": round(result.estimated_cost_usd, 6),
        },
    )
    return result
