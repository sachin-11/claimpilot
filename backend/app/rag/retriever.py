"""Hybrid retrieval: pgvector cosine search + Postgres full-text search, fused with RRF."""

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.policy import EMBEDDING_DIM
from app.observability.tracing import span
from app.rag.config import RagSettings
from app.rag.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_COLUMNS = """c.id, p.policy_code, c.section_path, c.page_start, c.page_end, c.content"""

# MATERIALIZED + outer ORDER BY: with iterative index scans pgvector may return rows slightly
# out of order, and the filters (policy / section) are applied after the index scan.
_VECTOR_SQL = f"""
WITH nearest AS MATERIALIZED (
    SELECT {_COLUMNS}, c.embedding <=> :qvec AS distance
    FROM policy_chunks c JOIN policies p ON p.id = c.policy_id
    WHERE p.is_active AND c.embedding IS NOT NULL {{filters}}
    ORDER BY c.embedding <=> :qvec
    LIMIT :k
)
SELECT * FROM nearest ORDER BY distance
"""

_KEYWORD_SQL = f"""
WITH q AS (SELECT websearch_to_tsquery('english', :query) AS tsq)
SELECT {_COLUMNS}, ts_rank_cd(c.search_tsv, q.tsq) AS score
FROM policy_chunks c JOIN policies p ON p.id = c.policy_id, q
WHERE p.is_active AND c.search_tsv @@ q.tsq {{filters}}
ORDER BY score DESC, p.policy_code, c.chunk_index
LIMIT :k
"""

# Words that carry no signal for a full-text search of policy wording.
STOPWORDS = frozenset(
    "a an and any are as at be by can could do does for from has have how i if in is it its "
    "me my of on or please shall should tell that the there this to under was what when "
    "where which who will with would you your".split()
)
_TERM_RE = re.compile(r"\w[\w\-/.%₹]*")
_PHRASE_RE = re.compile(r'"([^"]+)"')


def build_keyword_query(question: str, max_terms: int = 12) -> str:
    """Turn a natural-language question into a websearch_to_tsquery string of OR-ed terms.

    websearch_to_tsquery ANDs every word, so "What is the waiting period for cataract surgery?"
    would only match chunks containing all of them. OR-ing the content words and letting
    ts_rank_cd rank by how many (and how close) they match is far more forgiving. Quoted
    phrases are kept as phrases.
    """
    phrases = [f'"{p.strip()}"' for p in _PHRASE_RE.findall(question) if p.strip()]
    remainder = _PHRASE_RE.sub(" ", question)
    terms: list[str] = []
    for raw in _TERM_RE.findall(remainder):
        term = raw.strip(".-/").lower()
        if len(term) < 2 and not term.isdigit():
            continue
        if term in STOPWORDS or term in terms:
            continue
        terms.append(term)
    return " or ".join([*phrases, *terms[:max_terms]])


def rrf_fuse(
    vector_hits: Sequence[RetrievedChunk],
    keyword_hits: Sequence[RetrievedChunk],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    merged: dict[UUID, RetrievedChunk] = {}
    for hits, field_name in ((vector_hits, "vector_rank"), (keyword_hits, "keyword_rank")):
        for rank, hit in enumerate(hits, start=1):
            chunk = merged.setdefault(hit.chunk_id, hit.model_copy())
            setattr(chunk, field_name, rank)
            chunk.fused_score += 1.0 / (k + rank)
    ordered = sorted(
        merged.values(),
        key=lambda c: (
            -c.fused_score,
            min(r for r in (c.vector_rank, c.keyword_rank) if r is not None),
            str(c.chunk_id),
        ),
    )
    return ordered[:top_k]


def merge_ranked(
    lists: Sequence[Sequence[RetrievedChunk]], *, k: int = 60, top_k: int = 20
) -> list[RetrievedChunk]:
    """RRF over several already-ranked lists (multi-query retrieval)."""
    if len(lists) == 1:
        return list(lists[0])[:top_k]
    merged: dict[UUID, RetrievedChunk] = {}
    scores: dict[UUID, float] = {}
    for ranked in lists:
        for rank, hit in enumerate(ranked, start=1):
            merged.setdefault(hit.chunk_id, hit.model_copy())
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, chunk in merged.items():
        chunk.fused_score = scores[chunk_id]
    return sorted(merged.values(), key=lambda c: (-c.fused_score, str(c.chunk_id)))[:top_k]


@dataclass
class HybridResult:
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _row_to_chunk(row, **ranks: int) -> RetrievedChunk:  # type: ignore[no-untyped-def]
    return RetrievedChunk(
        chunk_id=row.id,
        policy_code=row.policy_code,
        section_path=row.section_path,
        page_start=row.page_start,
        page_end=row.page_end,
        content=row.content,
        **ranks,
    )


class Retriever:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], settings: RagSettings):
        self.session_factory = session_factory
        self.settings = settings

    @staticmethod
    def _filters(policy_codes: Sequence[str] | None, section_prefix: str | None):  # type: ignore[no-untyped-def]
        sql, params, binds = "", {}, []
        if policy_codes:
            sql += " AND p.policy_code = ANY(:codes)"
            params["codes"] = list(policy_codes)
            binds.append(bindparam("codes", type_=ARRAY(String)))
        if section_prefix:
            sql += " AND starts_with(c.section_path, :prefix)"
            params["prefix"] = section_prefix
        return sql, params, binds

    async def vector_search(
        self,
        embedding: Sequence[float],
        *,
        policy_codes: Sequence[str] | None = None,
        section_prefix: str | None = None,
    ) -> list[RetrievedChunk]:
        filters, params, binds = self._filters(policy_codes, section_prefix)
        statement = text(_VECTOR_SQL.format(filters=filters)).bindparams(
            bindparam("qvec", type_=Vector(EMBEDDING_DIM)), *binds
        )
        async with self.session_factory() as session:
            # SET LOCAL takes no bind parameters; the values are ints from settings.
            await session.execute(
                text(f"SET LOCAL hnsw.ef_search = {int(self.settings.hnsw_ef_search)}")
            )
            await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            rows = await session.execute(
                statement, {"qvec": list(embedding), "k": self.settings.vector_top_k, **params}
            )
            return [_row_to_chunk(r, vector_rank=i) for i, r in enumerate(rows, start=1)]

    async def keyword_search(
        self,
        query: str,
        *,
        policy_codes: Sequence[str] | None = None,
        section_prefix: str | None = None,
    ) -> list[RetrievedChunk]:
        tsquery_text = build_keyword_query(query, self.settings.max_keyword_terms)
        if not tsquery_text:
            return []
        filters, params, binds = self._filters(policy_codes, section_prefix)
        statement = text(_KEYWORD_SQL.format(filters=filters)).bindparams(*binds)
        async with self.session_factory() as session:
            rows = await session.execute(
                statement, {"query": tsquery_text, "k": self.settings.keyword_top_k, **params}
            )
            return [_row_to_chunk(r, keyword_rank=i) for i, r in enumerate(rows, start=1)]

    async def hybrid(
        self,
        query: str,
        embedding: Sequence[float] | None,
        *,
        policy_codes: Sequence[str] | None = None,
        section_prefix: str | None = None,
    ) -> HybridResult:
        """Vector and keyword search run concurrently (own sessions), then RRF."""
        timings: dict[str, float] = {}
        warnings: list[str] = []

        async def timed(name: str, coro):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            try:
                with span(f"rag.{name}", as_type="retriever", input={"query": query}) as s:
                    hits = await coro
                    s.update(output={"hits": len(hits)})
                    return hits
            finally:
                timings[name] = round((time.perf_counter() - start) * 1000, 1)

        tasks = {}
        if self.settings.keyword_search_enabled:
            tasks["keyword"] = timed(
                "keyword",
                self.keyword_search(
                    query, policy_codes=policy_codes, section_prefix=section_prefix
                ),
            )
        if self.settings.vector_search_enabled:
            if embedding is not None:
                tasks["vector"] = timed(
                    "vector",
                    self.vector_search(
                        embedding, policy_codes=policy_codes, section_prefix=section_prefix
                    ),
                )
            else:
                warnings.append("vector search skipped: no query embedding (keyword results only)")
        results = dict(zip(tasks, await asyncio.gather(*tasks.values()), strict=True))
        keyword_hits = results.get("keyword", [])
        vector_hits = results.get("vector", [])

        start = time.perf_counter()
        with span("rag.fuse"):
            fused = rrf_fuse(
                vector_hits, keyword_hits, k=self.settings.rrf_k, top_k=self.settings.fused_top_k
            )
        timings["fuse"] = round((time.perf_counter() - start) * 1000, 1)
        return HybridResult(chunks=fused, timings_ms=timings, warnings=warnings)
