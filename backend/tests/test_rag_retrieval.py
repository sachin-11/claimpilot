import pytest
from rag_db import (  # noqa: F401  (session_factory is a fixture)
    A_CATARACT,
    A_LAW,
    A_ROOM,
    B_CATARACT,
    QUERY_VEC,
    X_CATARACT,
    session_factory,
)
from rag_fakes import make_chunk, rag_settings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.rag.retriever import Retriever, build_keyword_query, merge_ranked, rrf_fuse


def chunk_ids(chunks):
    return [c.chunk_id.int for c in chunks]


# ---------------------------------------------------------------- pure logic


def test_rrf_fusion_orders_by_combined_rank() -> None:
    a, b, c, d = (make_chunk(n) for n in (1, 2, 3, 4))
    vector = [a, b, c]  # A is best by vector
    keyword = [c, a, d]  # C is best by keyword

    fused = rrf_fuse(vector, keyword, k=60, top_k=10)

    # A: 1/61 + 1/62, C: 1/63 + 1/61, B: 1/62 only, D: 1/63 only
    assert chunk_ids(fused) == [1, 3, 2, 4]
    assert fused[0].fused_score == pytest.approx(1 / 61 + 1 / 62)
    assert (fused[0].vector_rank, fused[0].keyword_rank) == (1, 2)
    assert (fused[2].vector_rank, fused[2].keyword_rank) == (2, None)
    assert (fused[3].vector_rank, fused[3].keyword_rank) == (None, 3)
    assert chunk_ids(rrf_fuse(vector, keyword, top_k=2)) == [1, 3]  # top_k respected


def test_rrf_does_not_mutate_its_inputs() -> None:
    a = make_chunk(1)
    rrf_fuse([a], [a])
    assert a.fused_score == 0.0 and a.vector_rank is None


def test_merge_ranked_for_multi_query() -> None:
    first = rrf_fuse([make_chunk(1), make_chunk(2)], [])
    second = rrf_fuse([make_chunk(2), make_chunk(3)], [])
    merged = merge_ranked([first, second], k=60, top_k=10)
    assert chunk_ids(merged) == [2, 1, 3]  # chunk 2 appears in both lists
    assert chunk_ids(merge_ranked([first], top_k=1)) == [1]


def test_keyword_query_keeps_content_words_and_numbers() -> None:
    query = build_keyword_query("What is the waiting period for cataract surgery?")
    assert query == "waiting or period or cataract or surgery"
    assert build_keyword_query("Is the 24 months waiting period for hernia?") == (
        "24 or months or waiting or period or hernia"
    )
    assert build_keyword_query('what does "room rent" cover?') == '"room rent" or cover'
    assert build_keyword_query("is it the of a") == ""  # only stopwords
    assert build_keyword_query("co-payment 5%") == "co-payment or 5%"
    assert (
        build_keyword_query(" ".join(f"term{i}" for i in range(30)), max_terms=3).count(" or ") == 2
    )


# ---------------------------------------------------------------- database


def contents(chunks) -> list[str]:
    return [c.content for c in chunks]


async def test_keyword_search_finds_exact_term_that_vector_search_ranks_low(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    retriever = Retriever(session_factory, rag_settings())
    query = "cataract 24 months"

    result = await retriever.hybrid(query, QUERY_VEC)

    by_content = {c.content: c for c in result.chunks}
    target = by_content[A_CATARACT]
    assert target.keyword_rank == 1  # exact terms: cataract, 24, months
    assert target.vector_rank == 5  # orthogonal to the query: the worst of the five active chunks
    assert by_content[A_LAW].vector_rank == 1 and by_content[A_LAW].keyword_rank is None
    assert result.chunks[0].content == A_CATARACT  # fusion lifts it to the top
    assert X_CATARACT not in contents(result.chunks)  # inactive policy never returned
    assert set(result.timings_ms) == {"keyword", "vector", "fuse"}


async def test_policy_code_filter_excludes_other_policies(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    retriever = Retriever(session_factory, rag_settings())

    only_a = await retriever.hybrid("cataract waiting months", QUERY_VEC, policy_codes=["POL-A"])
    only_b = await retriever.hybrid("cataract waiting months", QUERY_VEC, policy_codes=["POL-B"])
    both = await retriever.hybrid(
        "cataract waiting months", QUERY_VEC, policy_codes=["POL-A", "POL-B"]
    )
    everything = await retriever.hybrid("cataract waiting months", QUERY_VEC)

    assert {c.policy_code for c in only_a.chunks} == {"POL-A"}
    assert {c.policy_code for c in only_b.chunks} == {"POL-B"}
    assert {c.policy_code for c in both.chunks} == {"POL-A", "POL-B"}
    assert {c.policy_code for c in everything.chunks} == {"POL-A", "POL-B"}  # POL-X is inactive
    assert len(only_a.chunks) == 3 and len(only_b.chunks) == 2


async def test_section_prefix_filter(session_factory: async_sessionmaker[AsyncSession]) -> None:
    retriever = Retriever(session_factory, rag_settings())

    result = await retriever.hybrid(
        "cataract", QUERY_VEC, policy_codes=["POL-A"], section_prefix="6 Waiting Period"
    )

    assert contents(result.chunks) == [A_CATARACT]


async def test_without_an_embedding_only_keyword_results_are_returned(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    retriever = Retriever(session_factory, rag_settings())

    result = await retriever.hybrid("cataract", None)

    assert {c.content for c in result.chunks} == {A_CATARACT, B_CATARACT}
    assert all(c.vector_rank is None for c in result.chunks)
    assert "vector" not in result.timings_ms
    assert any("keyword results only" in w for w in result.warnings)


async def test_query_with_only_stopwords_still_returns_vector_hits(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    retriever = Retriever(session_factory, rag_settings(vector_top_k=2))

    result = await retriever.hybrid("is it the of", QUERY_VEC)

    assert all(c.keyword_rank is None for c in result.chunks)
    assert contents(result.chunks) == [A_LAW, A_ROOM]  # nearest two by cosine distance


async def test_search_switches_give_vector_only_and_keyword_only_baselines(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    both = await Retriever(session_factory, rag_settings()).hybrid("cataract 24 months", QUERY_VEC)
    vector_only = await Retriever(
        session_factory, rag_settings(keyword_search_enabled=False)
    ).hybrid("cataract 24 months", QUERY_VEC)
    keyword_only = await Retriever(
        session_factory, rag_settings(vector_search_enabled=False)
    ).hybrid("cataract 24 months", QUERY_VEC)

    assert all(c.keyword_rank is None for c in vector_only.chunks)
    assert set(vector_only.timings_ms) == {"vector", "fuse"}
    assert (
        contents(vector_only.chunks)[0] == A_LAW
    )  # pure cosine order: the cataract clause is last
    assert contents(vector_only.chunks)[-1] == A_CATARACT

    assert all(c.vector_rank is None for c in keyword_only.chunks)
    assert set(keyword_only.timings_ms) == {"keyword", "fuse"}
    assert contents(keyword_only.chunks)[0] == A_CATARACT
    assert keyword_only.warnings == []  # switched off on purpose: not a degraded search

    assert {c.chunk_id for c in both.chunks} >= {c.chunk_id for c in keyword_only.chunks}


def test_at_least_one_search_mode_must_stay_enabled() -> None:
    with pytest.raises(ValueError, match="enable vector and/or keyword"):
        rag_settings(vector_search_enabled=False, keyword_search_enabled=False)
