import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Policy, PolicyChunk
from app.models.policy import EMBEDDING_DIM
from app.repositories import PolicyChunkRepository, PolicyRepository
from app.schemas.policy import PolicyChunkCreate


def make_chunk(index: int, content: str, **kwargs) -> PolicyChunkCreate:
    return PolicyChunkCreate(
        chunk_index=index,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        **kwargs,
    )


@pytest.fixture
async def policy(session: AsyncSession) -> Policy:
    created, _ = await PolicyRepository(session).upsert_from_rules(
        {"policy_id": "POL-T", "name": "Test Policy"}
    )
    return created


def unit_vector(position: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[position] = 1.0
    return vec


async def test_tsvector_is_generated_and_searchable(session: AsyncSession, policy: Policy) -> None:
    repo = PolicyChunkRepository(session)
    await repo.bulk_upsert(
        policy.id,
        [
            make_chunk(0, "Hospitalization expenses are covered up to the sum insured."),
            make_chunk(1, "Cosmetic surgery is permanently excluded."),
        ],
    )
    rows = (
        (
            await session.execute(
                select(PolicyChunk)
                .where(PolicyChunk.search_tsv.op("@@")(func_plainto("hospitalization expense")))
                .order_by(PolicyChunk.chunk_index)
            )
        )
        .scalars()
        .all()
    )
    # english config stems "hospitalization expenses" -> matches; the other chunk does not
    assert [c.chunk_index for c in rows] == [0]
    assert rows[0].search_tsv  # populated by the database, never set from Python


def func_plainto(query: str):
    from sqlalchemy import func

    return func.plainto_tsquery("english", query)


async def test_vector_cosine_distance_query(session: AsyncSession, policy: Policy) -> None:
    repo = PolicyChunkRepository(session)
    await repo.bulk_upsert(
        policy.id,
        [
            make_chunk(0, "alpha", embedding=unit_vector(0)),
            make_chunk(1, "beta", embedding=unit_vector(1)),
        ],
    )
    distance = PolicyChunk.embedding.cosine_distance(unit_vector(1))
    rows = (
        await session.execute(select(PolicyChunk.content, distance).order_by(distance).limit(2))
    ).all()
    assert rows[0][0] == "beta"
    assert rows[0][1] == pytest.approx(0.0, abs=1e-6)
    assert rows[1][1] == pytest.approx(1.0, abs=1e-6)  # orthogonal vectors


async def test_bulk_upsert_skips_unchanged_chunks(session: AsyncSession, policy: Policy) -> None:
    repo = PolicyChunkRepository(session)
    batch = [make_chunk(0, "first clause", page_start=1), make_chunk(1, "second clause")]

    first = await repo.bulk_upsert(policy.id, batch)
    assert (first.inserted, first.updated, first.unchanged) == (2, 0, 0)

    # Embed one chunk (as Module 3 will), then re-ingest identical content.
    chunk = (
        await session.execute(select(PolicyChunk).where(PolicyChunk.chunk_index == 0))
    ).scalar_one()
    chunk.embedding = unit_vector(3)
    await session.flush()

    again = await repo.bulk_upsert(policy.id, batch)
    assert (again.inserted, again.updated, again.unchanged) == (0, 0, 2)
    await session.refresh(chunk)
    assert chunk.embedding is not None  # untouched by the re-ingest

    # Same content moved to another page/index: updated in place, not duplicated.
    moved = await repo.bulk_upsert(policy.id, [make_chunk(5, "first clause", page_start=2)])
    assert (moved.inserted, moved.updated, moved.unchanged) == (0, 1, 0)
    assert await repo.count_by_policy(policy.id) == 2


async def test_delete_stale_removes_only_unlisted_hashes(
    session: AsyncSession, policy: Policy
) -> None:
    repo = PolicyChunkRepository(session)
    keep = make_chunk(0, "keep me")
    await repo.bulk_upsert(policy.id, [keep, make_chunk(1, "drop me")])

    deleted = await repo.delete_stale(policy.id, {keep.content_hash})

    assert deleted == 1
    assert await repo.count_by_policy(policy.id) == 1


async def test_policy_upsert_bumps_version_only_on_change(session: AsyncSession) -> None:
    repo = PolicyRepository(session)
    rules = {"policy_id": "POL-V", "name": "V", "co_pay_percent": 5}

    policy, changed = await repo.upsert_from_rules(rules)
    assert changed and policy.version == 1
    _, changed = await repo.upsert_from_rules(dict(rules))
    assert not changed and policy.version == 1
    policy, changed = await repo.upsert_from_rules({**rules, "co_pay_percent": 10})
    assert changed and policy.version == 2
