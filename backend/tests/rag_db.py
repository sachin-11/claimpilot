"""Committed fixture rows for the RAG tests (the retriever opens its own sessions)."""

import hashlib

import pytest
from rag_fakes import vec
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.models import Policy, PolicyChunk

A_ROOM = (
    "4.1 Hospitalization: Room rent is covered up to 2% of the sum insured, "
    "maximum Rs.5000 per day."
)
A_CATARACT = (
    "6.3 Specific waiting period: Cataract is covered only after 24 months of continuous coverage."
)
A_LAW = "General conditions: this policy is governed by the laws of India and the courts of India."
B_CATARACT = "Cataract treatment is covered after 12 months of continuous coverage."
B_AMBULANCE = "Road ambulance charges are payable up to Rs.2000 per hospitalisation."
X_CATARACT = "Cataract after 99 months waiting period (inactive policy)."

# query vector = axis c. Cosine distance to it: A_LAW 0, A_ROOM 0.2, B_AMBULANCE 0.3,
# B_CATARACT 0.4, A_CATARACT 1.0 (orthogonal): the cataract clause is the WORST vector match.
QUERY_VEC = vec(c=1)
SEED = [
    ("POL-A", True, "4. COVERAGE", 1, A_ROOM, vec(c=0.8, a=0.6)),
    ("POL-A", True, "6 Waiting Period > 6.3 Specific", 2, A_CATARACT, vec(b=1)),
    ("POL-A", True, "10 GENERAL TERMS", 3, A_LAW, vec(c=1)),
    ("POL-B", True, "5 Cataract", 1, B_CATARACT, vec(c=0.6, d=0.8)),
    ("POL-B", True, "7 Extras", 2, B_AMBULANCE, vec(c=0.7, e=0.714)),
    ("POL-X", False, "1 Old", 1, X_CATARACT, vec(c=1)),
]


@pytest.fixture
async def session_factory(test_database_url: str):
    """Committed fixture rows (the retriever opens its own sessions), removed afterwards."""
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        policies: dict[str, Policy] = {}
        for code, active, _, _, _, _ in SEED:
            if code not in policies:
                policies[code] = Policy(policy_code=code, name=code, is_active=active, rules={})
                session.add(policies[code])
        await session.flush()
        for index, (code, _, section, page, content, embedding) in enumerate(SEED):
            session.add(
                PolicyChunk(
                    policy_id=policies[code].id,
                    chunk_index=index,
                    content=content,
                    section_path=section,
                    page_start=page,
                    page_end=page,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    embedding=embedding,
                )
            )
        await session.commit()
    try:
        yield factory
    finally:
        async with factory() as session:
            await session.execute(text("TRUNCATE policies CASCADE"))
            await session.commit()
        await engine.dispose()
