import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, TimestampMixin, UUIDPrimaryKeyMixin

EMBEDDING_DIM = 1536  # text-embedding-3-small


class Policy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "policies"

    policy_code: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    insurer: Mapped[str | None] = mapped_column(String(255))
    source_pdf: Mapped[str | None] = mapped_column(String(255))
    rules: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    version: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))

    chunks: Mapped[list["PolicyChunk"]] = relationship(
        back_populates="policy", cascade="all, delete-orphan", passive_deletes=True
    )


class PolicyChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "policy_chunks"
    __table_args__ = (
        UniqueConstraint("policy_id", "content_hash"),
        Index("ix_policy_chunks_policy_id", "policy_id"),
        Index(
            "ix_policy_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_policy_chunks_search_tsv", "search_tsv", postgresql_using="gin"),
        Index("ix_policy_chunks_metadata", "metadata", postgresql_using="gin"),
    )

    policy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("policies.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    # "metadata" is reserved on declarative classes, so the attribute is called meta.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=text("'{}'::jsonb")
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    search_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    policy: Mapped[Policy] = relationship(back_populates="chunks")
