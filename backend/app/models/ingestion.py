import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import IngestionStatus
from app.models.types import pg_enum


class IngestionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One attempt to ingest one policy PDF. Also the source of truth for change detection:
    a policy is skipped when its file hash equals the one of the last successful run."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_policy_id_started_at", "policy_id", "started_at"),)

    policy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("policies.id", ondelete="CASCADE")
    )
    # NULL when the PDF could not be read at all.
    file_sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[IngestionStatus] = mapped_column(
        pg_enum(IngestionStatus, "ingestion_status"),
        server_default=IngestionStatus.RUNNING.value,
    )
    chunks_total: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    chunks_new: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    chunks_deleted: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    tokens_used: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
