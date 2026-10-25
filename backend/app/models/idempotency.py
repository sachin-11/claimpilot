import uuid
from typing import Any

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base, CreatedAtMixin, UUIDPrimaryKeyMixin


class ToolIdempotency(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Result of a write tool call, keyed by (tool, idempotency_key). A repeat call with the same
    key returns `response` instead of writing again."""

    __tablename__ = "tool_idempotency"
    __table_args__ = (UniqueConstraint("tool", "idempotency_key"),)

    tool: Mapped[str] = mapped_column(String(100))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))  # same key, other arguments -> error
    claim_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    caller: Mapped[str | None] = mapped_column(String(100))
    response: Mapped[dict[str, Any]] = mapped_column(JSONB)
