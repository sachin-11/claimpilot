import uuid

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base, CreatedAtMixin, UUIDPrimaryKeyMixin
from app.models.enums import NotificationChannel, NotificationStatus
from app.models.types import pg_enum


class Notification(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A customer message written by the notify MCP server. Delivery is mocked: nothing is sent,
    the row (status `sent_mock`) is the record."""

    __tablename__ = "notifications"
    __table_args__ = (Index("ix_notifications_claim_id_created_at", "claim_id", "created_at"),)

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE")
    )
    channel: Mapped[NotificationChannel] = mapped_column(
        pg_enum(NotificationChannel, "notification_channel")
    )
    template: Mapped[str] = mapped_column(String(50))
    rendered_message: Mapped[str] = mapped_column(Text)
    status: Mapped[NotificationStatus] = mapped_column(
        pg_enum(NotificationStatus, "notification_status"),
        server_default=NotificationStatus.QUEUED.value,
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
