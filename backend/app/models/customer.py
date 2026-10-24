import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy import Date, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Customer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "customers"

    full_name: Mapped[str] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(32))
    email: Mapped[str | None] = mapped_column(String(255))
    dob: Mapped[date | None] = mapped_column(Date)
    address: Mapped[str | None] = mapped_column(Text)

    policies: Mapped[list["CustomerPolicy"]] = relationship(back_populates="customer")


class CustomerPolicy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "customer_policies"
    __table_args__ = (
        Index("ix_customer_policies_customer_id", "customer_id"),
        Index("ix_customer_policies_policy_id", "policy_id"),
    )

    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("customers.id", ondelete="CASCADE")
    )
    policy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("policies.id"))
    policy_number: Mapped[str] = mapped_column(String(100), unique=True)
    sum_insured: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    start_date: Mapped[date] = mapped_column(Date)
    # NULL = open-ended / renews yearly (the generated data has no end date).
    end_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), server_default="active")

    customer: Mapped[Customer] = relationship(back_populates="policies")
