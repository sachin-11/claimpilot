"""mcp: notifications and tool idempotency

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-25 10:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('notifications',
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('channel', postgresql.ENUM('email', 'sms', name='notification_channel'), nullable=False),
    sa.Column('template', sa.String(length=50), nullable=False),
    sa.Column('rendered_message', sa.Text(), nullable=False),
    sa.Column('status', postgresql.ENUM('queued', 'sent_mock', name='notification_status'), server_default='queued', nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_notifications_claim_id_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_notifications')),
    sa.UniqueConstraint('idempotency_key', name=op.f('uq_notifications_idempotency_key'))
    )
    op.create_index('ix_notifications_claim_id_created_at', 'notifications', ['claim_id', 'created_at'], unique=False)
    op.create_table('tool_idempotency',
    sa.Column('tool', sa.String(length=100), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('claim_id', sa.UUID(), nullable=True),
    sa.Column('caller', sa.String(length=100), nullable=True),
    sa.Column('response', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_tool_idempotency_claim_id_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_tool_idempotency')),
    sa.UniqueConstraint('tool', 'idempotency_key', name=op.f('uq_tool_idempotency_tool_idempotency_key'))
    )


def downgrade() -> None:
    op.drop_table('tool_idempotency')
    op.drop_index('ix_notifications_claim_id_created_at', table_name='notifications')
    op.drop_table('notifications')
    op.execute("DROP TYPE notification_status")  # tables do not drop their enum types
    op.execute("DROP TYPE notification_channel")
