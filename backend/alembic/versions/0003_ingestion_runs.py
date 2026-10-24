"""ingestion runs

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-24 05:15:18.284861
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('ingestion_runs',
    sa.Column('policy_id', sa.UUID(), nullable=False),
    sa.Column('file_sha256', sa.String(length=64), nullable=True),
    sa.Column('status', sa.Enum('running', 'success', 'failed', name='ingestion_status'), server_default='running', nullable=False),
    sa.Column('chunks_total', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('chunks_new', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('chunks_deleted', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('tokens_used', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['policy_id'], ['policies.id'], name=op.f('fk_ingestion_runs_policy_id_policies'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ingestion_runs'))
    )
    op.create_index('ix_ingestion_runs_policy_id_started_at', 'ingestion_runs', ['policy_id', 'started_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_ingestion_runs_policy_id_started_at', table_name='ingestion_runs')
    op.drop_table('ingestion_runs')
    op.execute("DROP TYPE ingestion_status")  # tables do not drop their enum types
