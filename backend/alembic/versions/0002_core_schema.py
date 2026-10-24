"""core schema

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-24 04:59:09.529308
"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('audit_logs',
    sa.Column('actor_type', sa.Enum('user', 'agent', 'system', name='actor_type'), nullable=False),
    sa.Column('actor_id', sa.String(length=100), nullable=True),
    sa.Column('action', sa.String(length=100), nullable=False),
    sa.Column('entity_type', sa.String(length=50), nullable=False),
    sa.Column('entity_id', sa.String(length=64), nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('request_id', sa.String(length=64), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_logs'))
    )
    op.create_index('ix_audit_logs_entity', 'audit_logs', ['entity_type', 'entity_id'], unique=False)
    # Append-only: reject UPDATE and DELETE (TRUNCATE is still possible for admins/tests).
    op.execute("""
        CREATE FUNCTION audit_logs_block_mutation() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_logs is append-only';
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE TRIGGER trg_audit_logs_append_only
        BEFORE UPDATE OR DELETE ON audit_logs
        FOR EACH ROW EXECUTE FUNCTION audit_logs_block_mutation()
    """)
    op.create_table('customers',
    sa.Column('full_name', sa.String(length=255), nullable=False),
    sa.Column('phone', sa.String(length=32), nullable=True),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.Column('dob', sa.Date(), nullable=True),
    sa.Column('address', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_customers'))
    )
    op.create_table('policies',
    sa.Column('policy_code', sa.String(length=50), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('insurer', sa.String(length=255), nullable=True),
    sa.Column('source_pdf', sa.String(length=255), nullable=True),
    sa.Column('rules', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('version', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_policies')),
    sa.UniqueConstraint('policy_code', name=op.f('uq_policies_policy_code'))
    )
    op.create_table('users',
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=255), nullable=False),
    sa.Column('role', sa.Enum('officer', 'admin', name='user_role'), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )
    op.create_table('customer_policies',
    sa.Column('customer_id', sa.UUID(), nullable=False),
    sa.Column('policy_id', sa.UUID(), nullable=False),
    sa.Column('policy_number', sa.String(length=100), nullable=False),
    sa.Column('sum_insured', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=True),
    sa.Column('status', sa.String(length=20), server_default='active', nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], name=op.f('fk_customer_policies_customer_id_customers'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['policy_id'], ['policies.id'], name=op.f('fk_customer_policies_policy_id_policies')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_customer_policies')),
    sa.UniqueConstraint('policy_number', name=op.f('uq_customer_policies_policy_number'))
    )
    op.create_index('ix_customer_policies_customer_id', 'customer_policies', ['customer_id'], unique=False)
    op.create_index('ix_customer_policies_policy_id', 'customer_policies', ['policy_id'], unique=False)
    op.create_table('policy_chunks',
    sa.Column('policy_id', sa.UUID(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('section_path', sa.Text(), nullable=True),
    sa.Column('page_start', sa.Integer(), nullable=True),
    sa.Column('page_end', sa.Integer(), nullable=True),
    sa.Column('token_count', sa.Integer(), nullable=True),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
    sa.Column('search_tsv', postgresql.TSVECTOR(), sa.Computed("to_tsvector('english', content)", persisted=True), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['policy_id'], ['policies.id'], name=op.f('fk_policy_chunks_policy_id_policies'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_policy_chunks')),
    sa.UniqueConstraint('policy_id', 'content_hash', name=op.f('uq_policy_chunks_policy_id_content_hash'))
    )
    op.create_index('ix_policy_chunks_embedding_hnsw', 'policy_chunks', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('ix_policy_chunks_metadata', 'policy_chunks', ['metadata'], unique=False, postgresql_using='gin')
    op.create_index('ix_policy_chunks_policy_id', 'policy_chunks', ['policy_id'], unique=False)
    op.create_index('ix_policy_chunks_search_tsv', 'policy_chunks', ['search_tsv'], unique=False, postgresql_using='gin')
    op.create_table('claims',
    sa.Column('claim_number', sa.String(length=50), nullable=False),
    sa.Column('customer_policy_id', sa.UUID(), nullable=True),
    sa.Column('hospital_name', sa.String(length=255), nullable=True),
    sa.Column('diagnosis_text', sa.Text(), nullable=True),
    sa.Column('icd10_code', sa.String(length=16), nullable=True),
    sa.Column('admission_date', sa.Date(), nullable=True),
    sa.Column('discharge_date', sa.Date(), nullable=True),
    sa.Column('claimed_amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('approved_amount', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('status', sa.Enum('submitted', 'processing', 'awaiting_approval', 'approved', 'partially_approved', 'rejected', 'documents_requested', 'failed', name='claim_status'), server_default='submitted', nullable=False),
    sa.Column('source', sa.Enum('seed', 'upload', name='claim_source'), server_default='upload', nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['customer_policy_id'], ['customer_policies.id'], name=op.f('fk_claims_customer_policy_id_customer_policies')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claims')),
    sa.UniqueConstraint('claim_number', name=op.f('uq_claims_claim_number'))
    )
    op.create_index('ix_claims_customer_policy_id', 'claims', ['customer_policy_id'], unique=False)
    op.create_index('ix_claims_status', 'claims', ['status'], unique=False)
    op.create_table('claim_decisions',
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('decision', sa.Enum('approve', 'partial_approve', 'reject', 'flag_for_review', 'request_documents', name='decision_type'), nullable=False),
    sa.Column('payable_amount', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('deductions', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('reasons', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('cited_clauses', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('risk_flags', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('agent_run_id', sa.String(length=100), nullable=True),
    sa.Column('model_used', sa.String(length=100), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_claim_decisions_claim_id_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_decisions'))
    )
    op.create_index('ix_claim_decisions_claim_id', 'claim_decisions', ['claim_id'], unique=False)
    op.create_table('claim_documents',
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('doc_type', sa.Enum('hospital_bill', 'discharge_summary', 'claim_form', 'other', name='doc_type'), nullable=False),
    sa.Column('file_path', sa.Text(), nullable=False),
    sa.Column('mime_type', sa.String(length=100), nullable=False),
    sa.Column('is_scanned', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('page_count', sa.Integer(), nullable=True),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_claim_documents_claim_id_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_documents')),
    sa.UniqueConstraint('claim_id', 'file_path', name=op.f('uq_claim_documents_claim_id_file_path'))
    )
    op.create_index('ix_claim_documents_claim_id', 'claim_documents', ['claim_id'], unique=False)
    op.create_table('claim_extractions',
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('extracted_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('model_used', sa.String(length=100), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('validation_errors', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('version', sa.Integer(), server_default=sa.text('1'), nullable=False),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_claim_extractions_claim_id_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_extractions')),
    sa.UniqueConstraint('claim_id', 'version', name=op.f('uq_claim_extractions_claim_id_version'))
    )
    op.create_index('ix_claim_extractions_claim_id', 'claim_extractions', ['claim_id'], unique=False)
    op.create_index('ix_claim_extractions_extracted_data', 'claim_extractions', ['extracted_data'], unique=False, postgresql_using='gin')
    op.create_table('claim_approvals',
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('decision_id', sa.UUID(), nullable=False),
    sa.Column('reviewer_id', sa.UUID(), nullable=False),
    sa.Column('action', sa.Enum('approved', 'rejected', 'edited', name='approval_action'), nullable=False),
    sa.Column('final_amount', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], name=op.f('fk_claim_approvals_claim_id_claims'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['decision_id'], ['claim_decisions.id'], name=op.f('fk_claim_approvals_decision_id_claim_decisions'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['reviewer_id'], ['users.id'], name=op.f('fk_claim_approvals_reviewer_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_approvals'))
    )
    op.create_index('ix_claim_approvals_claim_id', 'claim_approvals', ['claim_id'], unique=False)
    op.create_index('ix_claim_approvals_decision_id', 'claim_approvals', ['decision_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_claim_approvals_decision_id', table_name='claim_approvals')
    op.drop_index('ix_claim_approvals_claim_id', table_name='claim_approvals')
    op.drop_table('claim_approvals')
    op.drop_index('ix_claim_extractions_extracted_data', table_name='claim_extractions', postgresql_using='gin')
    op.drop_index('ix_claim_extractions_claim_id', table_name='claim_extractions')
    op.drop_table('claim_extractions')
    op.drop_index('ix_claim_documents_claim_id', table_name='claim_documents')
    op.drop_table('claim_documents')
    op.drop_index('ix_claim_decisions_claim_id', table_name='claim_decisions')
    op.drop_table('claim_decisions')
    op.drop_index('ix_claims_status', table_name='claims')
    op.drop_index('ix_claims_customer_policy_id', table_name='claims')
    op.drop_table('claims')
    op.drop_index('ix_policy_chunks_search_tsv', table_name='policy_chunks', postgresql_using='gin')
    op.drop_index('ix_policy_chunks_policy_id', table_name='policy_chunks')
    op.drop_index('ix_policy_chunks_metadata', table_name='policy_chunks', postgresql_using='gin')
    op.drop_index('ix_policy_chunks_embedding_hnsw', table_name='policy_chunks', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_table('policy_chunks')
    op.drop_index('ix_customer_policies_policy_id', table_name='customer_policies')
    op.drop_index('ix_customer_policies_customer_id', table_name='customer_policies')
    op.drop_table('customer_policies')
    op.drop_table('users')
    op.drop_table('policies')
    op.drop_table('customers')
    op.execute("DROP TRIGGER trg_audit_logs_append_only ON audit_logs")
    op.execute("DROP FUNCTION audit_logs_block_mutation()")
    op.drop_index('ix_audit_logs_entity', table_name='audit_logs')
    op.drop_table('audit_logs')
    # Tables do not drop their native enum types.
    for enum_name in (
        'approval_action', 'decision_type', 'doc_type', 'claim_source', 'claim_status',
        'user_role', 'actor_type',
    ):
        op.execute(f"DROP TYPE {enum_name}")
