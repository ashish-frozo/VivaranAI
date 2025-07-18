"""Initial database schema

Revision ID: a592d8a8dc41
Revises: 
Create Date: 2025-07-07 08:49:32.169593

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a592d8a8dc41'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('system_metrics',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('metric_name', sa.String(length=255), nullable=False),
    sa.Column('metric_value', sa.Numeric(precision=15, scale=6), nullable=False),
    sa.Column('metric_type', sa.String(length=50), nullable=False),
    sa.Column('labels', sa.JSON(), nullable=True),
    sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metrics_name', 'system_metrics', ['metric_name'], unique=False)
    op.create_index('idx_metrics_timestamp', 'system_metrics', ['timestamp'], unique=False)
    op.create_index('idx_metrics_type', 'system_metrics', ['metric_type'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('username', sa.String(length=100), nullable=False),
    sa.Column('full_name', sa.String(length=255), nullable=False),
    sa.Column('role', sa.String(length=50), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('is_verified', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_created_at', 'users', ['created_at'], unique=False)
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_role', 'users', ['role'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_table('api_keys',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('key_hash', sa.String(length=255), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('role', sa.String(length=50), nullable=False),
    sa.Column('permissions', sa.JSON(), nullable=False),
    sa.Column('rate_limit_per_minute', sa.Integer(), nullable=True),
    sa.Column('rate_limit_per_hour', sa.Integer(), nullable=True),
    sa.Column('rate_limit_per_day', sa.Integer(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
    sa.Column('usage_count', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_api_keys_active', 'api_keys', ['is_active'], unique=False)
    op.create_index('idx_api_keys_hash', 'api_keys', ['key_hash'], unique=False)
    op.create_index('idx_api_keys_user_id', 'api_keys', ['user_id'], unique=False)
    op.create_index(op.f('ix_api_keys_key_hash'), 'api_keys', ['key_hash'], unique=True)
    op.create_table('audit_logs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('action', sa.String(length=50), nullable=False),
    sa.Column('resource_type', sa.String(length=100), nullable=False),
    sa.Column('resource_id', sa.String(length=255), nullable=True),
    sa.Column('endpoint', sa.String(length=255), nullable=True),
    sa.Column('method', sa.String(length=10), nullable=True),
    sa.Column('user_agent', sa.String(length=500), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.Column('status_code', sa.Integer(), nullable=True),
    sa.Column('response_time', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_logs_action', 'audit_logs', ['action'], unique=False)
    op.create_index('idx_audit_logs_created_at', 'audit_logs', ['created_at'], unique=False)
    op.create_index('idx_audit_logs_ip', 'audit_logs', ['ip_address'], unique=False)
    op.create_index('idx_audit_logs_resource', 'audit_logs', ['resource_type'], unique=False)
    op.create_index('idx_audit_logs_user_id', 'audit_logs', ['user_id'], unique=False)
    op.create_table('bill_analyses',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=False),
    sa.Column('file_hash', sa.String(length=64), nullable=False),
    sa.Column('file_size', sa.BigInteger(), nullable=False),
    sa.Column('content_type', sa.String(length=100), nullable=False),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('analysis_type', sa.String(length=100), nullable=False),
    sa.Column('total_amount', sa.Numeric(precision=10, scale=2), nullable=True),
    sa.Column('suspected_overcharges', sa.Numeric(precision=10, scale=2), nullable=True),
    sa.Column('accuracy_score', sa.Numeric(precision=5, scale=4), nullable=True),
    sa.Column('confidence_level', sa.Numeric(precision=5, scale=4), nullable=True),
    sa.Column('raw_analysis', sa.JSON(), nullable=True),
    sa.Column('structured_results', sa.JSON(), nullable=True),
    sa.Column('recommendations', sa.JSON(), nullable=True),
    sa.Column('processing_time', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('agent_used', sa.String(length=100), nullable=True),
    sa.Column('ai_model_used', sa.String(length=100), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('retry_count', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_analyses_created_at', 'bill_analyses', ['created_at'], unique=False)
    op.create_index('idx_analyses_file_hash', 'bill_analyses', ['file_hash'], unique=False)
    op.create_index('idx_analyses_status', 'bill_analyses', ['status'], unique=False)
    op.create_index('idx_analyses_type', 'bill_analyses', ['analysis_type'], unique=False)
    op.create_index('idx_analyses_user_id', 'bill_analyses', ['user_id'], unique=False)
    op.create_index(op.f('ix_bill_analyses_file_hash'), 'bill_analyses', ['file_hash'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_bill_analyses_file_hash'), table_name='bill_analyses')
    op.drop_index('idx_analyses_user_id', table_name='bill_analyses')
    op.drop_index('idx_analyses_type', table_name='bill_analyses')
    op.drop_index('idx_analyses_status', table_name='bill_analyses')
    op.drop_index('idx_analyses_file_hash', table_name='bill_analyses')
    op.drop_index('idx_analyses_created_at', table_name='bill_analyses')
    op.drop_table('bill_analyses')
    op.drop_index('idx_audit_logs_user_id', table_name='audit_logs')
    op.drop_index('idx_audit_logs_resource', table_name='audit_logs')
    op.drop_index('idx_audit_logs_ip', table_name='audit_logs')
    op.drop_index('idx_audit_logs_created_at', table_name='audit_logs')
    op.drop_index('idx_audit_logs_action', table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index(op.f('ix_api_keys_key_hash'), table_name='api_keys')
    op.drop_index('idx_api_keys_user_id', table_name='api_keys')
    op.drop_index('idx_api_keys_hash', table_name='api_keys')
    op.drop_index('idx_api_keys_active', table_name='api_keys')
    op.drop_table('api_keys')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index('idx_users_role', table_name='users')
    op.drop_index('idx_users_email', table_name='users')
    op.drop_index('idx_users_created_at', table_name='users')
    op.drop_table('users')
    op.drop_index('idx_metrics_type', table_name='system_metrics')
    op.drop_index('idx_metrics_timestamp', table_name='system_metrics')
    op.drop_index('idx_metrics_name', table_name='system_metrics')
    op.drop_table('system_metrics')
    # ### end Alembic commands ###
