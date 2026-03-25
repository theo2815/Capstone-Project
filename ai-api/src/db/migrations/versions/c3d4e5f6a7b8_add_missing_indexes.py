"""add missing indexes on frequently queried columns

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # DB-1: Add indexes on columns used in WHERE clauses
    op.create_index('ix_jobs_status', 'jobs', ['status'])
    op.create_index('ix_jobs_created_at', 'jobs', ['created_at'])
    op.create_index('ix_webhook_subscriptions_api_key_id', 'webhook_subscriptions', ['api_key_id'])
    op.create_index('ix_webhook_subscriptions_active', 'webhook_subscriptions', ['active'])
    op.create_index('ix_face_embeddings_source_image_hash', 'face_embeddings', ['source_image_hash'])


def downgrade() -> None:
    op.drop_index('ix_face_embeddings_source_image_hash', table_name='face_embeddings')
    op.drop_index('ix_webhook_subscriptions_active', table_name='webhook_subscriptions')
    op.drop_index('ix_webhook_subscriptions_api_key_id', table_name='webhook_subscriptions')
    op.drop_index('ix_jobs_created_at', table_name='jobs')
    op.drop_index('ix_jobs_status', table_name='jobs')
