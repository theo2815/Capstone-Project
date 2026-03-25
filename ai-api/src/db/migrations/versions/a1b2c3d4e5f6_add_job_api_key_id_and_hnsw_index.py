"""add job api_key_id and face_embeddings HNSW index

Revision ID: a1b2c3d4e5f6
Revises: bf56c918b961
Create Date: 2026-03-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'bf56c918b961'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add api_key_id to jobs for tenant isolation
    op.add_column('jobs', sa.Column('api_key_id', sa.String(255), nullable=True))
    op.create_index('ix_jobs_api_key_id', 'jobs', ['api_key_id'])

    # Add HNSW index on face_embeddings.embedding for fast vector search
    # This replaces the sequential scan with an approximate nearest-neighbor index
    op.execute(
        """
        CREATE INDEX ix_face_embeddings_embedding_hnsw
        ON face_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.drop_index('ix_face_embeddings_embedding_hnsw', table_name='face_embeddings')
    op.drop_index('ix_jobs_api_key_id', table_name='jobs')
    op.drop_column('jobs', 'api_key_id')
