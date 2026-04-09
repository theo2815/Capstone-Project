"""Tune HNSW parameters and add GIN index for webhook events

Upgrades the face embeddings HNSW index from m=16,ef_construction=64 to
m=24,ef_construction=128 for better recall at production scale (10k+ enrolled
faces).  Adds a GIN index on webhook_subscriptions.events so that the JSONB
@> operator used on every job completion no longer does a full table scan.

NOTE: The HNSW rebuild drops and recreates the index, which will briefly lock
the face_embeddings table.  For a zero-downtime rebuild on a live system with
data, run the following *outside* Alembic (cannot run inside a transaction):

    DROP INDEX CONCURRENTLY IF EXISTS ix_face_embeddings_embedding_hnsw;
    CREATE INDEX CONCURRENTLY ix_face_embeddings_embedding_hnsw
        ON face_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128);

    CREATE INDEX CONCURRENTLY ix_webhook_subscriptions_events_gin
        ON webhook_subscriptions
        USING gin (events);

Revision ID: h8c9d0e1f2g3
Revises: g7b8c9d0e1f2
Create Date: 2026-03-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'h8c9d0e1f2g3'
down_revision: Union[str, None] = 'g7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- HNSW index: rebuild with better parameters ---
    # m=24 vs 16: ~15% more memory, better graph connectivity for 512-dim vectors
    # ef_construction=128 vs 64: ~2x build time, significantly better recall at scale
    op.drop_index(
        'ix_face_embeddings_embedding_hnsw',
        table_name='face_embeddings',
        if_exists=True,
    )
    op.execute("""
        CREATE INDEX ix_face_embeddings_embedding_hnsw
        ON face_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 128)
    """)

    # --- GIN index on webhook_subscriptions.events ---
    # Fixes: JSONB @> operator used in list_by_event() does a full table scan
    # without this index.  Called on every job completion + failure.
    op.create_index(
        'ix_webhook_subscriptions_events_gin',
        'webhook_subscriptions',
        ['events'],
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('ix_webhook_subscriptions_events_gin', table_name='webhook_subscriptions')

    op.drop_index(
        'ix_face_embeddings_embedding_hnsw',
        table_name='face_embeddings',
        if_exists=True,
    )
    op.execute("""
        CREATE INDEX ix_face_embeddings_embedding_hnsw
        ON face_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
