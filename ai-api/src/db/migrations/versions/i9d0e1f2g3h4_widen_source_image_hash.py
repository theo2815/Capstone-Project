"""widen face_embeddings.source_image_hash for per-face suffix

The multi-face enroll fix stores each detected face under a per-face-distinct
key `f"{image_hash}:{face_index}"` so several runners in one crowd shot no longer
collapse on the (person_id, source_image_hash) unique index. But `image_hash` is
a SHA-256 hexdigest — exactly 64 characters, which already fully saturates the
original VARCHAR(64) column. Appending `:{i}` makes every value 66+ chars, so
every enroll INSERT would raise StringDataRightTruncation (22001) against real
PostgreSQL. Widen the column so the suffixed key fits.

Increasing a varchar length is a metadata-only change in PostgreSQL (no table
rewrite, no index rebuild), so the existing index and the
(person_id, source_image_hash) unique constraint remain valid.

Revision ID: i9d0e1f2g3h4
Revises: h8c9d0e1f2g3
Create Date: 2026-06-02 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'i9d0e1f2g3h4'
down_revision: Union[str, None] = 'h8c9d0e1f2g3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'face_embeddings',
        'source_image_hash',
        existing_type=sa.String(length=64),
        type_=sa.String(length=80),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'face_embeddings',
        'source_image_hash',
        existing_type=sa.String(length=80),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
