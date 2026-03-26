"""add unique constraint on face_embeddings (person_id, source_image_hash)

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-26 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_face_embeddings_person_hash',
        'face_embeddings',
        ['person_id', 'source_image_hash'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('ix_face_embeddings_person_hash', table_name='face_embeddings')
