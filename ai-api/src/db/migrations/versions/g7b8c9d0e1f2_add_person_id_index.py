"""add standalone index on face_embeddings.person_id for CASCADE delete

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-03-26 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'g7b8c9d0e1f2'
down_revision: Union[str, None] = 'f6a7b8c9d0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'ix_face_embeddings_person_id',
        'face_embeddings',
        ['person_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_face_embeddings_person_id', table_name='face_embeddings')
