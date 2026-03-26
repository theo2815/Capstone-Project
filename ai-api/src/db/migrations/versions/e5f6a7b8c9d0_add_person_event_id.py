"""add event_id column to persons table for event isolation

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('persons', sa.Column('event_id', sa.String(255), nullable=True))
    op.create_index('ix_persons_event_id', 'persons', ['event_id'])


def downgrade() -> None:
    op.drop_index('ix_persons_event_id', table_name='persons')
    op.drop_column('persons', 'event_id')
