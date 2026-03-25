"""add persons.api_key_id for tenant isolation

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add api_key_id to persons for tenant isolation of biometric data
    op.add_column('persons', sa.Column('api_key_id', sa.String(255), nullable=True))
    op.create_index('ix_persons_api_key_id', 'persons', ['api_key_id'])


def downgrade() -> None:
    op.drop_index('ix_persons_api_key_id', table_name='persons')
    op.drop_column('persons', 'api_key_id')
