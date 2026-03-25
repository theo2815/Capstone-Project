"""widen webhook_subscriptions.secret from varchar(255) to text

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-25 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SEC-7: Fernet ciphertext can exceed 255 bytes for long secrets
    op.alter_column(
        'webhook_subscriptions', 'secret',
        type_=sa.Text(),
        existing_type=sa.String(255),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'webhook_subscriptions', 'secret',
        type_=sa.String(255),
        existing_type=sa.Text(),
        existing_nullable=True,
    )
