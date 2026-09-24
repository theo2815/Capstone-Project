"""Track worker attempt ownership and lease expiry for replay-safe async jobs.

Revision ID: j0e1f2g3h4i5
Revises: i9d0e1f2g3h4
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "j0e1f2g3h4i5"
down_revision = "i9d0e1f2g3h4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("attempt_token", sa.String(64), nullable=True))
    op.add_column("jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "lease_expires_at")
    op.drop_column("jobs", "attempt_token")
