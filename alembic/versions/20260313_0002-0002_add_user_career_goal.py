"""Add users.career_goal for Path-aligned recommendations.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-13
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_career_goal"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("career_goal", sa.String(length=160), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "career_goal")
