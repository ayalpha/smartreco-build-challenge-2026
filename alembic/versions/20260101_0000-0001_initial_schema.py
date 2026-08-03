"""Initial schema: users, products, events, recommendations, email_digests.

Revision ID: 0001_initial
Revises: None
Create Date: 2026-01-01 00:00:00

Notes
-----
* Enums are rendered as ``VARCHAR`` + ``CHECK`` (``native_enum=False``) so the
  same migration applies to PostgreSQL and SQLite without a dialect branch.
* ``events.metadata`` is generic ``JSON`` rather than PostgreSQL ``JSONB`` for the
  same portability reason; swap it in a follow-up migration if you settle on
  Postgres and want GIN indexing.
* The composite index ``ix_events_user_timestamp`` is the one that matters most:
  the agent's hot query is "last N events for this user, newest first".
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create every table and index."""

    # ------------------------------------------------------------------ users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("full_name", sa.String(length=160), nullable=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            sa.Enum("USER", "ADMIN", name="userrole", native_enum=False, length=16),
            nullable=False,
            server_default="user",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("preferred_skill_level", sa.String(length=32), nullable=True),
        sa.Column("digest_opt_in", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("telegram_chat_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # --------------------------------------------------------------- products
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=240), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("tags", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("skill_level", sa.String(length=32), nullable=False, server_default="beginner"),
        sa.Column("duration", sa.String(length=64), nullable=True),
        sa.Column("thumbnail_url", sa.String(length=600), nullable=True),
        sa.Column("instructor", sa.String(length=160), nullable=True),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_products_id", "products", ["id"], unique=False)
    op.create_index("ix_products_title", "products", ["title"], unique=False)
    op.create_index("ix_products_category", "products", ["category"], unique=False)
    op.create_index("ix_products_category_active", "products", ["category", "is_active"])
    op.create_index("ix_products_skill_price", "products", ["skill_level", "price"])

    # ----------------------------------------------------------------- events
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=False, server_default="anonymous"),
        sa.Column("event_type", sa.String(length=48), nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=True),
        sa.Column("path", sa.String(length=500), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_events_id", "events", ["id"], unique=False)
    op.create_index("ix_events_user_id", "events", ["user_id"], unique=False)
    op.create_index("ix_events_product_id", "events", ["product_id"], unique=False)
    op.create_index("ix_events_timestamp", "events", ["timestamp"], unique=False)
    # The agent's primary access pattern.
    op.create_index("ix_events_user_timestamp", "events", ["user_id", "timestamp"])
    op.create_index("ix_events_session", "events", ["session_id"])
    op.create_index("ix_events_type_timestamp", "events", ["event_type", "timestamp"])

    # -------------------------------------------------------- recommendations
    op.create_table(
        "recommendations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("narrative", sa.Text(), nullable=False, server_default=""),
        sa.Column("headline", sa.String(length=240), nullable=True),
        sa.Column("products", sa.JSON(), nullable=False),
        sa.Column("interest_signals", sa.JSON(), nullable=False),
        sa.Column("behavior_digest", sa.Text(), nullable=True),
        sa.Column("retrieval_query", sa.String(length=1000), nullable=True),
        sa.Column("trigger_event_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("trigger_reason", sa.String(length=80), nullable=True),
        sa.Column("agent_trace", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("degraded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recommendations_id", "recommendations", ["id"], unique=False)
    op.create_index("ix_recommendations_user_id", "recommendations", ["user_id"], unique=False)
    op.create_index("ix_recommendations_is_active", "recommendations", ["is_active"])
    op.create_index("ix_recommendations_created_at", "recommendations", ["created_at"])
    op.create_index("ix_recommendations_user_active", "recommendations", ["user_id", "is_active"])
    op.create_index("ix_recommendations_user_created", "recommendations", ["user_id", "created_at"])

    # ---------------------------------------------------------- email_digests
    op.create_table(
        "email_digests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("recommendation_id", sa.Integer(), nullable=True),
        sa.Column("channel", sa.String(length=24), nullable=False, server_default="email"),
        sa.Column("backend", sa.String(length=24), nullable=False, server_default="console"),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="sent"),
        sa.Column("subject", sa.String(length=300), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["recommendation_id"], ["recommendations.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_email_digests_id", "email_digests", ["id"], unique=False)
    op.create_index("ix_email_digests_user_id", "email_digests", ["user_id"], unique=False)
    op.create_index("ix_email_digests_sent_at", "email_digests", ["sent_at"], unique=False)
    op.create_index("ix_email_digests_user_sent", "email_digests", ["user_id", "sent_at"])


def downgrade() -> None:
    """Drop everything, children first to satisfy foreign keys."""
    op.drop_index("ix_email_digests_user_sent", table_name="email_digests")
    op.drop_index("ix_email_digests_sent_at", table_name="email_digests")
    op.drop_index("ix_email_digests_user_id", table_name="email_digests")
    op.drop_index("ix_email_digests_id", table_name="email_digests")
    op.drop_table("email_digests")

    op.drop_index("ix_recommendations_user_created", table_name="recommendations")
    op.drop_index("ix_recommendations_user_active", table_name="recommendations")
    op.drop_index("ix_recommendations_created_at", table_name="recommendations")
    op.drop_index("ix_recommendations_is_active", table_name="recommendations")
    op.drop_index("ix_recommendations_user_id", table_name="recommendations")
    op.drop_index("ix_recommendations_id", table_name="recommendations")
    op.drop_table("recommendations")

    op.drop_index("ix_events_type_timestamp", table_name="events")
    op.drop_index("ix_events_session", table_name="events")
    op.drop_index("ix_events_user_timestamp", table_name="events")
    op.drop_index("ix_events_timestamp", table_name="events")
    op.drop_index("ix_events_product_id", table_name="events")
    op.drop_index("ix_events_user_id", table_name="events")
    op.drop_index("ix_events_id", table_name="events")
    op.drop_table("events")

    op.drop_index("ix_products_skill_price", table_name="products")
    op.drop_index("ix_products_category_active", table_name="products")
    op.drop_index("ix_products_category", table_name="products")
    op.drop_index("ix_products_title", table_name="products")
    op.drop_index("ix_products_id", table_name="products")
    op.drop_table("products")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
