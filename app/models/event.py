"""Behavioural event model.

Events are the raw fuel for the recommendation agent.  The table is
write-heavy and read by ``(user_id, timestamp DESC)``, so that composite index
is declared explicitly.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.user import User


class EventType(str, enum.Enum):
    """The behavioural signals the frontend tracker emits."""

    PAGE_VIEW = "page_view"
    PRODUCT_CLICK = "product_click"
    SEARCH_QUERY = "search_query"
    TIME_SPENT = "time_spent"
    ADD_TO_CART = "add_to_cart"
    RECOMMENDATION_CLICK = "recommendation_click"

    @classmethod
    def values(cls) -> set[str]:
        """All valid event-type strings."""
        return {member.value for member in cls}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(Base):
    """A single tracked user interaction.

    ``metadata_json`` holds the event-specific payload (search text, seconds
    spent, click source …).  It is deliberately schemaless so new frontend
    signals need no migration.  The attribute is *not* called ``metadata``
    because that name is reserved by SQLAlchemy's declarative API; the column in
    the database is still ``metadata``.
    """

    __tablename__ = "events"
    __table_args__ = (
        # Primary access pattern: "recent events for this user".
        Index("ix_events_user_timestamp", "user_id", "timestamp"),
        Index("ix_events_session", "session_id"),
        Index("ix_events_type_timestamp", "event_type", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, default="anonymous")
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    product_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("products.id", ondelete="SET NULL"), nullable=True, index=True
    )
    path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped[Optional["User"]] = relationship(back_populates="events")

    # -------------------------------------------------------------- helpers
    def to_dict(self) -> dict[str, Any]:
        """Compact dict passed into the agent's ``raw_events`` state slot."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "product_id": self.product_id,
            "path": self.path,
            "metadata": self.metadata_json or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def describe(self, product_title: Optional[str] = None) -> str:
        """One-line human/LLM-readable rendering of the event.

        Used to build the prompt for the ``activity_analyzer`` node — keeping
        this in the model means prompt text and schema evolve together.

        Args:
            product_title: Authoritative title looked up from the catalog. Takes
                precedence over any client-supplied title in ``metadata``, which
                must never be trusted for prompt content.
        """
        meta = self.metadata_json or {}
        stamp = self.timestamp.strftime("%Y-%m-%d %H:%M") if self.timestamp else "unknown-time"
        title = product_title or meta.get("product_title") or meta.get("title")

        if self.event_type == EventType.SEARCH_QUERY.value:
            return (
                f"[{stamp}] searched for {meta.get('query', '')!r} "
                f"({meta.get('result_count', 0)} results)"
            )
        if self.event_type == EventType.TIME_SPENT.value:
            return (
                f"[{stamp}] spent {meta.get('seconds', 0)}s on "
                f"{title or f'product #{self.product_id}'}"
            )
        if self.event_type == EventType.PRODUCT_CLICK.value:
            return (
                f"[{stamp}] clicked {title or f'product #{self.product_id}'} "
                f"(from {meta.get('source', 'unknown')})"
            )
        if self.event_type == EventType.ADD_TO_CART.value:
            return f"[{stamp}] added {title or f'product #{self.product_id}'} to cart"
        if self.event_type == EventType.RECOMMENDATION_CLICK.value:
            return (
                f"[{stamp}] clicked recommended {title or f'product #{self.product_id}'} "
                f"(recommendation #{meta.get('recommendation_id')})"
            )
        return f"[{stamp}] viewed page {self.path or meta.get('path', '/')}"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Event id={self.id} user={self.user_id} type={self.event_type!r} "
            f"product={self.product_id}>"
        )
