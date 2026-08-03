"""Recommendation model — the persisted output of the LangGraph agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.email_digest import EmailDigest
    from app.models.user import User


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Recommendation(Base):
    """One agent run's persuasive output for a single user.

    Only one row per user is ``is_active`` at a time: the
    ``recommendation_storer`` node deactivates previous rows inside the same
    transaction that inserts the new one.
    """

    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_recommendations_user_active", "user_id", "is_active"),
        Index("ix_recommendations_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: The persuasive narrative written by the ``persuasion_writer`` node.
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")
    headline: Mapped[Optional[str]] = mapped_column(String(240), nullable=True)

    #: List of product dicts, each with an agent-written ``reason``/``pitch``.
    products: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    #: ``[{"topic", "confidence", "evidence"}]`` — powers "Why this?" in the UI.
    interest_signals: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )

    behavior_digest: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieval_query: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    trigger_event_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    trigger_reason: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)

    #: Diagnostics captured from the graph run (node timings, retries, models).
    agent_trace: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped["User"] = relationship(back_populates="recommendations")
    digests: Mapped[list["EmailDigest"]] = relationship(
        back_populates="recommendation", cascade="all, delete-orphan", passive_deletes=True
    )

    # -------------------------------------------------------------- helpers
    @property
    def product_ids(self) -> list[int]:
        """IDs of the recommended products, skipping malformed entries."""
        ids: list[int] = []
        for item in self.products or []:
            raw = item.get("id") if isinstance(item, dict) else None
            if isinstance(raw, int):
                ids.append(raw)
        return ids

    @property
    def top_signal(self) -> Optional[str]:
        """Highest-confidence interest topic, for compact UI badges."""
        signals = [s for s in (self.interest_signals or []) if isinstance(s, dict)]
        if not signals:
            return None
        best = max(signals, key=lambda s: float(s.get("confidence") or 0.0))
        topic = best.get("topic")
        return str(topic) if topic else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable projection consumed by the frontend poller."""
        return {
            "id": self.id,
            "headline": self.headline,
            "narrative": self.narrative,
            "products": self.products or [],
            "interest_signals": self.interest_signals or [],
            "trigger_event_count": self.trigger_event_count,
            "trigger_reason": self.trigger_reason,
            "degraded": self.degraded,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Recommendation id={self.id} user={self.user_id} "
            f"products={len(self.products or [])} active={self.is_active}>"
        )
