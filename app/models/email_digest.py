"""Audit trail for proactively delivered digests (BONUS 2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.recommendation import Recommendation
    from app.models.user import User


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmailDigest(Base):
    """One outbound digest attempt (email and/or Telegram).

    Rows are written for failures too, with ``status="failed"`` and the error in
    ``error``, so the scheduler's behaviour is fully auditable.
    """

    __tablename__ = "email_digests"
    __table_args__ = (Index("ix_email_digests_user_sent", "user_id", "sent_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    recommendation_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL"), nullable=True
    )

    channel: Mapped[str] = mapped_column(String(24), default="email", nullable=False)
    backend: Mapped[str] = mapped_column(String(24), default="console", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="sent", nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped["User"] = relationship(back_populates="digests")
    recommendation: Mapped[Optional["Recommendation"]] = relationship(back_populates="digests")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<EmailDigest id={self.id} user={self.user_id} "
            f"channel={self.channel} status={self.status}>"
        )
