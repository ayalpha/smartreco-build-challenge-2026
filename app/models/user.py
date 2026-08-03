"""User model and role enumeration."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Enum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.models.email_digest import EmailDigest
    from app.models.event import Event
    from app.models.recommendation import Recommendation


class UserRole(str, enum.Enum):
    """Authorisation roles.

    ``USER`` browses the catalog and receives AI recommendations; ``ADMIN``
    additionally manages the product catalog.
    """

    USER = "user"
    ADMIN = "admin"


def _utcnow() -> datetime:
    """Timezone-aware UTC now (SQLite has no native ``now()`` with tz)."""
    return datetime.now(timezone.utc)


class User(Base):
    """A registered account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=16, validate_strings=True),
        default=UserRole.USER,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # --- personalisation hints, learned from behaviour or set by the user ---
    preferred_skill_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    digest_opt_in: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ------------------------------------------------------------ relations
    events: Mapped[list["Event"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    digests: Mapped[list["EmailDigest"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    # -------------------------------------------------------------- helpers
    @property
    def is_admin(self) -> bool:
        """True when this account may manage the catalog."""
        return self.role == UserRole.ADMIN

    @property
    def display_name(self) -> str:
        """Human-friendly label for templates and email greetings."""
        if self.full_name:
            return self.full_name
        return self.email.split("@", 1)[0]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User id={self.id} email={self.email!r} role={self.role.value}>"
