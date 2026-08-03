"""Product (course) model — the catalog entity that is dual-written to Qdrant."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: Canonical skill levels used for metadata filtering during retrieval.
SKILL_LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced")


class Product(Base):
    """A course / product in the learning marketplace.

    Every mutation of this table is *dual-written*: the SQL row is the system of
    record, and an embedded mirror is upserted into (or deleted from) the Qdrant
    collection by :mod:`app.vector_store.sync`.
    """

    __tablename__ = "products"
    __table_args__ = (
        Index("ix_products_category_active", "category", "is_active"),
        Index("ix_products_skill_price", "skill_level", "price"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(240), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(80), nullable=False, index=True)

    #: Comma-separated tag list.  Stored as text for portability across
    #: PostgreSQL and SQLite; exposed as a real list via :attr:`tag_list`.
    tags: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    skill_level: Mapped[str] = mapped_column(String(32), nullable=False, default="beginner")
    duration: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(600), nullable=True)
    instructor: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    #: Bumped on every content change so the vector mirror can be audited.
    revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    # -------------------------------------------------------------- helpers
    @property
    def tag_list(self) -> list[str]:
        """Tags as a cleaned list of lower-cased strings."""
        return [t.strip().lower() for t in (self.tags or "").split(",") if t.strip()]

    @staticmethod
    def normalise_tags(raw: Any) -> str:
        """Normalise user input (list or comma string) into the stored format.

        Args:
            raw: A list of tags or a comma-separated string.

        Returns:
            A comma-separated, de-duplicated, lower-cased tag string.
        """
        if raw is None:
            return ""
        items = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
        seen: list[str] = []
        for item in items:
            tag = str(item).strip().lower()
            if tag and tag not in seen:
                seen.append(tag)
        return ", ".join(seen)

    def embedding_text(self) -> str:
        """Build the natural-language document that gets embedded for retrieval.

        Packing the structured attributes into prose measurably improves dense
        retrieval quality versus embedding the title alone.
        """
        parts = [
            f"{self.title}.",
            f"Category: {self.category}.",
            f"Skill level: {self.skill_level}.",
        ]
        if self.duration:
            parts.append(f"Duration: {self.duration}.")
        if self.tag_list:
            parts.append("Topics: " + ", ".join(self.tag_list) + ".")
        if self.instructor:
            parts.append(f"Instructor: {self.instructor}.")
        parts.append(self.description or "")
        return " ".join(p for p in parts if p).strip()

    def keyword_text(self) -> str:
        """Lower-cased bag of words used by the BM25 half of hybrid search."""
        return " ".join(
            [self.title, self.category, " ".join(self.tag_list), self.description or ""]
        ).lower()

    def to_public_dict(self) -> dict[str, Any]:
        """Serialise to the plain dict shape used by the agent and templates."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": self.tag_list,
            "price": self.price,
            "skill_level": self.skill_level,
            "duration": self.duration,
            "thumbnail_url": self.thumbnail_url,
            "instructor": self.instructor,
            "rating": self.rating,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Product id={self.id} title={self.title!r} category={self.category!r}>"
