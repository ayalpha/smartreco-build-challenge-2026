"""Catalog schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.product import SKILL_LEVELS


class ProductBase(BaseModel):
    """Fields shared by create and update payloads."""

    title: str = Field(..., min_length=3, max_length=240)
    description: str = Field(default="", max_length=8000)
    category: str = Field(..., min_length=2, max_length=80)
    tags: Union[list[str], str] = Field(
        default_factory=list, description="List of tags, or a comma-separated string"
    )
    price: float = Field(default=0.0, ge=0, le=100000)
    skill_level: str = Field(default="beginner")
    duration: Optional[str] = Field(default=None, max_length=64)
    thumbnail_url: Optional[str] = Field(default=None, max_length=600)
    instructor: Optional[str] = Field(default=None, max_length=160)
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    is_active: bool = True

    @field_validator("skill_level")
    @classmethod
    def _valid_skill_level(cls, value: str) -> str:
        """Constrain skill level to the canonical set (case-insensitive)."""
        cleaned = value.strip().lower()
        if cleaned not in SKILL_LEVELS:
            raise ValueError(f"skill_level must be one of {', '.join(SKILL_LEVELS)}")
        return cleaned

    @field_validator("category")
    @classmethod
    def _clean_category(cls, value: str) -> str:
        """Trim surrounding whitespace from the category label."""
        return value.strip()

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: Union[list[str], str]) -> list[str]:
        """Normalise tags into a de-duplicated, lower-cased list."""
        items = value if isinstance(value, list) else str(value).split(",")
        out: list[str] = []
        for item in items:
            tag = str(item).strip().lower()
            if tag and tag not in out:
                out.append(tag)
        return out


class ProductCreate(ProductBase):
    """Payload for creating a catalog entry."""


class ProductUpdate(BaseModel):
    """Payload for a partial catalog update — every field is optional."""

    title: Optional[str] = Field(default=None, min_length=3, max_length=240)
    description: Optional[str] = Field(default=None, max_length=8000)
    category: Optional[str] = Field(default=None, min_length=2, max_length=80)
    tags: Optional[Union[list[str], str]] = None
    price: Optional[float] = Field(default=None, ge=0, le=100000)
    skill_level: Optional[str] = None
    duration: Optional[str] = Field(default=None, max_length=64)
    thumbnail_url: Optional[str] = Field(default=None, max_length=600)
    instructor: Optional[str] = Field(default=None, max_length=160)
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    is_active: Optional[bool] = None

    @field_validator("skill_level")
    @classmethod
    def _valid_skill_level(cls, value: Optional[str]) -> Optional[str]:
        """Constrain skill level to the canonical set when provided."""
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in SKILL_LEVELS:
            raise ValueError(f"skill_level must be one of {', '.join(SKILL_LEVELS)}")
        return cleaned

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: Optional[Union[list[str], str]]) -> Optional[list[str]]:
        """Normalise tags when provided, leaving None untouched."""
        if value is None:
            return None
        items = value if isinstance(value, list) else str(value).split(",")
        out: list[str] = []
        for item in items:
            tag = str(item).strip().lower()
            if tag and tag not in out:
                out.append(tag)
        return out


class ProductOut(BaseModel):
    """Public projection of a catalog entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    price: float
    skill_level: str
    duration: Optional[str] = None
    thumbnail_url: Optional[str] = None
    instructor: Optional[str] = None
    rating: Optional[float] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_from_model(cls, value: Any) -> list[str]:
        """Accept the ORM's comma-separated string as well as a real list."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [tag.strip() for tag in str(value).split(",") if tag.strip()]


class ProductListResponse(BaseModel):
    """Paginated catalog listing."""

    items: list[ProductOut]
    total: int
    page: int
    page_size: int
    query: Optional[str] = None
    category: Optional[str] = None
