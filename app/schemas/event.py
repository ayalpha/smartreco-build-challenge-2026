"""Behavioural event schemas.

The ingest endpoint is on the hot path for every page view, so validation is kept
deliberately permissive: unknown metadata keys are preserved, oversized batches
are truncated rather than rejected, and a single malformed event never fails the
whole batch (the router skips it and reports the count).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.event import EventType

#: Hard cap on events accepted in one request (protects the DB from abuse).
MAX_BATCH_SIZE = 200

#: Cap on a single ``time_spent`` measurement (12h) — anything larger is a bug
#: or a tab left open for days, and would skew the behavioural weighting.
MAX_TIME_SPENT_SECONDS = 43_200


class EventIn(BaseModel):
    """A single tracked event as sent by ``tracker.js``."""

    model_config = ConfigDict(extra="allow")

    event_type: str = Field(..., max_length=48)
    product_id: Optional[int] = Field(default=None, ge=1)
    path: Optional[str] = Field(default=None, max_length=500)
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def _known_event_type(cls, value: str) -> str:
        """Reject unknown event types so typos surface immediately."""
        cleaned = value.strip().lower()
        if cleaned not in EventType.values():
            raise ValueError(
                f"Unknown event_type {value!r}; expected one of "
                f"{', '.join(sorted(EventType.values()))}"
            )
        return cleaned

    def collected_metadata(self) -> dict[str, Any]:
        """Merge the explicit ``metadata`` dict with any extra top-level keys.

        ``tracker.js`` sends convenience fields flat (``{"query": "..."}``) as well
        as nested, so both shapes are supported.  Reserved field names are
        stripped, and ``seconds`` is clamped.
        """
        reserved = {"event_type", "product_id", "path", "timestamp", "metadata"}
        merged: dict[str, Any] = {}

        extras = self.model_extra or {}
        for key, value in extras.items():
            if key not in reserved:
                merged[key] = value
        merged.update(self.metadata or {})

        if "seconds" in merged:
            try:
                merged["seconds"] = max(0, min(MAX_TIME_SPENT_SECONDS, int(float(merged["seconds"]))))
            except (TypeError, ValueError):
                merged.pop("seconds", None)

        # Client-supplied titles are never trusted for prompts, but they are
        # useful for debugging, so keep them namespaced and truncated.
        for key in ("query", "source", "product_title", "title", "recommendation_id"):
            if key in merged and isinstance(merged[key], str):
                merged[key] = merged[key][:300]

        return merged


class EventBatchIn(BaseModel):
    """A batch of events flushed by the frontend tracker."""

    session_id: str = Field(default="anonymous", max_length=64)
    events: list[EventIn] = Field(default_factory=list)

    @field_validator("events")
    @classmethod
    def _cap_batch(cls, value: list[EventIn]) -> list[EventIn]:
        """Truncate oversized batches instead of rejecting them."""
        return value[:MAX_BATCH_SIZE]

    @field_validator("session_id")
    @classmethod
    def _clean_session(cls, value: str) -> str:
        """Normalise a blank session id to ``anonymous``."""
        cleaned = value.strip()
        return cleaned or "anonymous"


class EventBatchResponse(BaseModel):
    """Ingest acknowledgement — intentionally tiny and fast to serialise."""

    accepted: int
    rejected: int = 0
    triggered: bool = False
    trigger_reason: Optional[str] = None


class EventOut(BaseModel):
    """Stored event projection (admin/debug views)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int] = None
    session_id: str
    event_type: str
    product_id: Optional[int] = None
    path: Optional[str] = None
    timestamp: Optional[datetime] = None
