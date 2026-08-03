"""Recommendation schemas consumed by the polling frontend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class InterestSignalOut(BaseModel):
    """One interest signal, rendered in the "Why this recommendation?" panel."""

    topic: str
    confidence: float = 0.0
    evidence: str = ""


class RecommendationProductOut(BaseModel):
    """A recommended product plus the agent's per-product pitch."""

    id: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    price: Optional[float] = None
    skill_level: Optional[str] = None
    duration: Optional[str] = None
    thumbnail_url: Optional[str] = None
    instructor: Optional[str] = None
    rating: Optional[float] = None
    pitch: str = ""
    reason: str = ""
    relevance_score: Optional[float] = None
    retrieval_mode: Optional[str] = None


class RecommendationOut(BaseModel):
    """Full recommendation payload."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    headline: Optional[str] = None
    narrative: str = ""
    products: list[RecommendationProductOut] = Field(default_factory=list)
    interest_signals: list[InterestSignalOut] = Field(default_factory=list)
    trigger_event_count: int = 0
    trigger_reason: Optional[str] = None
    degraded: bool = False
    latency_ms: Optional[float] = None
    created_at: Optional[datetime] = None


class RecommendationStatusOut(BaseModel):
    """Response for the 60-second polling endpoint.

    ``generating`` drives the loading skeleton in the UI: it is True when a run
    has been dispatched but has not yet stored its result.
    """

    has_recommendation: bool
    generating: bool = False
    pending_reason: Optional[str] = None
    recommendation: Optional[RecommendationOut] = None
    next_trigger: Optional[dict[str, Any]] = None
    served_from_cache: bool = False


class TriggerDecisionOut(BaseModel):
    """Diagnostics for the trigger policy (admin dashboard + tests)."""

    should_run: bool
    reason: str
    event_count: int
    new_events: int
    detail: str = ""
