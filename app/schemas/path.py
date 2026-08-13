"""Learning-path request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PathRequest(BaseModel):
    """JSON body for ``POST /api/path``."""

    goal: str = Field(..., min_length=3, max_length=160)
    weekly_hours: int = Field(default=5, ge=1, le=40)


class PathStepOut(BaseModel):
    """One ordered course step on a path."""

    order: int
    title: str
    product_id: int
    level: Optional[str] = None
    duration: Optional[str] = None
    weeks: int = 1
    why: str = ""
    milestone: str = ""


class PathOut(BaseModel):
    """Full path payload returned by the builder and the JSON API."""

    headline: str
    summary: str
    steps: list[PathStepOut] = Field(default_factory=list)
    degraded: bool = False
    interests: list[str] = Field(default_factory=list)
    goal: str = ""
    weekly_hours: int = 5

    @classmethod
    def from_builder(
        cls,
        path: dict[str, Any],
        *,
        goal: str,
        weekly_hours: int,
    ) -> "PathOut":
        """Project the internal builder dict into a response model."""
        steps = [
            PathStepOut(
                order=int(step.get("order") or index),
                title=str(step.get("title") or f"Course #{step.get('product_id')}"),
                product_id=int(step["product_id"]),
                level=step.get("level"),
                duration=step.get("duration"),
                weeks=int(step.get("weeks") or 1),
                why=str(step.get("why") or ""),
                milestone=str(step.get("milestone") or ""),
            )
            for index, step in enumerate(path.get("steps") or [], start=1)
            if step.get("product_id") is not None
        ]
        return cls(
            headline=str(path.get("headline") or f"Your path toward {goal}"),
            summary=str(path.get("summary") or ""),
            steps=steps,
            degraded=bool(path.get("degraded")),
            interests=[str(i) for i in (path.get("interests") or []) if i],
            goal=goal,
            weekly_hours=weekly_hours,
        )
