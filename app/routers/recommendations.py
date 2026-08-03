"""Recommendation surfaces: the "For You" homepage, the profile page and the
polling API that keeps them fresh.

The polling contract (consumed by ``static/js/recommendations.js``) is:

``GET /api/recommendations/latest`` →
    ``{has_recommendation, generating, pending_reason, recommendation, next_trigger}``

``generating`` is what drives the loading skeleton: it is true while a dispatched
run has not yet stored its result.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.runner import run_agent_now
from app.agent.triggers import REASON_MANUAL, active_recommendation, evaluate
from app.cache import (
    acquire_agent_lock,
    cache_active_recommendation,
    get_agent_pending,
    get_cached_recommendation,
    mark_agent_pending,
    release_agent_lock,
)
from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user, get_current_user_optional, render_page
from app.models.event import Event
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.schemas.recommendation import (
    RecommendationOut,
    RecommendationStatusOut,
    TriggerDecisionOut,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["recommendations"])
api_router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


# --------------------------------------------------------------------------- #
# HTML pages                                                                  #
# --------------------------------------------------------------------------- #

@router.get("/", include_in_schema=False)
def home_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the homepage, including the "For You" panel for signed-in users."""
    featured = list(
        db.scalars(
            select(Product)
            .where(Product.is_active.is_(True))
            .order_by(Product.rating.desc().nullslast(), Product.id.desc())
            .limit(6)
        )
    )
    categories = [
        str(row[0])
        for row in db.execute(
            select(Product.category)
            .where(Product.is_active.is_(True))
            .distinct()
            .order_by(Product.category)
        ).all()
        if row[0]
    ]

    recommendation = active_recommendation(db, user.id) if user else None
    pending = get_agent_pending(user.id) if user else None

    return render_page(
        request,
        "index.html",
        user,
        featured=featured,
        categories=categories,
        recommendation=recommendation,
        pending_reason=pending,
        product_count=len(featured),
    )


@router.get("/profile", include_in_schema=False)
def profile_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    """Render the profile page: activity summary and recommendation history."""
    history = list(
        db.scalars(
            select(Recommendation)
            .where(Recommendation.user_id == user.id)
            .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
            .limit(20)
        )
    )

    events = list(
        db.scalars(
            select(Event)
            .where(Event.user_id == user.id)
            .order_by(Event.timestamp.desc(), Event.id.desc())
            .limit(25)
        )
    )
    titles = {
        int(row[0]): str(row[1])
        for row in db.execute(
            select(Product.id, Product.title).where(
                Product.id.in_([e.product_id for e in events if e.product_id] or [-1])
            )
        ).all()
    }
    activity = [
        {
            "event_type": event.event_type,
            "description": event.describe(product_title=titles.get(event.product_id or -1)),
            "timestamp": event.timestamp,
        }
        for event in events
    ]

    counts: dict[str, int] = {}
    for row in db.execute(select(Event.event_type).where(Event.user_id == user.id)).all():
        key = str(row[0])
        counts[key] = counts.get(key, 0) + 1

    decision = evaluate(db, user.id)

    return render_page(
        request,
        "profile.html",
        user,
        history=history,
        activity=activity,
        event_counts=counts,
        total_events=sum(counts.values()),
        trigger=decision.as_dict(),
    )


# --------------------------------------------------------------------------- #
# JSON API                                                                    #
# --------------------------------------------------------------------------- #

def _serialise(record: Recommendation) -> RecommendationOut:
    """Convert an ORM row into the API projection."""
    return RecommendationOut.model_validate(record.to_dict())


@api_router.get("/latest", response_model=RecommendationStatusOut)
def latest_recommendation_endpoint(
    include_trigger: bool = Query(default=True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RecommendationStatusOut:
    """Return the active recommendation, or the generating state.

    Called every 60 seconds by the frontend poller.  The Redis result cache
    absorbs that traffic: the DB is only touched on a cache miss or when the
    trigger diagnostics are requested.
    """
    pending = get_agent_pending(user.id)

    cached = get_cached_recommendation(user.id)
    if cached and not include_trigger:
        return RecommendationStatusOut(
            has_recommendation=True,
            generating=bool(pending),
            pending_reason=pending,
            recommendation=RecommendationOut.model_validate(cached),
            served_from_cache=True,
        )

    record = active_recommendation(db, user.id)
    next_trigger: Optional[dict[str, Any]] = None
    if include_trigger:
        next_trigger = TriggerDecisionOut(**evaluate(db, user.id).as_dict()).model_dump()

    if record is None:
        return RecommendationStatusOut(
            has_recommendation=False,
            generating=bool(pending),
            pending_reason=pending,
            next_trigger=next_trigger,
        )

    payload = record.to_dict()
    cache_active_recommendation(user.id, payload)

    return RecommendationStatusOut(
        has_recommendation=True,
        generating=bool(pending),
        pending_reason=pending,
        recommendation=RecommendationOut.model_validate(payload),
        next_trigger=next_trigger,
        served_from_cache=False,
    )


@api_router.post("/refresh", response_model=RecommendationStatusOut)
def refresh_recommendation(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RecommendationStatusOut:
    """Force a synchronous agent run for the signed-in user.

    Powers the "Refresh my picks" button.  The per-user lock is honoured, so
    hammering the button cannot start parallel runs.

    Raises:
        HTTPException: ``409`` when a run is already in flight.
    """
    if not acquire_agent_lock(user.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A recommendation is already being generated for you. Please wait.",
        )

    mark_agent_pending(user.id, REASON_MANUAL)
    try:
        # The lock is already held by this request, so the run must not try to
        # claim it again — it would collide with itself and be skipped.
        result = run_agent_now(user.id, reason=REASON_MANUAL, respect_lock=False)
    finally:
        release_agent_lock(user.id)

    if not result.ok:
        logger.warning("Manual refresh failed for user=%s: %s", user.id, result.error)

    # The agent commits from its own session. This request's session opened a
    # read transaction earlier (resolving the current user), so it still holds a
    # snapshot from before that commit. Ending it here means the query below sees
    # the freshly-written recommendation.
    db.rollback()

    record = active_recommendation(db, user.id)
    return RecommendationStatusOut(
        has_recommendation=record is not None,
        generating=False,
        recommendation=_serialise(record) if record else None,
        next_trigger=TriggerDecisionOut(**evaluate(db, user.id).as_dict()).model_dump(),
    )


@api_router.get("/history", response_model=list[RecommendationOut])
def recommendation_history(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[RecommendationOut]:
    """Return the user's recommendation history, newest first."""
    rows = list(
        db.scalars(
            select(Recommendation)
            .where(Recommendation.user_id == user.id)
            .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
            .limit(limit)
        )
    )
    return [_serialise(row) for row in rows]


@api_router.get("/trigger", response_model=TriggerDecisionOut)
def trigger_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> TriggerDecisionOut:
    """Explain whether the agent would run right now, and why."""
    return TriggerDecisionOut(**evaluate(db, user.id).as_dict())


@api_router.get("/{recommendation_id}", response_model=RecommendationOut)
def get_recommendation(
    recommendation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RecommendationOut:
    """Fetch one of the user's own recommendations.

    Raises:
        HTTPException: ``404`` when it does not exist or belongs to someone else.
    """
    record = db.get(Recommendation, recommendation_id)
    if record is None or record.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recommendation not found"
        )
    return _serialise(record)
