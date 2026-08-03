"""Smart triggering policy — decide *when* the agent should run.

Running the graph on every event would be wasteful (each run costs several Mesh
calls) and running it once per session would make recommendations stale.  The
policy below encodes three rules, checked in priority order:

1. **first_time** — the user has events but no recommendation yet.
2. **event_threshold** — ``AGENT_EVENT_TRIGGER_INTERVAL`` new events have
   accumulated since the last recommendation (i.e. every 10th new event).
3. **stale** — the last recommendation is older than ``AGENT_STALE_HOURS`` *and*
   at least ``AGENT_STALE_MIN_EVENTS`` new events exist.

The decision is pure and side-effect free, which makes it directly unit-testable;
:mod:`app.agent.runner` owns the locking and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.event import Event
from app.models.recommendation import Recommendation

logger = logging.getLogger(__name__)
settings = get_settings()

#: Canonical trigger reason strings (also used as LangSmith tags).
REASON_FIRST_TIME = "first_time"
REASON_EVENT_THRESHOLD = "event_threshold"
REASON_STALE = "stale"
REASON_MANUAL = "manual"
REASON_SCHEDULED_DIGEST = "scheduled_digest"
REASON_NONE = "none"


@dataclass(frozen=True)
class TriggerDecision:
    """The outcome of evaluating the trigger policy for one user."""

    should_run: bool
    reason: str
    event_count: int
    new_events: int
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable projection for the API and logs."""
        return {
            "should_run": self.should_run,
            "reason": self.reason,
            "event_count": self.event_count,
            "new_events": self.new_events,
            "detail": self.detail,
        }


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Coerce a possibly naive datetime to UTC-aware.

    SQLite round-trips ``DateTime(timezone=True)`` as naive values, so timestamps
    read back from the DB must be normalised before arithmetic.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def count_events(db: Session, user_id: int, since: Optional[datetime] = None) -> int:
    """Count a user's events, optionally only those after ``since``."""
    query = select(func.count()).select_from(Event).where(Event.user_id == user_id)
    if since is not None:
        query = query.where(Event.timestamp > since)
    return int(db.scalar(query) or 0)


def latest_recommendation(db: Session, user_id: int) -> Optional[Recommendation]:
    """Return the user's most recent recommendation, active or not."""
    return db.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
        .limit(1)
    ).first()


def active_recommendation(db: Session, user_id: int) -> Optional[Recommendation]:
    """Return the user's current active recommendation, if any."""
    return db.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user_id, Recommendation.is_active.is_(True))
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
        .limit(1)
    ).first()


def evaluate(db: Session, user_id: int) -> TriggerDecision:
    """Apply the trigger policy for ``user_id``.

    Args:
        db: Open session.
        user_id: The user to evaluate.

    Returns:
        A :class:`TriggerDecision`.  ``should_run`` is False with reason
        ``"none"`` when no rule fires.
    """
    total_events = count_events(db, user_id)
    if total_events == 0:
        return TriggerDecision(
            should_run=False,
            reason=REASON_NONE,
            event_count=0,
            new_events=0,
            detail="no events recorded for this user yet",
        )

    latest = latest_recommendation(db, user_id)

    # Rule 1 — cold start.
    if latest is None:
        return TriggerDecision(
            should_run=True,
            reason=REASON_FIRST_TIME,
            event_count=total_events,
            new_events=total_events,
            detail=f"first recommendation for this user ({total_events} events available)",
        )

    created_at = _as_utc(latest.created_at) or datetime.now(timezone.utc)
    new_events = count_events(db, user_id, since=created_at)

    # Rule 2 — every Nth new event.
    if new_events >= settings.agent_event_trigger_interval:
        return TriggerDecision(
            should_run=True,
            reason=REASON_EVENT_THRESHOLD,
            event_count=total_events,
            new_events=new_events,
            detail=(
                f"{new_events} new events since recommendation #{latest.id} "
                f"(threshold {settings.agent_event_trigger_interval})"
            ),
        )

    # Rule 3 — stale recommendation with meaningful new activity.
    age = datetime.now(timezone.utc) - created_at
    if age > timedelta(hours=settings.agent_stale_hours) and (
        new_events >= settings.agent_stale_min_events
    ):
        return TriggerDecision(
            should_run=True,
            reason=REASON_STALE,
            event_count=total_events,
            new_events=new_events,
            detail=(
                f"recommendation #{latest.id} is {age.total_seconds() / 3600:.1f}h old "
                f"with {new_events} new events"
            ),
        )

    return TriggerDecision(
        should_run=False,
        reason=REASON_NONE,
        event_count=total_events,
        new_events=new_events,
        detail=(
            f"{new_events} new events since recommendation #{latest.id}; "
            f"needs {settings.agent_event_trigger_interval} "
            f"(or {settings.agent_stale_min_events} after "
            f"{settings.agent_stale_hours}h)"
        ),
    )
