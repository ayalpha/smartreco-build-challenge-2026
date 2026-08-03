"""Behavioural event ingestion — the fire-and-forget hot path.

Design constraints
------------------
* **Never block the UI.** The endpoint validates, bulk-inserts and returns.  The
  agent itself is dispatched to a background worker (see
  :func:`app.agent.runner.maybe_dispatch`), never run inline.
* **Accept ``sendBeacon``.** On page unload the browser sends the batch with a
  ``text/plain`` content type, which a normal Pydantic body parameter would
  reject with a 422.  The body is therefore read raw and validated manually, so
  both ``fetch`` and ``navigator.sendBeacon`` work.
* **Partial success beats failure.** One malformed event in a batch of fifty must
  not lose the other forty-nine, so events are validated individually and the
  rejected count is reported back.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.runner import maybe_dispatch
from app.database import get_db
from app.dependencies import get_current_user_optional
from app.models.event import Event
from app.models.product import Product
from app.models.user import User
from app.schemas.event import EventBatchIn, EventBatchResponse, EventIn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


async def _read_batch(request: Request) -> EventBatchIn:
    """Parse the request body into an :class:`EventBatchIn`.

    Accepts any content type so ``navigator.sendBeacon`` (``text/plain``) works
    alongside ``fetch`` (``application/json``).  A malformed body yields an empty
    batch rather than an error — losing analytics is always preferable to
    surfacing an error in the user's console on page unload.
    """
    raw = await request.body()
    if not raw:
        return EventBatchIn()

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("Discarding an event batch with an unparseable body")
        return EventBatchIn()

    if not isinstance(payload, dict):
        return EventBatchIn()

    # Validate the envelope leniently, then each event individually.
    session_id = str(payload.get("session_id") or "anonymous")[:64]
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raw_events = []

    batch = EventBatchIn(session_id=session_id, events=[])
    valid: list[EventIn] = []
    rejected = 0
    for item in raw_events[:200]:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            valid.append(EventIn.model_validate(item))
        except ValidationError as exc:
            rejected += 1
            logger.debug("Rejected an event: %s", exc.errors()[:1])

    batch.events = valid
    # Stash the reject count on the model instance for the caller.
    setattr(batch, "_rejected", rejected)
    return batch


def _known_product_ids(db: Session, product_ids: set[int]) -> set[int]:
    """Return the subset of ``product_ids`` that exist in the catalog.

    Unknown ids are nulled out before insert so a stale client cannot trip a
    foreign-key violation and lose the whole batch.
    """
    if not product_ids:
        return set()
    rows = db.execute(select(Product.id).where(Product.id.in_(product_ids))).all()
    return {int(row[0]) for row in rows}


@router.post("", response_model=EventBatchResponse, status_code=status.HTTP_200_OK)
async def ingest_events(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> EventBatchResponse:
    """Accept a batch of behavioural events.

    Anonymous batches are stored (keyed by ``session_id``) but never trigger the
    agent, since there is no user to recommend to.

    Returns:
        Counts plus whether a background agent run was dispatched.
    """
    batch = await _read_batch(request)
    rejected = int(getattr(batch, "_rejected", 0) or 0)

    if not batch.events:
        return EventBatchResponse(accepted=0, rejected=rejected)

    candidate_ids = {event.product_id for event in batch.events if event.product_id}
    known_ids = _known_product_ids(db, candidate_ids)

    rows: list[Event] = []
    for event in batch.events:
        product_id = event.product_id if event.product_id in known_ids else None
        row = Event(
            user_id=user.id if user else None,
            session_id=batch.session_id,
            event_type=event.event_type,
            product_id=product_id,
            path=(event.path or "")[:500] or None,
            metadata_json=event.collected_metadata(),
        )
        if event.timestamp is not None:
            row.timestamp = event.timestamp
        rows.append(row)

    try:
        db.add_all(rows)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist %d event(s)", len(rows))
        # Ingest failures must not surface to the user's console.
        response.status_code = status.HTTP_202_ACCEPTED
        return EventBatchResponse(accepted=0, rejected=rejected + len(rows))

    logger.debug(
        "Ingested %d event(s) for user=%s session=%s",
        len(rows), user.id if user else None, batch.session_id,
    )

    if user is None:
        return EventBatchResponse(accepted=len(rows), rejected=rejected)

    # Cheap indexed COUNT queries; the agent run itself goes to a worker thread.
    decision = maybe_dispatch(user.id)
    return EventBatchResponse(
        accepted=len(rows),
        rejected=rejected,
        triggered=decision.should_run,
        trigger_reason=decision.reason if decision.should_run else None,
    )


@router.get("/summary", response_model=dict)
def event_summary(
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> dict[str, Any]:
    """Return a small activity summary for the signed-in user.

    Powers the "your activity" strip on the profile page.  Anonymous callers get
    zeroed counters rather than a 401, because this is decorative.
    """
    if user is None:
        return {"authenticated": False, "total": 0, "by_type": {}}

    rows = db.execute(
        select(Event.event_type, Event.id).where(Event.user_id == user.id)
    ).all()
    by_type: dict[str, int] = {}
    for row in rows:
        key = str(row[0])
        by_type[key] = by_type.get(key, 0) + 1

    return {"authenticated": True, "total": len(rows), "by_type": by_type}
