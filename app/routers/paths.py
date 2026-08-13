"""Personalized learning paths — goal-driven curricula from catalog + behaviour.

Learners state what they want to become (e.g. "AI platform engineer"). We fuse
that goal with interest signals from their latest recommendation and recent
events, run hybrid retrieval over the live catalog, then either:

* ask Mesh to sequence the candidates into ordered steps with milestones, or
* fall back to a deterministic skill-level sort when Mesh is unavailable.

No prices, urgency claims, or courses outside the retrieved catalog are invented.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.mesh_client import MeshUnavailableError, call_llm_json, mesh_available
from app.database import get_db
from app.dependencies import get_current_user, render_page
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.schemas.path import PathOut, PathRequest
from app.vector_store.sync import hybrid_retrieve

logger = logging.getLogger(__name__)

router = APIRouter(tags=["paths"])
api_router = APIRouter(prefix="/api/path", tags=["paths"])

_SKILL_ORDER = {"beginner": 0, "intermediate": 1, "advanced": 2}
_MAX_STEPS = 6
_WEEKLY_HOUR_CHOICES = (3, 5, 8, 10, 15, 20)


def _duration_hours(value: Optional[str]) -> float:
    """Parse a rough hour count from a free-text duration like ``14 hours``."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value or "")
    return float(match.group(1)) if match else 0.0


def _weeks_for(duration: Optional[str], weekly_hours: int) -> int:
    """Estimate whole weeks needed given a weekly time budget."""
    hours = _duration_hours(duration)
    if hours <= 0:
        return 1
    return max(1, int(round(hours / max(1, weekly_hours))))


def behaviour_context(db: Session, user_id: int) -> tuple[list[str], str]:
    """Derive interest topics and recent search text for a user.

    Prefers the latest active recommendation's interest signals (already
    produced by the agent graph). Falls back to categories of products the user
    recently interacted with, plus any search queries still in the event log.
    """
    recommendation = db.scalar(
        select(Recommendation)
        .where(
            Recommendation.user_id == user_id,
            Recommendation.is_active.is_(True),
        )
        .order_by(Recommendation.created_at.desc())
    )

    signals: list[str] = []
    if recommendation and recommendation.interest_signals:
        for item in recommendation.interest_signals:
            if isinstance(item, dict) and item.get("topic"):
                signals.append(str(item["topic"]).strip())

    events = list(
        db.scalars(
            select(Event)
            .where(Event.user_id == user_id)
            .order_by(Event.timestamp.desc())
            .limit(40)
        )
    )

    searches = [
        str((event.metadata_json or {}).get("query", "")).strip()
        for event in events
        if event.event_type == EventType.SEARCH_QUERY.value
    ]

    product_ids = [event.product_id for event in events if event.product_id]
    if product_ids:
        categories = Counter(
            db.scalars(select(Product.category).where(Product.id.in_(product_ids))).all()
        )
        signals.extend(name for name, _ in categories.most_common(3) if name)

    # Preserve order, drop empties/dupes.
    unique = list(dict.fromkeys(s for s in signals if s))[:6]
    search_blob = "; ".join(q for q in searches if q)[:500]
    return unique, search_blob


def _candidate_products(db: Session, query: str) -> list[dict[str, Any]]:
    """Hybrid-retrieve catalog candidates; fall back to top-rated courses."""
    hits = hybrid_retrieve(db, query, limit=10)
    products: list[dict[str, Any]] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        payload["id"] = hit.product_id
        payload["score"] = hit.fused_score
        if payload.get("title"):
            products.append(payload)

    if products:
        return products

    rows = list(
        db.scalars(
            select(Product)
            .where(Product.is_active.is_(True))
            .order_by(Product.rating.desc().nullslast(), Product.id.desc())
            .limit(8)
        )
    )
    return [row.to_public_dict() | {"score": 0.0} for row in rows]


def fallback_path(
    goal: str,
    products: list[dict[str, Any]],
    interests: list[str],
    weekly_hours: int,
) -> dict[str, Any]:
    """Deterministic curriculum when Mesh is down or returns unusable JSON."""
    ordered = sorted(
        products,
        key=lambda p: (
            _SKILL_ORDER.get(str(p.get("skill_level", "")).lower(), 1),
            -float(p.get("score") or 0.0),
        ),
    )

    steps: list[dict[str, Any]] = []
    for index, product in enumerate(ordered[:_MAX_STEPS], start=1):
        title = str(product.get("title") or f"Course #{product.get('id')}")
        category = product.get("category") or "core"
        steps.append(
            {
                "order": index,
                "title": title,
                "product_id": int(product["id"]),
                "level": product.get("skill_level"),
                "duration": product.get("duration"),
                "weeks": _weeks_for(product.get("duration"), weekly_hours),
                "why": (
                    f"Builds the {category} capability that supports your goal: {goal}."
                ),
                "milestone": (
                    f"Ship a small portfolio piece that applies the core skills from {title}."
                ),
            }
        )

    interest_clause = (
        f" and your recent focus on {', '.join(interests[:3])}" if interests else ""
    )
    return {
        "headline": f"Your path toward {goal}",
        "summary": (
            f"A catalog-grounded progression shaped by your goal{interest_clause}. "
            "Steps are ordered beginner → advanced where skill levels are known."
        ),
        "steps": steps,
        "degraded": True,
        "interests": interests,
    }


def _mesh_path(
    goal: str,
    products: list[dict[str, Any]],
    interests: list[str],
    weekly_hours: int,
) -> Optional[dict[str, Any]]:
    """Ask Mesh to sequence only the supplied products into a path."""
    compact = [
        {
            "id": p["id"],
            "title": p.get("title"),
            "category": p.get("category"),
            "skill_level": p.get("skill_level"),
            "duration": p.get("duration"),
            "score": round(float(p.get("score") or 0.0), 4),
        }
        for p in products
    ]

    result = call_llm_json(
        [
            {
                "role": "system",
                "content": (
                    "You design realistic career learning paths for a course marketplace. "
                    "Return JSON only with keys: headline (string), summary (string), "
                    "steps (array). Each step must include product_id (int from the "
                    "candidates), why (one sentence), milestone (one concrete practice). "
                    "Use ONLY the supplied products. Order prerequisites and beginner "
                    "material before advanced work. Do not invent prices, discounts, "
                    "urgency, or courses that are not in the candidate list. Cap at "
                    f"{_MAX_STEPS} steps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n"
                    f"Weekly hours available: {weekly_hours}\n"
                    f"Learned interests from behaviour: {interests or 'none yet'}\n"
                    f"Catalog candidates (JSON): {compact}"
                ),
            },
        ],
        purpose="path_builder",
        max_tokens=1400,
    )

    if not isinstance(result, dict):
        return None

    by_id = {int(p["id"]): p for p in products}
    steps: list[dict[str, Any]] = []
    seen: set[int] = set()

    for item in result.get("steps") or []:
        if not isinstance(item, dict):
            continue
        try:
            product_id = int(item.get("product_id", 0))
        except (TypeError, ValueError):
            continue
        if product_id not in by_id or product_id in seen:
            continue
        seen.add(product_id)
        product = by_id[product_id]
        steps.append(
            {
                "order": len(steps) + 1,
                "title": product.get("title"),
                "product_id": product_id,
                "level": product.get("skill_level"),
                "duration": product.get("duration"),
                "weeks": _weeks_for(product.get("duration"), weekly_hours),
                "why": str(item.get("why") or "").strip()
                or f"Supports your goal: {goal}.",
                "milestone": str(item.get("milestone") or "").strip()
                or f"Apply the key skills from {product.get('title')}.",
            }
        )
        if len(steps) >= _MAX_STEPS:
            break

    if not steps:
        return None

    headline = str(result.get("headline") or "").strip() or f"Your path toward {goal}"
    summary = str(result.get("summary") or "").strip() or (
        "A Mesh-sequenced path using only courses already in the Nexora catalog."
    )
    return {
        "headline": headline[:240],
        "summary": summary[:800],
        "steps": steps,
        "degraded": False,
        "interests": interests,
    }


def _persist_career_goal(db: Session, user: User, goal: str) -> None:
    """Store the learner's stated goal so later agent runs can use it."""
    cleaned = (goal or "").strip()[:160]
    if not cleaned:
        return
    # Re-load on this session so we don't detach issues across request/agent code.
    row = db.get(User, user.id)
    if row is None:
        return
    if row.career_goal != cleaned:
        row.career_goal = cleaned
        db.add(row)
        db.commit()
        db.refresh(row)
        # Keep the request-scoped user object in sync for templates.
        user.career_goal = cleaned
        logger.info("Saved career_goal for user %s", user.id)


def build_path(
    db: Session,
    user: User,
    goal: str,
    weekly_hours: int = 5,
    *,
    persist_goal: bool = True,
) -> dict[str, Any]:
    """Build a personalized path for ``user`` aiming at ``goal``.

    Always returns a dict with ``headline``, ``summary``, ``steps``,
    ``degraded``, and ``interests`` — never raises for Mesh/Qdrant outages.

    When ``persist_goal`` is True (default), the cleaned goal is written to
    ``User.career_goal`` so the recommendation graph can bias retrieval.
    """
    cleaned_goal = (goal or "").strip()
    if not cleaned_goal:
        cleaned_goal = (user.career_goal or "").strip() or "your next role"

    if persist_goal and cleaned_goal != "your next role":
        _persist_career_goal(db, user, cleaned_goal)

    hours = max(1, min(40, int(weekly_hours or 5)))
    interests, searches = behaviour_context(db, user.id)
    query = " ".join(part for part in (cleaned_goal, *interests, searches) if part).strip()
    products = _candidate_products(db, query)
    fallback = fallback_path(cleaned_goal, products, interests, hours)

    if not products:
        return fallback

    if not mesh_available():
        logger.info("Path builder using heuristic fallback (Mesh unavailable)")
        return fallback

    try:
        mesh_result = _mesh_path(cleaned_goal, products, interests, hours)
        if mesh_result:
            return mesh_result
    except (MeshUnavailableError, ValueError, TypeError) as exc:
        logger.warning("Path builder Mesh call failed: %s", exc)

    return fallback


# --------------------------------------------------------------------------- #
# HTML pages                                                                  #
# --------------------------------------------------------------------------- #

@router.get("/path", include_in_schema=False)
def path_page(
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """Render the path builder form, pre-filled with any saved career goal."""
    return render_page(
        request,
        "path.html",
        user,
        path=None,
        goal=user.career_goal or "",
        weekly_hours=5,
        weekly_hour_choices=_WEEKLY_HOUR_CHOICES,
    )


@router.post("/path", include_in_schema=False)
def create_path(
    request: Request,
    goal: str = Form(min_length=3, max_length=160),
    weekly_hours: int = Form(default=5, ge=1, le=40),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    """Build and render a personalized path from the submitted goal."""
    cleaned = goal.strip()
    path = build_path(db, user, cleaned, weekly_hours)
    return render_page(
        request,
        "path.html",
        user,
        path=path,
        goal=cleaned,
        weekly_hours=weekly_hours,
        weekly_hour_choices=_WEEKLY_HOUR_CHOICES,
    )


# --------------------------------------------------------------------------- #
# JSON API                                                                    #
# --------------------------------------------------------------------------- #

@api_router.get("", response_model=dict[str, Any])
def get_saved_goal(user: User = Depends(get_current_user)) -> dict[str, Any]:
    """Return the learner's currently saved career goal (if any)."""
    return {
        "goal": user.career_goal or "",
        "has_goal": bool(user.career_goal),
    }


@api_router.post("", response_model=PathOut)
def create_path_api(
    body: PathRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PathOut:
    """Build a personalized path and return it as JSON."""
    cleaned = body.goal.strip()
    path = build_path(db, user, cleaned, body.weekly_hours)
    return PathOut.from_builder(path, goal=cleaned, weekly_hours=body.weekly_hours)
