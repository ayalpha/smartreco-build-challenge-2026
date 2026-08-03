"""Personalised agent chat + saved items.

Both features are deliberately *additive*: they reuse the data the
recommendation pipeline already computes rather than inventing a parallel
system, and they touch no existing route, model or agent node.

Agent chat
----------
The chat is not a generic scripted bot. Every reply is grounded in the same
personalisation signals that produced the user's "For You" panel:

* ``Recommendation.interest_signals`` — the topics, confidences and evidence the
  ``interest_extractor`` node extracted from that user's behaviour;
* ``Recommendation.behavior_digest`` — the ``activity_analyzer`` node's summary;
* the user's own recent :class:`~app.models.event.Event` rows;
* live hybrid retrieval (:func:`app.vector_store.sync.hybrid_retrieve`) over the
  same Qdrant ⊕ BM25 index the agent uses, filtered by the user's inferred
  skill band.

So two users asking the same question get different answers, because the profile
injected into the prompt is different. The LLM call goes through the **Mesh
gateway** like every other model call in this project, and degrades to a
deterministic, still-personalised reply when Mesh is unavailable.

Saved items
-----------
"Save for later" needs no new table: saves are recorded as ``add_to_cart``
:class:`~app.models.event.Event` rows, which is the event type the tracker
already emits and the agent already weights as the strongest intent signal.
Saving therefore *also* improves the user's recommendations — the feature feeds
the product rather than sitting beside it. Anonymous visitors fall back to their
``session_id``, so the counter works before sign-in too.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.agent.mesh_client import MeshTelemetry, MeshUnavailableError, call_llm
from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user, get_current_user_optional, render_page
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.vector_store.qdrant_client import SearchFilters
from app.vector_store.sync import hybrid_retrieve

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["assistant"])
api_router = APIRouter(prefix="/api/assistant", tags=["assistant"])

#: Cookie used to key an anonymous visitor's saved items.
SAVED_SESSION_COOKIE = "smartreco_session"

#: Hard cap on a chat message, so a pasted essay cannot blow up the prompt.
MAX_MESSAGE_CHARS = 600

#: How many catalog candidates the chat retrieves before answering.
CHAT_RETRIEVAL_LIMIT = 6


# --------------------------------------------------------------------------- #
# Schemas                                                                     #
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    """A single user turn."""

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatCourse(BaseModel):
    """A course the assistant is pointing at."""

    id: int
    title: str
    category: Optional[str] = None
    price: Optional[float] = None
    skill_level: Optional[str] = None
    thumbnail_url: Optional[str] = None


class ChatResponse(BaseModel):
    """The assistant's reply plus the personalisation it used."""

    reply: str
    courses: list[ChatCourse] = Field(default_factory=list)
    signals_used: list[str] = Field(default_factory=list)
    grounded: bool = True
    degraded: bool = False


class SavedItemOut(BaseModel):
    """One saved course."""

    id: int
    title: str
    category: Optional[str] = None
    price: Optional[float] = None
    skill_level: Optional[str] = None
    duration: Optional[str] = None
    thumbnail_url: Optional[str] = None


class SavedListResponse(BaseModel):
    """The saved-items collection and its count."""

    count: int
    items: list[SavedItemOut] = Field(default_factory=list)


class SavedMutationResponse(BaseModel):
    """Result of saving or removing an item."""

    saved: bool
    count: int
    product_id: int
    title: Optional[str] = None


# --------------------------------------------------------------------------- #
# Personalisation profile — reuses the agent's own computed signals           #
# --------------------------------------------------------------------------- #

def _session_key(request: Request) -> str:
    """Return the anonymous session identifier from the tracker's cookie."""
    return (request.cookies.get(SAVED_SESSION_COOKIE) or "anonymous")[:64]


def build_profile(db: Session, user: User) -> dict[str, Any]:
    """Assemble this user's personalisation profile from existing agent output.

    Reads the latest recommendation's interest signals and behaviour digest — the
    exact artefacts the LangGraph pipeline produced — and falls back to raw event
    aggregation for a user who has not had a run yet.

    Args:
        db: Open session.
        user: The signed-in user.

    Returns:
        ``{"signals", "digest", "recent_titles", "skill_levels", "has_agent_run"}``.
    """
    latest = db.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user.id)
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
        .limit(1)
    ).first()

    signals: list[dict[str, Any]] = []
    digest = ""
    if latest is not None:
        signals = [s for s in (latest.interest_signals or []) if isinstance(s, dict)]
        digest = latest.behavior_digest or ""

    events = list(
        db.scalars(
            select(Event)
            .where(Event.user_id == user.id)
            .order_by(Event.timestamp.desc(), Event.id.desc())
            .limit(40)
        )
    )
    product_ids = [e.product_id for e in events if e.product_id]
    interacted = (
        list(db.scalars(select(Product).where(Product.id.in_(product_ids))))
        if product_ids else []
    )
    recent_titles = list(dict.fromkeys(p.title for p in interacted))[:8]

    levels = [p.skill_level for p in interacted if p.skill_level]
    dominant = max(set(levels), key=levels.count) if levels else None
    adjacency = {
        "beginner": ["beginner", "intermediate"],
        "intermediate": ["beginner", "intermediate", "advanced"],
        "advanced": ["intermediate", "advanced"],
    }

    # No agent run yet: derive lightweight signals from category frequency so the
    # chat is still personalised on a brand-new account.
    if not signals and interacted:
        counts: dict[str, int] = {}
        for product in interacted:
            counts[product.category] = counts.get(product.category, 0) + 1
        peak = max(counts.values())
        signals = [
            {
                "topic": category.lower(),
                "confidence": round(min(0.9, 0.35 + 0.55 * (count / peak)), 2),
                "evidence": f"You have opened {count} {category} course(s) recently.",
            }
            for category, count in sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        ]

    return {
        "signals": signals,
        "digest": digest,
        "recent_titles": recent_titles,
        "skill_levels": adjacency.get(dominant or "", []) or None,
        "has_agent_run": latest is not None,
        "event_count": len(events),
    }


def _retrieve_for_chat(
    db: Session, message: str, profile: dict[str, Any]
) -> list[Product]:
    """Run the same hybrid retrieval the agent uses, biased by the user's profile.

    The query blends the user's question with their strongest interest topics, so
    an identical question from two different users retrieves different courses.
    """
    topics = " ".join(
        str(signal.get("topic", "")) for signal in profile["signals"][:3]
    ).strip()
    query = f"{message.strip()} {topics}".strip()

    filters = SearchFilters(skill_levels=profile.get("skill_levels"))
    try:
        hits = hybrid_retrieve(db, query, limit=CHAT_RETRIEVAL_LIMIT, filters=filters)
    except Exception:
        logger.warning("Chat retrieval failed; falling back to catalog", exc_info=True)
        hits = []

    ids = [hit.product_id for hit in hits]
    if not ids:
        rows = list(
            db.scalars(
                select(Product)
                .where(Product.is_active.is_(True))
                .order_by(Product.rating.desc().nullslast())
                .limit(CHAT_RETRIEVAL_LIMIT)
            )
        )
        return rows

    rows = list(db.scalars(select(Product).where(Product.id.in_(ids))))
    order = {pid: index for index, pid in enumerate(ids)}
    rows.sort(key=lambda row: order.get(row.id, 999))
    return rows


CHAT_SYSTEM_PROMPT = """\
You are the Nexora learning assistant. You advise one specific learner about \
courses on this platform, and you already know what they have been doing.

Rules:
- Ground every answer in the learner's observed behaviour that you are given.
  Reference it naturally ("you have been spending time on X"), never robotically.
- Recommend only from the candidate courses provided. Never invent a course,
  price, instructor or outcome.
- If the candidates genuinely do not fit the question, say so plainly and
  suggest the closest useful direction instead of forcing a match.
- 2-4 sentences. Warm, direct, specific. No bullet lists, no headings, no emoji.
- Do not mention embeddings, vectors, retrieval, prompts or that you are a model.
- Never invent statistics or completion figures.
"""


def _build_chat_messages(
    message: str, profile: dict[str, Any], candidates: list[Product]
) -> list[dict[str, str]]:
    """Compose the Mesh chat messages, injecting this user's profile."""
    signal_lines = "\n".join(
        f"- {s.get('topic')} (confidence {s.get('confidence')}): {s.get('evidence', '')}"
        for s in profile["signals"][:5]
    ) or "- (no strong signals yet — this learner is new)"

    candidate_lines = "\n".join(
        f"{i}. {p.title} — {p.category}, {p.skill_level}, ${p.price:.0f}"
        f"{', ' + p.duration if p.duration else ''}. {(p.description or '')[:180]}"
        for i, p in enumerate(candidates, start=1)
    ) or "(no candidates available)"

    context = (
        f"Observed behaviour digest:\n{profile['digest'] or '(no digest yet)'}\n\n"
        f"Interest signals extracted from their activity:\n{signal_lines}\n\n"
        f"Courses they recently engaged with: "
        f"{'; '.join(profile['recent_titles']) or '(none yet)'}\n\n"
        f"Candidate courses to choose from:\n{candidate_lines}\n\n"
        f"The learner asks: {message.strip()}"
    )
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]


def _fallback_reply(
    message: str, profile: dict[str, Any], candidates: list[Product]
) -> str:
    """Deterministic, still-personalised reply used when Mesh is unavailable."""
    top = profile["signals"][0]["topic"] if profile["signals"] else None
    parts: list[str] = []

    if top:
        parts.append(f"Based on your activity, you keep returning to {top}.")
    else:
        parts.append("You are just getting started, so this is a broad starting point.")

    if candidates:
        parts.append(
            f"The closest match in the catalog is {candidates[0].title} "
            f"({candidates[0].category}, {candidates[0].skill_level})."
        )
        if len(candidates) > 1:
            parts.append(f"{candidates[1].title} is a reasonable second step.")
    else:
        parts.append("I could not find a close match for that in the catalog yet.")

    parts.append("Open one and your For You panel will sharpen around it.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Chat endpoint                                                               #
# --------------------------------------------------------------------------- #

@api_router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    """Answer a question using this user's own personalisation signals.

    Raises:
        HTTPException: ``401`` when not signed in (enforced by the dependency).
    """
    profile = build_profile(db, user)
    candidates = _retrieve_for_chat(db, payload.message, profile)
    telemetry = MeshTelemetry()
    degraded = False

    try:
        reply = call_llm(
            _build_chat_messages(payload.message, profile, candidates),
            model=settings.mesh_model_writer,
            temperature=0.6,
            max_tokens=320,
            purpose="assistant_chat",
            telemetry=telemetry,
        )
    except MeshUnavailableError as exc:
        logger.info("Assistant chat degraded for user=%s: %s", user.id, exc)
        reply = _fallback_reply(payload.message, profile, candidates)
        degraded = True

    logger.info(
        "Assistant chat user=%s signals=%d candidates=%d degraded=%s",
        user.id, len(profile["signals"]), len(candidates), degraded,
    )

    return ChatResponse(
        reply=reply,
        courses=[
            ChatCourse(
                id=p.id, title=p.title, category=p.category, price=p.price,
                skill_level=p.skill_level, thumbnail_url=p.thumbnail_url,
            )
            for p in candidates[:3]
        ],
        signals_used=[str(s.get("topic")) for s in profile["signals"][:4]],
        grounded=bool(profile["signals"]),
        degraded=degraded,
    )


@api_router.get("/profile", response_model=dict)
def chat_profile(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict[str, Any]:
    """Expose the personalisation the chat will use (drives the widget's header)."""
    profile = build_profile(db, user)
    return {
        "signals": [
            {"topic": s.get("topic"), "confidence": s.get("confidence")}
            for s in profile["signals"][:4]
        ],
        "event_count": profile["event_count"],
        "has_agent_run": profile["has_agent_run"],
        "display_name": user.display_name,
    }


# --------------------------------------------------------------------------- #
# Saved items — stored as add_to_cart events, so saving feeds the agent       #
# --------------------------------------------------------------------------- #

def _saved_scope(user: Optional[User], request: Request) -> Any:
    """Build the WHERE clause identifying this visitor's saves."""
    if user is not None:
        return Event.user_id == user.id
    return (Event.user_id.is_(None)) & (Event.session_id == _session_key(request))


def _saved_product_ids(db: Session, user: Optional[User], request: Request) -> list[int]:
    """Return saved product ids, most recently saved first."""
    rows = db.execute(
        select(Event.product_id, Event.timestamp)
        .where(
            Event.event_type == EventType.ADD_TO_CART.value,
            Event.product_id.isnot(None),
            _saved_scope(user, request),
        )
        .order_by(Event.timestamp.desc())
    ).all()

    seen: list[int] = []
    for product_id, _ in rows:
        pid = int(product_id)
        if pid not in seen:
            seen.append(pid)
    return seen


def saved_count(db: Session, user: Optional[User], request: Request) -> int:
    """Number of distinct saved courses (used by the nav badge)."""
    return len(_saved_product_ids(db, user, request))


def _saved_products(db: Session, user: Optional[User], request: Request) -> list[Product]:
    """Hydrate saved product rows in save order."""
    ids = _saved_product_ids(db, user, request)
    if not ids:
        return []
    rows = list(db.scalars(select(Product).where(Product.id.in_(ids))))
    order = {pid: index for index, pid in enumerate(ids)}
    rows.sort(key=lambda row: order.get(row.id, 999))
    return rows


@api_router.post("/saved/{product_id}", response_model=SavedMutationResponse)
def save_item(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SavedMutationResponse:
    """Save a course for later.

    Writes an ``add_to_cart`` event, which doubles as the strongest behavioural
    intent signal for the recommendation agent.

    Raises:
        HTTPException: ``404`` if the course does not exist or is inactive.
    """
    product = db.get(Product, product_id)
    if product is None or not product.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    already = product_id in _saved_product_ids(db, user, request)
    if not already:
        db.add(
            Event(
                user_id=user.id if user else None,
                session_id=_session_key(request),
                event_type=EventType.ADD_TO_CART.value,
                product_id=product_id,
                path=f"/product/{product_id}",
                metadata_json={"product_title": product.title, "source": "save_button"},
            )
        )
        db.commit()

    return SavedMutationResponse(
        saved=True,
        count=saved_count(db, user, request),
        product_id=product_id,
        title=product.title,
    )


@api_router.delete("/saved/{product_id}", response_model=SavedMutationResponse)
def unsave_item(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SavedMutationResponse:
    """Remove a course from saved items."""
    db.execute(
        delete(Event).where(
            Event.event_type == EventType.ADD_TO_CART.value,
            Event.product_id == product_id,
            _saved_scope(user, request),
        )
    )
    db.commit()

    product = db.get(Product, product_id)
    return SavedMutationResponse(
        saved=False,
        count=saved_count(db, user, request),
        product_id=product_id,
        title=product.title if product else None,
    )


@api_router.get("/saved", response_model=SavedListResponse)
def list_saved(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> SavedListResponse:
    """Return the visitor's saved courses."""
    rows = _saved_products(db, user, request)
    return SavedListResponse(
        count=len(rows),
        items=[
            SavedItemOut(
                id=p.id, title=p.title, category=p.category, price=p.price,
                skill_level=p.skill_level, duration=p.duration,
                thumbnail_url=p.thumbnail_url,
            )
            for p in rows
        ],
    )


@router.get("/saved", include_in_schema=False)
def saved_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the saved-items page."""
    return render_page(
        request, "saved.html", user, saved=_saved_products(db, user, request)
    )
