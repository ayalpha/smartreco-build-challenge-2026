"""LangGraph node implementations for the recommendation agent.

Each public function in this module is a graph node: it takes the shared
:class:`~app.agent.state.RecommendationState` and returns a *partial* update.

Two invariants hold for every node
----------------------------------
1. **It never raises.** The :func:`instrumented` decorator catches everything,
   records the failure in ``node_log``, flags the run ``degraded`` and lets the
   graph continue.  A recommendation engine that 500s is worse than one that
   returns a slightly weaker answer.
2. **Mesh is the only AI path, and it is optional-by-design.** Every node that
   calls an LLM has a deterministic heuristic fallback used when
   :class:`~app.agent.mesh_client.MeshUnavailableError` is raised.  That is what
   makes the graph runnable in CI with no ``MESH_API_KEY``.

Nodes open their own short-lived database sessions.  The graph therefore has no
session affinity and is safe to invoke from APScheduler worker threads.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.mesh_client import (
    MeshTelemetry,
    MeshUnavailableError,
    call_llm_json,
    mesh_available,
)
from app.agent.prompts import (
    ACTIVITY_ANALYZER_SYSTEM,
    INTEREST_EXTRACTOR_SYSTEM,
    PERSUASION_WRITER_SYSTEM,
    RELEVANCE_GRADER_SYSTEM,
    RETRIEVAL_REFINER_SYSTEM,
    activity_analyzer_user,
    interest_extractor_user,
    persuasion_writer_user,
    relevance_grader_user,
    retrieval_refiner_user,
)
from app.agent.state import NodeExecution, RecommendationState
from app.cache import cache_active_recommendation, clear_agent_pending, invalidate_recommendation_cache
from app.config import get_settings
from app.database import session_scope
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.vector_store.qdrant_client import SearchFilters
from app.vector_store.sync import hybrid_retrieve

logger = logging.getLogger(__name__)
settings = get_settings()

#: Node names — also used for the conditional-edge routing keys in graph.py.
NODE_ACTIVITY_ANALYZER = "activity_analyzer"
NODE_INTEREST_EXTRACTOR = "interest_extractor"
NODE_RETRIEVAL = "retrieval_node"
NODE_RELEVANCE_GRADER = "relevance_grader"
NODE_RETRIEVAL_REFINER = "retrieval_refiner"
NODE_PERSUASION_WRITER = "persuasion_writer"
NODE_RECOMMENDATION_STORER = "recommendation_storer"

#: Relevance at or above this counts as "relevant" in the heuristic fallback.
_HEURISTIC_RELEVANCE_THRESHOLD = 0.35

#: Weighting between the LLM judge and the retrieval engine when re-ranking.
_RERANK_JUDGE_WEIGHT = 0.65
_RERANK_RETRIEVAL_WEIGHT = 0.35


# --------------------------------------------------------------------------- #
# Instrumentation                                                             #
# --------------------------------------------------------------------------- #

NodeFn = Callable[[RecommendationState], dict[str, Any]]


def instrumented(name: str) -> Callable[[NodeFn], NodeFn]:
    """Decorate a node with timing, structured logging and error containment.

    The wrapped function may set two private keys on its return value to
    communicate with the decorator:

    * ``_degraded`` — True when a fallback path was taken;
    * ``_detail`` — short human-readable note stored in ``node_log``.

    Both are stripped before the update reaches LangGraph.
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(state: RecommendationState) -> dict[str, Any]:
            started = time.perf_counter()
            user_id = state.get("user_id")
            logger.debug("→ node %s (user=%s)", name, user_id)

            try:
                update: dict[str, Any] = dict(fn(state) or {})
                degraded = bool(update.pop("_degraded", False))
                detail = str(update.pop("_detail", ""))
            except Exception as exc:  # noqa: BLE001 - containment is the point
                logger.exception("Node %s failed for user=%s", name, user_id)
                update = {"error": f"{name}: {type(exc).__name__}: {exc}"}
                degraded = True
                detail = f"{type(exc).__name__}: {exc}"[:300]

            duration_ms = (time.perf_counter() - started) * 1000.0
            entry = NodeExecution(
                node=name, duration_ms=round(duration_ms, 2), degraded=degraded, detail=detail
            )
            update["node_log"] = list(state.get("node_log") or []) + [entry]
            if degraded:
                update["degraded"] = True

            logger.info(
                "← node %-22s user=%-5s %7.1fms%s%s",
                name, user_id, duration_ms,
                " [degraded]" if degraded else "",
                f" — {detail}" if detail else "",
            )
            return update

        return wrapper

    return decorator


def _telemetry(state: RecommendationState) -> MeshTelemetry:
    """Return the run's Mesh telemetry collector, creating one if absent."""
    collector = state.get("telemetry")
    if isinstance(collector, MeshTelemetry):
        return collector
    return MeshTelemetry()


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    """Coerce ``value`` to a float inside ``[low, high]``, falling back to ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


# --------------------------------------------------------------------------- #
# 1. activity_analyzer                                                        #
# --------------------------------------------------------------------------- #

def _load_recent_events(db: Session, user_id: int, limit: int) -> list[Event]:
    """Load a user's most recent events, returned oldest-first."""
    rows = list(
        db.scalars(
            select(Event)
            .where(Event.user_id == user_id)
            .order_by(Event.timestamp.desc(), Event.id.desc())
            .limit(limit)
        )
    )
    return list(reversed(rows))


def _product_titles(db: Session, product_ids: list[int]) -> dict[int, str]:
    """Map product ids to authoritative catalog titles."""
    if not product_ids:
        return {}
    rows = db.execute(
        select(Product.id, Product.title).where(Product.id.in_(product_ids))
    ).all()
    return {int(row[0]): str(row[1]) for row in rows}


def _heuristic_digest(
    events: list[Event], titles: dict[int, str], categories: Counter
) -> str:
    """Deterministic behaviour summary used when Mesh is unavailable.

    Deliberately factual rather than persuasive: it reports counts and the
    strongest signals so downstream nodes still receive usable grounding.
    """
    if not events:
        return (
            "This learner has no recorded activity yet, so there is no behavioural "
            "evidence to draw on. Treat them as a new visitor exploring the catalog."
        )

    type_counts = Counter(event.event_type for event in events)
    searches = [
        str((event.metadata_json or {}).get("query", "")).strip()
        for event in events
        if event.event_type == EventType.SEARCH_QUERY.value
    ]
    searches = [query for query in searches if query]

    dwell: Counter = Counter()
    for event in events:
        if event.event_type == EventType.TIME_SPENT.value and event.product_id:
            try:
                dwell[event.product_id] += int((event.metadata_json or {}).get("seconds", 0) or 0)
            except (TypeError, ValueError):
                continue

    cart_ids = [
        event.product_id
        for event in events
        if event.event_type == EventType.ADD_TO_CART.value and event.product_id
    ]

    parts: list[str] = [
        f"Across {len(events)} recent events this learner produced "
        f"{type_counts.get(EventType.PRODUCT_CLICK.value, 0)} product clicks, "
        f"{type_counts.get(EventType.SEARCH_QUERY.value, 0)} searches and "
        f"{type_counts.get(EventType.PAGE_VIEW.value, 0)} page views."
    ]
    if categories:
        top = ", ".join(f"{name} ({count})" for name, count in categories.most_common(3))
        parts.append(f"Their attention concentrates on: {top}.")
    if searches:
        parts.append("They searched for: " + "; ".join(f"'{q}'" for q in searches[-5:]) + ".")
    if dwell:
        best = ", ".join(
            f"{titles.get(pid, f'product #{pid}')} ({seconds}s)"
            for pid, seconds in dwell.most_common(3)
        )
        parts.append(f"Longest dwell times: {best}.")
    if cart_ids:
        wanted = ", ".join(titles.get(pid, f"product #{pid}") for pid in cart_ids[-3:])
        parts.append(f"They added these to the cart, the strongest intent signal: {wanted}.")
    if not categories and not searches and not dwell:
        parts.append(
            "The signal is sparse — mostly shallow browsing without a clear topic yet."
        )

    return " ".join(parts)


@instrumented(NODE_ACTIVITY_ANALYZER)
def activity_analyzer(state: RecommendationState) -> dict[str, Any]:
    """Summarise the user's recent behaviour into a dense digest.

    Reads the last ``AGENT_RECENT_EVENT_WINDOW`` events, renders them as a
    chronological log and asks Mesh for a factual behavioural digest.  Falls back
    to :func:`_heuristic_digest` when Mesh is unavailable.
    """
    user_id = int(state["user_id"])
    telemetry = _telemetry(state)

    with session_scope() as db:
        events = _load_recent_events(db, user_id, settings.agent_recent_event_window)
        product_ids = [event.product_id for event in events if event.product_id]
        titles = _product_titles(db, product_ids)
        raw_events = [event.to_dict() for event in events]
        event_lines = [
            event.describe(product_title=titles.get(event.product_id or -1))
            for event in events
        ]
        categories = Counter(
            row[0]
            for row in db.execute(
                select(Product.category).where(Product.id.in_(product_ids or [-1]))
            ).all()
        )
        seen_ids = sorted({int(pid) for pid in product_ids})
        cart_ids = sorted(
            {
                int(event.product_id)
                for event in events
                if event.event_type == EventType.ADD_TO_CART.value and event.product_id
            }
        )
        heuristic = _heuristic_digest(events, titles, categories)

    update: dict[str, Any] = {
        "raw_events": raw_events,
        "seen_product_ids": seen_ids,
        "telemetry": telemetry,
        "retrieval_filters": {
            **(state.get("retrieval_filters") or {}),
            "exclude_product_ids": cart_ids,
        },
    }

    if not events:
        update.update(
            behavior_digest=heuristic,
            _degraded=False,
            _detail="no events — cold-start path",
        )
        return update

    if not mesh_available():
        update.update(
            behavior_digest=heuristic, _degraded=True, _detail="Mesh not configured"
        )
        return update

    try:
        payload = call_llm_json(
            messages=[
                {"role": "system", "content": ACTIVITY_ANALYZER_SYSTEM
                 + "\n\nReturn ONLY JSON: {\"digest\": \"...\"}"},
                {
                    "role": "user",
                    "content": activity_analyzer_user(
                        event_lines, str(state.get("trigger_reason") or "manual")
                    ),
                },
            ],
            model=settings.mesh_model_reasoning,
            temperature=0.3,
            purpose=NODE_ACTIVITY_ANALYZER,
            telemetry=telemetry,
        )
        digest = ""
        if isinstance(payload, dict):
            digest = str(payload.get("digest") or "").strip()
        elif isinstance(payload, str):
            digest = payload.strip()

        if not digest:
            raise MeshUnavailableError("empty digest")

        update.update(
            behavior_digest=digest,
            _detail=f"{len(events)} events summarised via Mesh",
        )
        return update

    except MeshUnavailableError as exc:
        update.update(
            behavior_digest=heuristic,
            _degraded=True,
            _detail=f"Mesh fallback: {exc}"[:200],
        )
        return update


# --------------------------------------------------------------------------- #
# 2. interest_extractor                                                       #
# --------------------------------------------------------------------------- #

def _heuristic_interests(
    raw_events: list[dict[str, Any]], titles: dict[int, str], catalog_rows: list[Product]
) -> tuple[list[dict[str, Any]], str]:
    """Derive interest signals and a query without an LLM.

    Weights each interacted product's category/tags by event type (cart 3.0,
    dwell scaled by seconds, click 1.0, view 0.3), then normalises the top topics
    into confidences.
    """
    weights: Counter = Counter()
    by_id = {row.id: row for row in catalog_rows}
    searches: list[str] = []

    for event in raw_events:
        event_type = event.get("event_type")
        meta = event.get("metadata") or {}
        if event_type == EventType.SEARCH_QUERY.value:
            query = str(meta.get("query", "")).strip()
            if query:
                searches.append(query)
                weights[query.lower()] += 1.5
            continue

        product = by_id.get(event.get("product_id") or -1)
        if product is None:
            continue

        if event_type == EventType.ADD_TO_CART.value:
            weight = 3.0
        elif event_type == EventType.TIME_SPENT.value:
            try:
                seconds = float(meta.get("seconds", 0) or 0)
            except (TypeError, ValueError):
                seconds = 0.0
            weight = min(3.0, seconds / 45.0)
        elif event_type in (
            EventType.PRODUCT_CLICK.value,
            EventType.RECOMMENDATION_CLICK.value,
        ):
            weight = 1.0
        else:
            weight = 0.3

        weights[product.category.lower()] += weight
        for tag in product.tag_list[:4]:
            weights[tag] += weight * 0.6

    if not weights:
        return (
            [
                {
                    "topic": "getting started",
                    "confidence": 0.3,
                    "evidence": "No interactions yet — showing broadly popular starting points.",
                }
            ],
            "a well-reviewed introductory course that helps a brand new learner "
            "start building practical, in-demand technical skills",
        )

    top = weights.most_common(5)
    peak = top[0][1] or 1.0
    signals = [
        {
            "topic": topic,
            "confidence": round(min(0.95, 0.35 + 0.6 * (score / peak)), 2),
            "evidence": f"Accumulated behavioural weight of {score:.1f} across recent events.",
        }
        for topic, score in top
    ]

    topics = ", ".join(topic for topic, _ in top[:3])
    search_hint = f" The learner explicitly searched for {searches[-1]!r}." if searches else ""
    query = (
        f"A practical, hands-on course that deepens skills in {topics}, suitable as the "
        f"natural next step for someone actively studying these topics.{search_hint}"
    )
    return signals, query


@instrumented(NODE_INTEREST_EXTRACTOR)
def interest_extractor(state: RecommendationState) -> dict[str, Any]:
    """Turn the behaviour digest into interest signals, a query and filters.

    Also infers the metadata filters used by BONUS 4 retrieval: candidate skill
    levels and a price ceiling, but only when behaviour actually supports them.
    """
    telemetry = _telemetry(state)
    digest = str(state.get("behavior_digest") or "")
    raw_events = list(state.get("raw_events") or [])
    seen_ids = [int(pid) for pid in (state.get("seen_product_ids") or [])]

    with session_scope() as db:
        categories = [
            str(row[0])
            for row in db.execute(select(Product.category).distinct()).all()
            if row[0]
        ]
        interacted = (
            list(db.scalars(select(Product).where(Product.id.in_(seen_ids)))) if seen_ids else []
        )
        titles = {row.id: row.title for row in interacted}
        recent_titles = [titles[pid] for pid in seen_ids[-8:] if pid in titles]
        observed_prices = [row.price for row in interacted if row.price is not None]
        observed_levels = Counter(row.skill_level for row in interacted if row.skill_level)

    heuristic_signals, heuristic_query = _heuristic_interests(raw_events, titles, interacted)

    existing_filters = dict(state.get("retrieval_filters") or {})
    inferred_filters: dict[str, Any] = {
        "exclude_product_ids": existing_filters.get("exclude_product_ids") or []
    }
    # Price sensitivity: allow a 60% headroom over the dearest thing they engaged
    # with, so we never pitch a $400 bootcamp to someone browsing $20 courses.
    if observed_prices:
        ceiling = max(observed_prices)
        if ceiling > 0:
            inferred_filters["max_price"] = round(ceiling * 1.6, 2)
    if observed_levels:
        dominant = observed_levels.most_common(1)[0][0]
        adjacency = {
            "beginner": ["beginner", "intermediate"],
            "intermediate": ["beginner", "intermediate", "advanced"],
            "advanced": ["intermediate", "advanced"],
        }
        inferred_filters["skill_levels"] = adjacency.get(dominant, [dominant])

    if not mesh_available() or not digest:
        return {
            "interest_signals": heuristic_signals,
            "retrieval_query": heuristic_query,
            "retrieval_filters": inferred_filters,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": "heuristic interest extraction",
        }

    try:
        payload = call_llm_json(
            messages=[
                {"role": "system", "content": INTEREST_EXTRACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": interest_extractor_user(digest, categories, recent_titles),
                },
            ],
            model=settings.mesh_model_reasoning,
            temperature=0.2,
            purpose=NODE_INTEREST_EXTRACTOR,
            telemetry=telemetry,
        )
        if not isinstance(payload, dict):
            raise MeshUnavailableError("interest extractor returned a non-object")

        signals: list[dict[str, Any]] = []
        for item in payload.get("interest_signals") or []:
            if not isinstance(item, dict) or not item.get("topic"):
                continue
            signals.append(
                {
                    "topic": str(item["topic"])[:80],
                    "confidence": _clamp(item.get("confidence"), default=0.5),
                    "evidence": str(item.get("evidence") or "")[:400],
                }
            )
        signals.sort(key=lambda item: item["confidence"], reverse=True)
        signals = signals[:5] or heuristic_signals

        query = str(payload.get("retrieval_query") or "").strip() or heuristic_query

        levels = [
            str(level).lower()
            for level in (payload.get("inferred_skill_levels") or [])
            if str(level).lower() in {"beginner", "intermediate", "advanced"}
        ]
        if levels:
            inferred_filters["skill_levels"] = sorted(set(levels))

        max_price = payload.get("inferred_max_price")
        if isinstance(max_price, (int, float)) and max_price > 0:
            inferred_filters["max_price"] = float(max_price)

        return {
            "interest_signals": signals,
            "retrieval_query": query,
            "retrieval_filters": inferred_filters,
            "telemetry": telemetry,
            "_detail": f"{len(signals)} signals; filters={list(inferred_filters)}",
        }

    except MeshUnavailableError as exc:
        return {
            "interest_signals": heuristic_signals,
            "retrieval_query": heuristic_query,
            "retrieval_filters": inferred_filters,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": f"Mesh fallback: {exc}"[:200],
        }


# --------------------------------------------------------------------------- #
# 3. retrieval_node                                                           #
# --------------------------------------------------------------------------- #

def _filters_from_state(state: RecommendationState) -> SearchFilters:
    """Build a :class:`SearchFilters` from the state's inferred constraints."""
    raw = state.get("retrieval_filters") or {}
    return SearchFilters(
        skill_levels=raw.get("skill_levels") or None,
        categories=raw.get("categories") or None,
        max_price=raw.get("max_price"),
        min_price=raw.get("min_price"),
        exclude_product_ids=raw.get("exclude_product_ids") or None,
    )


def _fallback_catalog(db: Session, limit: int, exclude: list[int]) -> list[dict[str, Any]]:
    """Highest-rated active products — the safety net when retrieval finds nothing."""
    query = select(Product).where(Product.is_active.is_(True))
    if exclude:
        query = query.where(Product.id.notin_(exclude))
    rows = list(
        db.scalars(query.order_by(Product.rating.desc().nullslast(), Product.id.desc()).limit(limit))
    )
    products: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        payload = row.to_public_dict()
        payload.update(
            fused_score=max(0.05, 0.5 - rank * 0.02),
            dense_score=0.0,
            keyword_score=0.0,
            retrieval_mode="catalog_fallback",
            dense_rank=None,
            keyword_rank=None,
        )
        products.append(payload)
    return products


@instrumented(NODE_RETRIEVAL)
def retrieval_node(state: RecommendationState) -> dict[str, Any]:
    """Hybrid semantic + keyword retrieval over the Qdrant catalog (BONUS 4).

    Dense vectors come from Mesh embeddings; the sparse half is Okapi BM25 over
    the SQL catalog; the two rankings are fused with Reciprocal Rank Fusion.
    Metadata filters (skill level, price band, exclusions) are applied to both
    halves.  If the filtered search returns nothing, the filters are dropped once
    before falling back to the top-rated catalog slice.
    """
    query = str(state.get("retrieval_query") or "").strip()
    if not query:
        query = "popular practical technology course for a motivated new learner"

    filters = _filters_from_state(state)
    top_k = settings.vector_search_top_k
    notes: list[str] = []

    with session_scope() as db:
        hits = hybrid_retrieve(db, query, limit=top_k, filters=filters)

        if not hits and not filters.is_empty():
            notes.append("filters dropped (0 hits)")
            relaxed = SearchFilters(exclude_product_ids=filters.exclude_product_ids)
            hits = hybrid_retrieve(db, query, limit=top_k, filters=relaxed)

        products = [hit.as_dict() for hit in hits]

        if not products:
            notes.append("catalog fallback")
            products = _fallback_catalog(
                db, top_k, [int(pid) for pid in (filters.exclude_product_ids or [])]
            )

    return {
        "retrieved_products": products,
        "_detail": (
            f"{len(products)} candidates for {query[:48]!r}"
            + (f" ({'; '.join(notes)})" if notes else "")
        ),
        "_degraded": bool(notes),
    }


# --------------------------------------------------------------------------- #
# 4. relevance_grader (grading + LLM-as-judge re-ranking)                     #
# --------------------------------------------------------------------------- #

def _score_overlap(candidate: dict[str, Any], topic_terms: set[str]) -> float:
    """Fraction of signal terms present in a candidate's searchable text."""
    if not topic_terms:
        return 0.0
    tags = candidate.get("tags") or []
    tag_text = " ".join(tags) if isinstance(tags, list) else str(tags)
    haystack = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("category", "")),
            tag_text,
            str(candidate.get("description", ""))[:300],
        ]
    ).lower()
    hits = sum(1 for term in topic_terms if term and term in haystack)
    return hits / max(1, len(topic_terms))


def _rerank(graded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank graded candidates by blending judge score with retrieval rank.

    The LLM judge is accurate but noisy; the retrieval engine's fused RRF score is
    stable but shallow.  Blending both (65/35) beats either alone — this is the
    re-ranking stage of BONUS 4.
    """
    if not graded:
        return []

    fused_scores = [float(item["product"].get("fused_score") or 0.0) for item in graded]
    peak = max(fused_scores) or 1.0

    for item, fused in zip(graded, fused_scores):
        judge = float(item.get("relevance_score") or 0.0)
        item["rerank_score"] = round(
            _RERANK_JUDGE_WEIGHT * judge + _RERANK_RETRIEVAL_WEIGHT * (fused / peak), 4
        )

    graded.sort(
        key=lambda item: (bool(item.get("is_relevant")), item.get("rerank_score", 0.0)),
        reverse=True,
    )
    return graded


@instrumented(NODE_RELEVANCE_GRADER)
def relevance_grader(state: RecommendationState) -> dict[str, Any]:
    """Judge each retrieved candidate, then re-rank the survivors.

    Uses the cheap/fast Mesh grader model (``MESH_MODEL_GRADER``) to score every
    candidate 0-1 with a one-line justification, then blends those scores with the
    retrieval ranking in :func:`_rerank`.  The conditional edge downstream reads
    the number of relevant survivors to decide whether to refine the query.
    """
    telemetry = _telemetry(state)
    candidates = list(state.get("retrieved_products") or [])
    signals = list(state.get("interest_signals") or [])
    digest = str(state.get("behavior_digest") or "")

    if not candidates:
        return {
            "graded_products": [],
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": "nothing to grade",
        }

    topic_terms: set[str] = set()
    for signal in signals:
        topic_terms.update(
            term for term in str(signal.get("topic", "")).lower().split() if len(term) > 2
        )

    def heuristic() -> list[dict[str, Any]]:
        """Lexical-overlap grading, blended with retrieval score."""
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            overlap = _score_overlap(candidate, topic_terms)
            fused = float(candidate.get("fused_score") or 0.0)
            score = _clamp(0.55 * overlap + 0.45 * min(1.0, fused * 25.0), default=0.4)
            out.append(
                {
                    "product": candidate,
                    "relevance_score": round(score, 3),
                    "is_relevant": score >= _HEURISTIC_RELEVANCE_THRESHOLD,
                    "reason": (
                        f"Lexical overlap {overlap:.0%} with the learner's top topics "
                        f"and rank {candidate.get('dense_rank') or candidate.get('keyword_rank') or '-'} "
                        "in hybrid retrieval."
                    ),
                }
            )
        return out

    seen_titles: list[str] = []
    seen_ids = {int(pid) for pid in (state.get("seen_product_ids") or [])}
    for candidate in candidates:
        if int(candidate.get("id") or -1) in seen_ids:
            seen_titles.append(str(candidate.get("title")))

    if not mesh_available():
        graded = _rerank(heuristic())
        relevant = sum(1 for item in graded if item["is_relevant"])
        return {
            "graded_products": graded,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": f"heuristic grading — {relevant}/{len(graded)} relevant",
        }

    try:
        payload = call_llm_json(
            messages=[
                {"role": "system", "content": RELEVANCE_GRADER_SYSTEM},
                {
                    "role": "user",
                    "content": relevance_grader_user(digest, signals, candidates, seen_titles),
                },
            ],
            model=settings.mesh_model_grader,
            temperature=0.0,
            purpose=NODE_RELEVANCE_GRADER,
            telemetry=telemetry,
        )
        grades_raw = payload.get("grades") if isinstance(payload, dict) else payload
        if not isinstance(grades_raw, list) or not grades_raw:
            raise MeshUnavailableError("grader returned no grades")

        by_index: dict[int, dict[str, Any]] = {}
        for grade in grades_raw:
            if not isinstance(grade, dict):
                continue
            try:
                index = int(grade.get("index"))
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(candidates):
                by_index[index] = grade

        graded: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            grade = by_index.get(index)
            if grade is None:
                # Ungraded candidate: fall back to its retrieval standing rather
                # than silently dropping a potentially good course.
                overlap = _score_overlap(candidate, topic_terms)
                graded.append(
                    {
                        "product": candidate,
                        "relevance_score": round(_clamp(overlap, default=0.3), 3),
                        "is_relevant": overlap >= 0.5,
                        "reason": "Not graded by the judge; scored from retrieval overlap.",
                    }
                )
                continue
            score = _clamp(grade.get("relevance_score"), default=0.5)
            declared = grade.get("is_relevant")
            is_relevant = bool(declared) if isinstance(declared, bool) else score >= 0.5
            graded.append(
                {
                    "product": candidate,
                    "relevance_score": round(score, 3),
                    "is_relevant": is_relevant,
                    "reason": str(grade.get("reason") or "")[:300],
                }
            )

        graded = _rerank(graded)
        relevant = sum(1 for item in graded if item["is_relevant"])
        return {
            "graded_products": graded,
            "telemetry": telemetry,
            "_detail": f"judge graded {len(graded)} — {relevant} relevant",
        }

    except MeshUnavailableError as exc:
        graded = _rerank(heuristic())
        relevant = sum(1 for item in graded if item["is_relevant"])
        return {
            "graded_products": graded,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": f"Mesh fallback ({exc}) — {relevant} relevant"[:200],
        }


def count_relevant(state: RecommendationState) -> int:
    """Number of candidates the grader marked relevant."""
    return sum(1 for item in (state.get("graded_products") or []) if item.get("is_relevant"))


# --------------------------------------------------------------------------- #
# 5. retrieval_refiner                                                        #
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "a", "an", "and", "the", "that", "for", "with", "who", "someone", "course",
    "courses", "practical", "hands", "on", "learner", "their", "them", "this",
    "next", "step", "natural", "suitable", "as", "in", "of", "to", "is", "it",
}


def _broaden_query_heuristically(query: str, signals: list[dict[str, Any]]) -> str:
    """Widen a query without an LLM: keep salient nouns, add discipline framing."""
    keywords = [
        word.strip(".,'\"")
        for word in query.lower().split()
        if word.strip(".,'\"") not in _STOPWORDS and len(word) > 3
    ]
    top_topics = [str(signal.get("topic", "")) for signal in signals[:2] if signal.get("topic")]
    core = ", ".join(dict.fromkeys(top_topics + keywords[:6]))
    return (
        f"Introductory through intermediate courses covering {core} and closely "
        "related foundational skills, including adjacent tools and career-relevant "
        "applications of the same discipline."
    )


@instrumented(NODE_RETRIEVAL_REFINER)
def retrieval_refiner(state: RecommendationState) -> dict[str, Any]:
    """Broaden the retrieval query after an under-productive search.

    Entered only via the conditional edge from ``relevance_grader`` when fewer
    than ``AGENT_MIN_RELEVANT_PRODUCTS`` survivors remain and the retry budget is
    not exhausted.  May also drop the price and/or skill-level filters, which are
    the usual culprits behind an over-constrained search.
    """
    telemetry = _telemetry(state)
    previous_query = str(state.get("retrieval_query") or "")
    signals = list(state.get("interest_signals") or [])
    attempt = int(state.get("retry_count") or 0) + 1
    filters = dict(state.get("retrieval_filters") or {})
    history = list(state.get("refinement_history") or [])

    rejected = [
        str(item.get("reason"))
        for item in (state.get("graded_products") or [])
        if not item.get("is_relevant") and item.get("reason")
    ]

    heuristic_query = _broaden_query_heuristically(previous_query, signals)

    def relax(drop_price: bool, drop_skill: bool) -> dict[str, Any]:
        """Remove the named filters, keeping explicit exclusions intact."""
        relaxed = dict(filters)
        if drop_price:
            relaxed.pop("max_price", None)
            relaxed.pop("min_price", None)
        if drop_skill:
            relaxed.pop("skill_levels", None)
        return relaxed

    if not mesh_available():
        history.append(f"attempt {attempt}: heuristic broadening")
        return {
            "retrieval_query": heuristic_query,
            "retrieval_filters": relax(True, attempt >= 2),
            "retry_count": attempt,
            "refinement_history": history,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": "heuristic broadening",
        }

    try:
        payload = call_llm_json(
            messages=[
                {"role": "system", "content": RETRIEVAL_REFINER_SYSTEM},
                {
                    "role": "user",
                    "content": retrieval_refiner_user(
                        previous_query, count_relevant(state), attempt, filters, rejected
                    ),
                },
            ],
            model=settings.mesh_model_reasoning,
            temperature=0.4,
            purpose=NODE_RETRIEVAL_REFINER,
            telemetry=telemetry,
        )
        if not isinstance(payload, dict):
            raise MeshUnavailableError("refiner returned a non-object")

        new_query = str(payload.get("retrieval_query") or "").strip() or heuristic_query
        strategy = str(payload.get("strategy") or "broadened")[:120]
        history.append(f"attempt {attempt}: {strategy}")

        return {
            "retrieval_query": new_query,
            "retrieval_filters": relax(
                bool(payload.get("drop_price_filter")), bool(payload.get("drop_skill_filter"))
            ),
            "retry_count": attempt,
            "refinement_history": history,
            "telemetry": telemetry,
            "_detail": strategy,
        }

    except MeshUnavailableError as exc:
        history.append(f"attempt {attempt}: heuristic broadening after Mesh failure")
        return {
            "retrieval_query": heuristic_query,
            "retrieval_filters": relax(True, attempt >= 2),
            "retry_count": attempt,
            "refinement_history": history,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": f"Mesh fallback: {exc}"[:200],
        }


# --------------------------------------------------------------------------- #
# 6. persuasion_writer                                                        #
# --------------------------------------------------------------------------- #

def _select_final(state: RecommendationState) -> list[dict[str, Any]]:
    """Pick the products that make the cut, preferring relevant ones.

    If the grader found fewer than the target count, the best-ranked irrelevant
    candidates backfill the list — showing six good-enough courses beats showing
    two, and the narrative is honest about the weaker signal.
    """
    graded = list(state.get("graded_products") or [])
    target = settings.agent_final_product_count

    chosen: list[dict[str, Any]] = []
    for item in graded:
        if item.get("is_relevant"):
            product = dict(item["product"])
            product["reason"] = item.get("reason") or ""
            product["relevance_score"] = item.get("relevance_score")
            chosen.append(product)
        if len(chosen) >= target:
            break

    if len(chosen) < target:
        chosen_ids = {int(p.get("id") or -1) for p in chosen}
        for item in graded:
            if item.get("is_relevant"):
                continue
            product = dict(item["product"])
            if int(product.get("id") or -1) in chosen_ids:
                continue
            product["reason"] = item.get("reason") or ""
            product["relevance_score"] = item.get("relevance_score")
            chosen.append(product)
            if len(chosen) >= target:
                break

    return chosen


def _fallback_narrative(
    products: list[dict[str, Any]], signals: list[dict[str, Any]], display_name: str
) -> tuple[str, str]:
    """Compose a decent headline + narrative without an LLM.

    Deterministic but genuinely readable: it names the strongest observed topic,
    the evidence behind it, and the shape of the shortlist.
    """
    top_topic = str(signals[0].get("topic")) if signals else "your recent browsing"
    second = str(signals[1].get("topic")) if len(signals) > 1 else ""
    categories = [str(product.get("category")) for product in products if product.get("category")]
    unique_categories = list(dict.fromkeys(categories))
    levels = list(dict.fromkeys(str(p.get("skill_level")) for p in products if p.get("skill_level")))

    headline = f"Next steps in {top_topic}"[:60]

    body = [
        f"{display_name}, your recent activity keeps circling back to {top_topic}"
        + (f", with a secondary pull toward {second}" if second else "")
        + "."
    ]
    if unique_categories:
        body.append(
            "So this shortlist stays deliberately close to that: "
            + ", ".join(unique_categories[:3])
            + "."
        )
    if products:
        body.append(
            f"Start with {products[0].get('title')} — it lines up most directly with what "
            "you have been reading, and it gives you something you can apply immediately."
        )
    if len(products) > 1:
        body.append(
            f"The remaining {len(products) - 1} picks widen the same thread rather than "
            "changing the subject, so you can go deeper without losing momentum."
        )
    if levels:
        body.append(
            f"Everything here is pitched at the {'/'.join(levels[:2])} level you have been "
            "working at."
        )

    return headline, " ".join(body)


@instrumented(NODE_PERSUASION_WRITER)
def persuasion_writer(state: RecommendationState) -> dict[str, Any]:
    """Write the persuasive narrative and per-product pitches via Mesh.

    Uses ``MESH_MODEL_WRITER`` (Claude by default — the best persuasion writer of
    the routed models) at a higher temperature than the analytical nodes.
    """
    telemetry = _telemetry(state)
    products = _select_final(state)

    if not products:
        return {
            "final_products": [],
            "narrative": (
                "We don't have enough signal to recommend anything specific yet. "
                "Browse a few courses and your personalised picks will appear here."
            ),
            "headline": "Your recommendations are warming up",
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": "no products survived retrieval",
        }

    signals = list(state.get("interest_signals") or [])
    digest = str(state.get("behavior_digest") or "")

    with session_scope() as db:
        user = db.get(User, int(state["user_id"]))
        display_name = user.display_name if user else "there"

    if not mesh_available():
        headline, narrative = _fallback_narrative(products, signals, display_name)
        return {
            "final_products": products,
            "narrative": narrative,
            "headline": headline,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": "template narrative (Mesh not configured)",
        }

    try:
        payload = call_llm_json(
            messages=[
                {"role": "system", "content": PERSUASION_WRITER_SYSTEM},
                {
                    "role": "user",
                    "content": persuasion_writer_user(digest, signals, products, display_name),
                },
            ],
            model=settings.mesh_model_writer,
            temperature=0.75,
            purpose=NODE_PERSUASION_WRITER,
            telemetry=telemetry,
        )
        if not isinstance(payload, dict):
            raise MeshUnavailableError("writer returned a non-object")

        narrative = str(payload.get("narrative") or "").strip()
        headline = str(payload.get("headline") or "").strip()[:200]
        if not narrative:
            raise MeshUnavailableError("writer returned an empty narrative")

        pitches: dict[int, str] = {}
        for pitch in payload.get("pitches") or []:
            if not isinstance(pitch, dict):
                continue
            try:
                index = int(pitch.get("index"))
            except (TypeError, ValueError):
                continue
            text = str(pitch.get("pitch") or "").strip()
            if text and 1 <= index <= len(products):
                pitches[index] = text[:300]

        for index, product in enumerate(products, start=1):
            product["pitch"] = pitches.get(index) or product.get("reason") or ""

        fallback_headline, _ = _fallback_narrative(products, signals, display_name)
        return {
            "final_products": products,
            "narrative": narrative,
            "headline": headline or fallback_headline,
            "telemetry": telemetry,
            "_detail": f"{len(narrative)} chars, {len(pitches)}/{len(products)} pitches",
        }

    except MeshUnavailableError as exc:
        headline, narrative = _fallback_narrative(products, signals, display_name)
        return {
            "final_products": products,
            "narrative": narrative,
            "headline": headline,
            "telemetry": telemetry,
            "_degraded": True,
            "_detail": f"Mesh fallback: {exc}"[:200],
        }


# --------------------------------------------------------------------------- #
# 7. recommendation_storer                                                    #
# --------------------------------------------------------------------------- #

@instrumented(NODE_RECOMMENDATION_STORER)
def recommendation_storer(state: RecommendationState) -> dict[str, Any]:
    """Persist the recommendation, deactivate stale ones and refresh the cache.

    Deactivation and insertion happen in the same transaction, so a reader can
    never observe two active recommendations for one user.
    """
    from app.agent.state import summarise_state  # local import avoids a cycle

    started = time.perf_counter()
    user_id = int(state["user_id"])
    products = list(state.get("final_products") or [])
    telemetry = _telemetry(state)

    # The @instrumented decorator appends this node's log entry *after* the node
    # returns, which is too late to be persisted in the row it writes. So the
    # entry is added here, with the elapsed time up to the point of the insert
    # (it therefore excludes the commit itself — a sub-millisecond difference).
    node_log = list(state.get("node_log") or [])
    node_log.append(
        NodeExecution(
            node=NODE_RECOMMENDATION_STORER,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            degraded=False,
            detail="persist row, deactivate previous, invalidate caches",
        )
    )

    trace = summarise_state({**state, "node_log": node_log})  # type: ignore[arg-type]
    latency_ms = sum(float(entry.get("duration_ms") or 0.0) for entry in node_log)

    with session_scope() as db:
        # Invalidate previous recommendations for this user.
        stale = list(
            db.scalars(
                select(Recommendation).where(
                    Recommendation.user_id == user_id,
                    Recommendation.is_active.is_(True),
                )
            )
        )
        for row in stale:
            row.is_active = False

        record = Recommendation(
            user_id=user_id,
            narrative=str(state.get("narrative") or ""),
            headline=str(state.get("headline") or "")[:240] or None,
            products=[_slim_product(product) for product in products],
            interest_signals=list(state.get("interest_signals") or []),
            behavior_digest=str(state.get("behavior_digest") or "") or None,
            retrieval_query=str(state.get("retrieval_query") or "")[:1000] or None,
            trigger_event_count=int(state.get("trigger_event_count") or 0),
            trigger_reason=str(state.get("trigger_reason") or "manual")[:80],
            agent_trace=trace,
            latency_ms=round(latency_ms, 2),
            degraded=bool(state.get("degraded")),
            is_active=True,
        )
        db.add(record)
        db.flush()
        recommendation_id = int(record.id)
        payload = record.to_dict()

    invalidate_recommendation_cache(user_id)
    cache_active_recommendation(user_id, payload)
    clear_agent_pending(user_id)

    logger.info(
        "Stored recommendation id=%s user=%s products=%d degraded=%s latency=%.0fms "
        "mesh_calls=%d tokens=%d",
        recommendation_id, user_id, len(products), bool(state.get("degraded")),
        latency_ms, len(telemetry.calls), telemetry.total_tokens,
    )

    return {
        "recommendation_id": recommendation_id,
        "telemetry": telemetry,
        "_detail": f"stored id={recommendation_id} deactivated={len(stale)}",
    }


def _slim_product(product: dict[str, Any]) -> dict[str, Any]:
    """Trim a product dict to what the UI and email template actually render."""
    return {
        "id": product.get("id"),
        "title": product.get("title"),
        "description": str(product.get("description") or "")[:400],
        "category": product.get("category"),
        "tags": product.get("tags") or [],
        "price": product.get("price"),
        "skill_level": product.get("skill_level"),
        "duration": product.get("duration"),
        "thumbnail_url": product.get("thumbnail_url"),
        "instructor": product.get("instructor"),
        "rating": product.get("rating"),
        "pitch": product.get("pitch") or product.get("reason") or "",
        "reason": product.get("reason") or "",
        "relevance_score": product.get("relevance_score"),
        "retrieval_mode": product.get("retrieval_mode"),
    }


def utcnow_iso() -> str:
    """Current UTC timestamp in ISO-8601 form (used by the graph runner)."""
    return datetime.now(timezone.utc).isoformat()
