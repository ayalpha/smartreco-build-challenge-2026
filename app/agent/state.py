"""Typed state schema for the recommendation graph (BONUS 1).

LangGraph threads a single mutable state object through the node sequence; each
node returns a *partial* update which LangGraph merges.  Declaring the schema as
a ``TypedDict`` with ``total=False`` gives full editor/mypy support while letting
nodes return only the keys they own.

State flow
----------
::

    user_id ─┐
             ├─ activity_analyzer ──▶ behavior_digest
             ├─ interest_extractor ─▶ interest_signals, retrieval_query, filters
             ├─ retrieval_node ─────▶ retrieved_products
             ├─ relevance_grader ───▶ graded_products  ──┐ conditional edge
             ├─ retrieval_refiner ──▶ retrieval_query ◀──┘ (retry_count += 1)
             ├─ persuasion_writer ──▶ narrative, headline, final_products
             └─ recommendation_storer ▶ recommendation_id
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from app.agent.mesh_client import MeshTelemetry


class InterestSignal(TypedDict, total=False):
    """One extracted interest with the evidence that supports it."""

    topic: str
    confidence: float
    evidence: str


class GradedProduct(TypedDict, total=False):
    """A retrieved product after the relevance grader has judged it."""

    product: dict[str, Any]
    relevance_score: float
    is_relevant: bool
    reason: str


class NodeExecution(TypedDict, total=False):
    """Per-node timing/diagnostics entry appended to ``node_log``."""

    node: str
    duration_ms: float
    degraded: bool
    detail: str


class RecommendationState(TypedDict, total=False):
    """The state object threaded through the recommendation graph."""

    # ---- inputs -----------------------------------------------------------
    user_id: int
    trigger_reason: str
    trigger_event_count: int
    now_iso: str

    # ---- activity_analyzer ------------------------------------------------
    raw_events: list[dict[str, Any]]
    behavior_digest: str

    # ---- interest_extractor ----------------------------------------------
    interest_signals: list[InterestSignal]
    retrieval_query: str
    #: Inferred metadata constraints (skill level / price band) — BONUS 4.
    retrieval_filters: dict[str, Any]
    seen_product_ids: list[int]

    # ---- retrieval / grading ---------------------------------------------
    retrieved_products: list[dict[str, Any]]
    graded_products: list[GradedProduct]
    retry_count: int
    refinement_history: list[str]

    # ---- writer -----------------------------------------------------------
    final_products: list[dict[str, Any]]
    narrative: str
    headline: str

    # ---- storer -----------------------------------------------------------
    recommendation_id: Optional[int]

    # ---- diagnostics ------------------------------------------------------
    node_log: list[NodeExecution]
    degraded: bool
    telemetry: MeshTelemetry
    error: Optional[str]


def make_initial_state(
    user_id: int,
    *,
    trigger_reason: str = "manual",
    trigger_event_count: int = 0,
) -> RecommendationState:
    """Build a fully-populated starting state.

    Pre-seeding every key means nodes can read with plain subscripting and never
    need ``.get()`` guards for their own inputs.

    Args:
        user_id: The user the recommendation is for.
        trigger_reason: Why the agent is running (``event_threshold``,
            ``first_time``, ``stale``, ``manual``, ``scheduled_digest``).
        trigger_event_count: Total events the user had when the run was queued.

    Returns:
        A ready-to-invoke :class:`RecommendationState`.
    """
    return RecommendationState(
        user_id=user_id,
        trigger_reason=trigger_reason,
        trigger_event_count=trigger_event_count,
        now_iso=datetime.now(timezone.utc).isoformat(),
        raw_events=[],
        behavior_digest="",
        interest_signals=[],
        retrieval_query="",
        retrieval_filters={},
        seen_product_ids=[],
        retrieved_products=[],
        graded_products=[],
        retry_count=0,
        refinement_history=[],
        final_products=[],
        narrative="",
        headline="",
        recommendation_id=None,
        node_log=[],
        degraded=False,
        telemetry=MeshTelemetry(),
        error=None,
    )


def summarise_state(state: RecommendationState) -> dict[str, Any]:
    """Compact, JSON-safe view of a finished run for logs and ``agent_trace``."""
    telemetry = state.get("telemetry")
    return {
        "user_id": state.get("user_id"),
        "trigger_reason": state.get("trigger_reason"),
        "trigger_event_count": state.get("trigger_event_count"),
        "events_analysed": len(state.get("raw_events") or []),
        "interest_signals": state.get("interest_signals") or [],
        "retrieval_query": state.get("retrieval_query"),
        "retrieval_filters": state.get("retrieval_filters") or {},
        "retrieved": len(state.get("retrieved_products") or []),
        "relevant": sum(
            1 for g in (state.get("graded_products") or []) if g.get("is_relevant")
        ),
        "retry_count": state.get("retry_count", 0),
        "refinement_history": state.get("refinement_history") or [],
        "final_products": [p.get("id") for p in (state.get("final_products") or [])],
        "degraded": bool(state.get("degraded")),
        "error": state.get("error"),
        "node_log": state.get("node_log") or [],
        "mesh": telemetry.summary() if isinstance(telemetry, MeshTelemetry) else {},
    }
