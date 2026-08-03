"""The recommendation graph — a LangGraph state machine (BONUS 1).

Topology
--------
::

    ┌──────────────────────────────────────────────────────────────────────┐
    │                      RECOMMENDATION GRAPH                            │
    │                                                                      │
    │   START                                                              │
    │     │                                                                │
    │     ▼                                                                │
    │  activity_analyzer      summarise recent events -> behaviour digest   │
    │     │                                                                │
    │     ▼                                                                │
    │  interest_extractor     digest -> signals + query + metadata filters  │
    │     │                                                                │
    │     ▼                                                                │
    │  retrieval_node   ◀───────────────┐  hybrid dense⊕BM25 + RRF fusion   │
    │     │                             │                                  │
    │     ▼                             │                                  │
    │  relevance_grader                 │  LLM-as-judge grade + re-rank     │
    │     │                             │                                  │
    │     ├── enough relevant ──┐       │                                  │
    │     │                     │       │                                  │
    │     └── too few &         │       │                                  │
    │         retries left ──▶ retrieval_refiner  (broaden, relax filters)  │
    │                           │                                          │
    │                           ▼                                          │
    │                    persuasion_writer   narrative + per-product pitch  │
    │                           │                                          │
    │                           ▼                                          │
    │                  recommendation_storer  persist + invalidate caches   │
    │                           │                                          │
    │                           ▼                                          │
    │                          END                                         │
    └──────────────────────────────────────────────────────────────────────┘

The single conditional edge lives on ``relevance_grader`` and is implemented by
:func:`route_after_grading`.  The graph is compiled once per process with a
:class:`~langgraph.checkpoint.memory.MemorySaver` so every run is inspectable
step-by-step during debugging, and replayable by thread id.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    NODE_ACTIVITY_ANALYZER,
    NODE_INTEREST_EXTRACTOR,
    NODE_PERSUASION_WRITER,
    NODE_RECOMMENDATION_STORER,
    NODE_RELEVANCE_GRADER,
    NODE_RETRIEVAL,
    NODE_RETRIEVAL_REFINER,
    activity_analyzer,
    count_relevant,
    interest_extractor,
    persuasion_writer,
    recommendation_storer,
    relevance_grader,
    retrieval_node,
    retrieval_refiner,
)
from app.agent.observability import configure_langsmith
from app.agent.state import RecommendationState
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_compiled: Optional[Any] = None
_compile_lock = threading.Lock()


def route_after_grading(state: RecommendationState) -> str:
    """Conditional edge: proceed to the writer, or refine and retry retrieval.

    Decision table:

    ==========================================  =====================
    Condition                                   Next node
    ==========================================  =====================
    ``relevant >= AGENT_MIN_RELEVANT_PRODUCTS``  ``persuasion_writer``
    ``retry_count < AGENT_MAX_RETRIEVAL_RETRIES`` ``retrieval_refiner``
    otherwise (budget exhausted)                 ``persuasion_writer``
    ==========================================  =====================

    The third row is the important one: when the retry budget runs out we write a
    recommendation from whatever we have rather than returning nothing.
    """
    relevant = count_relevant(state)
    retries = int(state.get("retry_count") or 0)

    if relevant >= settings.agent_min_relevant_products:
        logger.info(
            "Routing to writer: %d relevant candidate(s) meets the threshold of %d",
            relevant, settings.agent_min_relevant_products,
        )
        return NODE_PERSUASION_WRITER

    if retries < settings.agent_max_retrieval_retries:
        logger.info(
            "Routing to refiner: only %d relevant candidate(s), retry %d/%d",
            relevant, retries + 1, settings.agent_max_retrieval_retries,
        )
        return NODE_RETRIEVAL_REFINER

    logger.info(
        "Retry budget exhausted (%d/%d) with %d relevant candidate(s) — writing anyway",
        retries, settings.agent_max_retrieval_retries, relevant,
    )
    return NODE_PERSUASION_WRITER


def build_graph() -> StateGraph:
    """Construct (but do not compile) the recommendation :class:`StateGraph`."""
    graph = StateGraph(RecommendationState)

    graph.add_node(NODE_ACTIVITY_ANALYZER, activity_analyzer)
    graph.add_node(NODE_INTEREST_EXTRACTOR, interest_extractor)
    graph.add_node(NODE_RETRIEVAL, retrieval_node)
    graph.add_node(NODE_RELEVANCE_GRADER, relevance_grader)
    graph.add_node(NODE_RETRIEVAL_REFINER, retrieval_refiner)
    graph.add_node(NODE_PERSUASION_WRITER, persuasion_writer)
    graph.add_node(NODE_RECOMMENDATION_STORER, recommendation_storer)

    graph.add_edge(START, NODE_ACTIVITY_ANALYZER)
    graph.add_edge(NODE_ACTIVITY_ANALYZER, NODE_INTEREST_EXTRACTOR)
    graph.add_edge(NODE_INTEREST_EXTRACTOR, NODE_RETRIEVAL)
    graph.add_edge(NODE_RETRIEVAL, NODE_RELEVANCE_GRADER)

    # The one conditional edge: grade -> (refine ⟲ retrieve) | write.
    graph.add_conditional_edges(
        NODE_RELEVANCE_GRADER,
        route_after_grading,
        {
            NODE_PERSUASION_WRITER: NODE_PERSUASION_WRITER,
            NODE_RETRIEVAL_REFINER: NODE_RETRIEVAL_REFINER,
        },
    )

    # Refinement loops straight back into retrieval.
    graph.add_edge(NODE_RETRIEVAL_REFINER, NODE_RETRIEVAL)

    graph.add_edge(NODE_PERSUASION_WRITER, NODE_RECOMMENDATION_STORER)
    graph.add_edge(NODE_RECOMMENDATION_STORER, END)

    return graph


def compile_graph(checkpointer: Optional[Any] = None) -> Any:
    """Compile the graph with a checkpointer.

    Args:
        checkpointer: Override for the checkpointer.  Defaults to a fresh
            :class:`MemorySaver`, which keeps every intermediate state in memory
            for debugging and replay.

    Returns:
        The compiled, invokable graph.
    """
    configure_langsmith()
    saver = checkpointer if checkpointer is not None else MemorySaver()
    compiled = build_graph().compile(checkpointer=saver)
    logger.info(
        "Recommendation graph compiled — 7 nodes, 1 conditional edge, "
        "checkpointer=%s", type(saver).__name__,
    )
    return compiled


def get_graph() -> Any:
    """Return the process-wide compiled graph, compiling it on first use."""
    global _compiled
    if _compiled is not None:
        return _compiled
    with _compile_lock:
        if _compiled is None:  # pragma: no branch - race guard
            _compiled = compile_graph()
    return _compiled


def reset_graph() -> None:
    """Drop the compiled singleton (used by tests)."""
    global _compiled
    with _compile_lock:
        _compiled = None


def render_ascii() -> str:
    """Return an ASCII rendering of the graph topology.

    Falls back to a hand-drawn diagram when the optional ``grandalf`` renderer is
    not installed, so this never fails in a minimal environment.
    """
    try:
        return get_graph().get_graph().draw_ascii()
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("ASCII graph rendering unavailable", exc_info=True)
        return (
            "START -> activity_analyzer -> interest_extractor -> retrieval_node -> "
            "relevance_grader -{enough}-> persuasion_writer -> recommendation_storer -> END\n"
            "                                       ^                 |\n"
            "                                       |            {too few}\n"
            "                                       +-- retrieval_refiner <-+"
        )


def render_mermaid() -> str:
    """Return a Mermaid definition of the graph (used in the README and admin UI)."""
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("Mermaid graph rendering unavailable", exc_info=True)
        return (
            "graph TD\n"
            "    START --> activity_analyzer\n"
            "    activity_analyzer --> interest_extractor\n"
            "    interest_extractor --> retrieval_node\n"
            "    retrieval_node --> relevance_grader\n"
            "    relevance_grader -->|>= 3 relevant| persuasion_writer\n"
            "    relevance_grader -->|< 3 relevant and retries left| retrieval_refiner\n"
            "    retrieval_refiner --> retrieval_node\n"
            "    persuasion_writer --> recommendation_storer\n"
            "    recommendation_storer --> END"
        )
