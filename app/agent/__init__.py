"""Agentic recommendation engine.

Public surface::

    from app.agent import get_graph, run_agent, maybe_dispatch

* :mod:`app.agent.mesh_client` — the single Mesh API gateway (mandatory for all AI)
* :mod:`app.agent.state`       — typed graph state
* :mod:`app.agent.nodes`       — the seven node implementations
* :mod:`app.agent.graph`       — LangGraph state machine + conditional routing
* :mod:`app.agent.triggers`    — when to run
* :mod:`app.agent.runner`      — how to run (locks, background dispatch)
* :mod:`app.agent.observability` — LangSmith tracing

Lazy re-exports
---------------
These names are resolved on first attribute access (PEP 562) rather than at
import time.  That is deliberate: ``app.vector_store.embeddings`` imports
``app.agent.mesh_client``, while ``app.agent.nodes`` imports
``app.vector_store.qdrant_client``.  Eagerly re-exporting the graph here would
turn that legitimate two-way package relationship into an import cycle whenever
``app.vector_store`` happened to be imported first (as ``scripts/seed_products.py``
does).  Lazy resolution keeps the convenient façade without the fragility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from app.agent.graph import compile_graph, get_graph, render_ascii, render_mermaid
    from app.agent.mesh_client import call_llm, embed_text, get_mesh_client
    from app.agent.runner import (
        AgentRunResult,
        maybe_dispatch,
        run_agent,
        run_agent_now,
    )
    from app.agent.state import RecommendationState, make_initial_state
    from app.agent.triggers import TriggerDecision, evaluate

#: Public name -> defining submodule.
_EXPORTS: dict[str, str] = {
    "AgentRunResult": "app.agent.runner",
    "RecommendationState": "app.agent.state",
    "TriggerDecision": "app.agent.triggers",
    "call_llm": "app.agent.mesh_client",
    "compile_graph": "app.agent.graph",
    "embed_text": "app.agent.mesh_client",
    "evaluate": "app.agent.triggers",
    "get_graph": "app.agent.graph",
    "get_mesh_client": "app.agent.mesh_client",
    "make_initial_state": "app.agent.state",
    "maybe_dispatch": "app.agent.runner",
    "render_ascii": "app.agent.graph",
    "render_mermaid": "app.agent.graph",
    "run_agent": "app.agent.runner",
    "run_agent_now": "app.agent.runner",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache so subsequent lookups skip this hook
    return value


def __dir__() -> list[str]:
    """Include the lazy exports in ``dir()`` output."""
    return sorted(set(globals()) | set(_EXPORTS))
