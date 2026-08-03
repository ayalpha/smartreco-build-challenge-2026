"""LangSmith observability wiring (BONUS 3).

LangSmith is configured through environment variables that the LangChain runtime
reads at trace time.  Rather than expecting operators to export them by hand, we
project our own settings onto those variables in :func:`configure_langsmith`, so
one ``LANGSMITH_TRACING=true`` in ``.env`` is enough.

Selective tracing
-----------------
Tracing is scoped to the agent graph only.  Health checks, the event-ingest hot
path and template rendering are never traced — they would bury the interesting
runs in noise and cost money for nothing.  The switch is the per-invocation
``config`` built by :func:`build_run_config`, plus a global kill-switch
(:func:`tracing_enabled`) that also protects against a missing API key.

Every traced run carries metadata (``user_id``, ``trigger_reason``,
``event_count``, models in play) and tags, so runs are filterable in the
LangSmith UI.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_configured = False


def tracing_enabled() -> bool:
    """True only when tracing is switched on *and* an API key is present."""
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_langsmith() -> bool:
    """Project settings onto the ``LANGSMITH_*`` environment variables.

    Idempotent, and safe to call when tracing is disabled (it then explicitly
    turns tracing off so a stray shell export cannot silently enable it).

    Returns:
        True when tracing is active for this process.
    """
    global _configured

    if not tracing_enabled():
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if settings.langsmith_tracing and not settings.langsmith_api_key:
            logger.warning(
                "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is missing — "
                "tracing stays disabled."
            )
        return False

    if _configured:
        return True

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = str(settings.langsmith_api_key)
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    # Legacy aliases, still honoured by older langchain-core releases.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = str(settings.langsmith_api_key)
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    _configured = True
    logger.info(
        "LangSmith tracing enabled — project=%r endpoint=%s",
        settings.langsmith_project, settings.langsmith_endpoint,
    )
    return True


def build_run_config(
    user_id: int,
    *,
    trigger_reason: str,
    event_count: int,
    thread_id: Optional[str] = None,
    extra_tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the LangGraph invocation config for one agent run.

    Always sets ``configurable.thread_id`` (required by the checkpointer) and,
    when tracing is on, attaches the run name, tags and custom metadata that make
    runs searchable in LangSmith.

    Args:
        user_id: Subject of the recommendation.
        trigger_reason: Why the run started.
        event_count: The user's event count at trigger time.
        thread_id: Checkpointer thread id.  Defaults to ``user-<id>`` so a user's
            successive runs share a checkpoint lineage and can be replayed.
        extra_tags: Additional LangSmith tags.

    Returns:
        A config dict suitable for ``graph.invoke(state, config=...)``.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id or f"user-{user_id}"},
        "recursion_limit": 40,
    }

    if not tracing_enabled():
        return config

    tags = ["smartreco", "recommendation-agent", f"trigger:{trigger_reason}"]
    if extra_tags:
        tags.extend(extra_tags)

    config.update(
        run_name=f"smartreco-recommendation-user-{user_id}",
        tags=tags,
        metadata={
            "user_id": user_id,
            "trigger_reason": trigger_reason,
            "event_count": event_count,
            "environment": settings.environment,
            "mesh_base_url": settings.mesh_base_url,
            "model_reasoning": settings.mesh_model_reasoning,
            "model_writer": settings.mesh_model_writer,
            "model_grader": settings.mesh_model_grader,
            "embedding_model": settings.mesh_embedding_model,
        },
    )
    return config


def observability_status() -> dict[str, Any]:
    """Snapshot of tracing configuration for ``/health`` and the admin dashboard."""
    return {
        "tracing_enabled": tracing_enabled(),
        "project": settings.langsmith_project if tracing_enabled() else None,
        "endpoint": settings.langsmith_endpoint if tracing_enabled() else None,
        "configured": _configured,
    }
