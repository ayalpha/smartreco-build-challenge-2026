"""Agent execution: locking, background dispatch and graph invocation.

Concurrency model
-----------------
Event ingestion must return in single-digit milliseconds, so it never runs the
graph inline.  Instead it calls :func:`maybe_dispatch`, which:

1. evaluates the trigger policy (:mod:`app.agent.triggers`);
2. claims a short-lived Redis lock for the user (:mod:`app.cache`) so a burst of
   events cannot start two concurrent runs for the same person;
3. hands the work to a background worker — the shared APScheduler instance when
   it is running, otherwise a small local thread pool.

:func:`run_agent` is the synchronous entry point used by the worker, the manual
"refresh" endpoint and the daily digest job.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from app.agent.graph import get_graph
from app.agent.observability import build_run_config
from app.agent.state import RecommendationState, make_initial_state, summarise_state
from app.agent.triggers import (
    REASON_MANUAL,
    TriggerDecision,
    evaluate,
)
from app.cache import (
    acquire_agent_lock,
    clear_agent_pending,
    mark_agent_pending,
    release_agent_lock,
)
from app.config import get_settings
from app.database import session_scope

logger = logging.getLogger(__name__)
settings = get_settings()

#: Fallback worker pool for environments without a running scheduler (tests, CLI).
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Return the lazily-created fallback thread pool."""
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:  # pragma: no branch - race guard
            _executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="smartreco-agent"
            )
    return _executor


def shutdown_executor(wait: bool = False) -> None:
    """Shut the fallback pool down (called from the app's lifespan teardown)."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=wait)
            _executor = None
            logger.info("Agent fallback executor shut down")


@dataclass
class AgentRunResult:
    """Outcome of one graph invocation."""

    ok: bool
    user_id: int
    recommendation_id: Optional[int] = None
    reason: str = REASON_MANUAL
    degraded: bool = False
    duration_ms: float = 0.0
    skipped: bool = False
    error: Optional[str] = None
    trace: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable projection for API responses."""
        return {
            "ok": self.ok,
            "user_id": self.user_id,
            "recommendation_id": self.recommendation_id,
            "reason": self.reason,
            "degraded": self.degraded,
            "duration_ms": round(self.duration_ms, 2),
            "skipped": self.skipped,
            "error": self.error,
        }


def run_agent(
    user_id: int,
    *,
    reason: str = REASON_MANUAL,
    event_count: int = 0,
    respect_lock: bool = True,
) -> AgentRunResult:
    """Run the recommendation graph for one user, end to end.

    Args:
        user_id: The user to generate a recommendation for.
        reason: Trigger reason, recorded on the row and used as a LangSmith tag.
        event_count: The user's event count at trigger time.
        respect_lock: When True (default) the run is skipped if another worker
            already holds this user's lock.  The manual endpoint passes False
            after claiming the lock itself.

    Returns:
        An :class:`AgentRunResult`.  Never raises — failures are captured.
    """
    started = time.perf_counter()
    holds_lock = False

    if respect_lock:
        if not acquire_agent_lock(user_id):
            return AgentRunResult(
                ok=False, user_id=user_id, reason=reason, skipped=True,
                error="another run is already in progress for this user",
            )
        holds_lock = True

    try:
        logger.info("Agent run starting — user=%s reason=%s events=%s",
                    user_id, reason, event_count)

        state: RecommendationState = make_initial_state(
            user_id, trigger_reason=reason, trigger_event_count=event_count
        )
        config = build_run_config(
            user_id, trigger_reason=reason, event_count=event_count
        )

        final_state: dict[str, Any] = get_graph().invoke(state, config=config)
        duration_ms = (time.perf_counter() - started) * 1000.0

        recommendation_id = final_state.get("recommendation_id")
        degraded = bool(final_state.get("degraded"))
        error = final_state.get("error")
        trace = summarise_state(final_state)  # type: ignore[arg-type]

        logger.info(
            "Agent run finished — user=%s reason=%s id=%s degraded=%s %.0fms",
            user_id, reason, recommendation_id, degraded, duration_ms,
        )

        return AgentRunResult(
            ok=recommendation_id is not None,
            user_id=user_id,
            recommendation_id=int(recommendation_id) if recommendation_id else None,
            reason=reason,
            degraded=degraded,
            duration_ms=duration_ms,
            error=str(error) if error else None,
            trace=trace,
        )

    except Exception as exc:  # noqa: BLE001 - a failed run must not kill the worker
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.exception("Agent run crashed for user=%s (reason=%s)", user_id, reason)
        return AgentRunResult(
            ok=False, user_id=user_id, reason=reason, degraded=True,
            duration_ms=duration_ms, error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        clear_agent_pending(user_id)
        if holds_lock:
            release_agent_lock(user_id)


def _submit(user_id: int, reason: str, event_count: int) -> None:
    """Hand a run to the scheduler if available, else to the local pool."""
    try:
        from app.scheduler.jobs import enqueue_agent_run  # late import breaks the cycle

        if enqueue_agent_run(user_id, reason, event_count):
            return
    except Exception:
        logger.debug("Scheduler dispatch unavailable — using the local pool",
                     exc_info=True)

    future: Future = _get_executor().submit(
        run_agent, user_id, reason=reason, event_count=event_count, respect_lock=False
    )
    future.add_done_callback(
        lambda done: logger.debug("Background agent run completed: %s", _future_summary(done))
    )


def _future_summary(future: Future) -> str:
    """Render a completed future for debug logging without raising."""
    try:
        result = future.result()
        return str(result.as_dict()) if isinstance(result, AgentRunResult) else str(result)
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}"


def maybe_dispatch(user_id: int) -> TriggerDecision:
    """Evaluate the trigger policy and dispatch a background run if warranted.

    This is the function the event-ingest endpoint calls.  It performs one cheap
    pair of ``COUNT`` queries and, in the common case, returns without doing
    anything else.

    Args:
        user_id: The user whose events just arrived.

    Returns:
        The :class:`TriggerDecision` that was made (useful for the API response
        and for tests).
    """
    with session_scope() as db:
        decision = evaluate(db, user_id)

    if not decision.should_run:
        logger.debug("No trigger for user=%s: %s", user_id, decision.detail)
        return decision

    if not acquire_agent_lock(user_id):
        logger.info("Trigger fired for user=%s but a run is already in flight", user_id)
        return TriggerDecision(
            should_run=False,
            reason=decision.reason,
            event_count=decision.event_count,
            new_events=decision.new_events,
            detail="a run is already in progress",
        )

    mark_agent_pending(user_id, decision.reason)
    logger.info("Dispatching agent run for user=%s (%s: %s)",
                user_id, decision.reason, decision.detail)
    _submit(user_id, decision.reason, decision.event_count)
    return decision


def run_agent_now(
    user_id: int, reason: str = REASON_MANUAL, *, respect_lock: bool = True
) -> AgentRunResult:
    """Run the agent synchronously, bypassing the trigger policy.

    Used by the "Refresh my picks" button, the digest job and the seed script —
    all of which need the result immediately rather than eventually.

    Args:
        user_id: The user to generate a recommendation for.
        reason: Trigger reason recorded on the row.
        respect_lock: Whether to claim the per-user lock.  Pass ``False`` when the
            caller has *already* claimed it (the refresh endpoint does, so that it
            can return ``409`` instead of silently no-op'ing); leaving it ``True``
            there would make the run collide with the caller's own lock.

    Returns:
        The :class:`AgentRunResult`.
    """
    with session_scope() as db:
        from app.agent.triggers import count_events

        event_count = count_events(db, user_id)

    mark_agent_pending(user_id, reason)
    return run_agent(
        user_id, reason=reason, event_count=event_count, respect_lock=respect_lock
    )
