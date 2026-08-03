"""APScheduler job definitions and the shared scheduler instance.

Jobs
----
``daily_digest``
    Runs at ``DIGEST_SCHEDULE_HOUR:DIGEST_SCHEDULE_MINUTE`` (18:00 by default).
    For every opted-in user with at least ``DIGEST_MIN_EVENTS_TODAY`` events
    today it runs the recommendation agent and emails the narrative plus the top
    ``DIGEST_PRODUCT_COUNT`` products (BONUS 2).

``housekeeping``
    Hourly: releases stale "pending" flags whose runs died, so the UI's
    generating state can never get stuck on.

The same scheduler also doubles as the **background task queue** for
event-triggered agent runs: :func:`enqueue_agent_run` adds a one-off ``date`` job,
which keeps ``POST /api/events`` free of any agent work.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import func, select

from app.agent.runner import run_agent
from app.agent.triggers import REASON_SCHEDULED_DIGEST
from app.cache import clear_agent_pending, get_agent_pending, release_agent_lock
from app.config import get_settings
from app.database import session_scope
from app.models.event import Event
from app.models.recommendation import Recommendation
from app.models.user import User
from app.scheduler.email_digest import deliver_digest

logger = logging.getLogger(__name__)
settings = get_settings()

JOB_DAILY_DIGEST = "smartreco:daily_digest"
JOB_HOUSEKEEPING = "smartreco:housekeeping"

_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Scheduler lifecycle                                                         #
# --------------------------------------------------------------------------- #

def get_scheduler() -> BackgroundScheduler:
    """Return the process-wide scheduler, creating it on first use."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    with _scheduler_lock:
        if _scheduler is None:  # pragma: no branch - race guard
            _scheduler = BackgroundScheduler(
                timezone=settings.scheduler_timezone,
                executors={"default": APThreadPoolExecutor(max_workers=4)},
                job_defaults={
                    "coalesce": True,      # collapse missed runs into one
                    "max_instances": 1,    # never overlap the same job
                    "misfire_grace_time": 900,
                },
            )
    return _scheduler


def start_scheduler() -> Optional[BackgroundScheduler]:
    """Register jobs and start the scheduler.

    Returns:
        The running scheduler, or None when ``SCHEDULER_ENABLED`` is false.
    """
    if not settings.scheduler_enabled:
        logger.info("Scheduler disabled (SCHEDULER_ENABLED=false)")
        return None

    scheduler = get_scheduler()
    if scheduler.running:
        return scheduler

    scheduler.add_job(
        daily_digest_job,
        trigger=CronTrigger(
            hour=settings.digest_schedule_hour,
            minute=settings.digest_schedule_minute,
            timezone=settings.scheduler_timezone,
        ),
        id=JOB_DAILY_DIGEST,
        name="Daily proactive recommendation digest",
        replace_existing=True,
    )
    scheduler.add_job(
        housekeeping_job,
        trigger=IntervalTrigger(hours=1),
        id=JOB_HOUSEKEEPING,
        name="Release stale agent flags",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started (tz=%s) — digest at %02d:%02d, housekeeping hourly",
        settings.scheduler_timezone, settings.digest_schedule_hour,
        settings.digest_schedule_minute,
    )
    return scheduler


def shutdown_scheduler(wait: bool = False) -> None:
    """Stop the scheduler if it is running."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=wait)
            logger.info("Scheduler shut down")
        _scheduler = None


def scheduler_status() -> dict[str, Any]:
    """Snapshot of scheduler state for ``/health`` and the admin dashboard."""
    scheduler = _scheduler
    if scheduler is None or not scheduler.running:
        return {"running": False, "enabled": settings.scheduler_enabled, "jobs": []}

    return {
        "running": True,
        "enabled": True,
        "timezone": str(settings.scheduler_timezone),
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in scheduler.get_jobs()
        ],
    }


# --------------------------------------------------------------------------- #
# Background dispatch for event-triggered runs                                #
# --------------------------------------------------------------------------- #

def enqueue_agent_run(user_id: int, reason: str, event_count: int) -> bool:
    """Queue an agent run as a one-off scheduler job.

    Args:
        user_id: Subject of the run.
        reason: Trigger reason.
        event_count: Event count at trigger time.

    Returns:
        True when the job was accepted; False when no scheduler is running (the
        caller then falls back to its local thread pool).
    """
    scheduler = _scheduler
    if scheduler is None or not scheduler.running:
        return False

    job_id = f"smartreco:agent:{user_id}"
    try:
        scheduler.add_job(
            run_agent,
            trigger=DateTrigger(run_date=datetime.now(timezone.utc)),
            kwargs={
                "user_id": user_id,
                "reason": reason,
                "event_count": event_count,
                "respect_lock": False,  # the caller already holds the lock
            },
            id=job_id,
            name=f"Agent run for user {user_id} ({reason})",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.debug("Queued agent run job %s", job_id)
        return True
    except Exception:
        logger.warning("Could not queue agent run for user=%s", user_id, exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Daily digest job (BONUS 2)                                                  #
# --------------------------------------------------------------------------- #

def _digest_candidates(min_events: int) -> list[int]:
    """Return ids of opted-in users with at least ``min_events`` events today.

    "Today" means since UTC midnight, matching the ``SCHEDULER_TIMEZONE=UTC``
    default; change both together if you localise the schedule.
    """
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    with session_scope() as db:
        rows = db.execute(
            select(Event.user_id, func.count(Event.id))
            .join(User, User.id == Event.user_id)
            .where(
                Event.user_id.isnot(None),
                Event.timestamp >= start_of_day,
                User.is_active.is_(True),
                User.digest_opt_in.is_(True),
            )
            .group_by(Event.user_id)
            .having(func.count(Event.id) >= min_events)
        ).all()

    return [int(row[0]) for row in rows if row[0] is not None]


def daily_digest_job() -> dict[str, Any]:
    """Generate and deliver the daily digest to every qualifying user.

    For each candidate: run the agent (so the digest reflects *today's*
    behaviour), then deliver the narrative and top products by email and, when
    configured, Telegram.

    Returns:
        ``{"candidates", "sent", "skipped", "failed", "details"}``.
    """
    summary: dict[str, Any] = {
        "candidates": 0, "sent": 0, "skipped": 0, "failed": 0, "details": [],
    }

    if not settings.email_enabled:
        logger.info("Daily digest skipped: EMAIL_ENABLED=false")
        return summary

    candidates = _digest_candidates(settings.digest_min_events_today)
    summary["candidates"] = len(candidates)
    logger.info("Daily digest starting for %d candidate user(s)", len(candidates))

    for user_id in candidates:
        try:
            result = run_agent(
                user_id, reason=REASON_SCHEDULED_DIGEST, event_count=0, respect_lock=True
            )
            if not result.ok or result.recommendation_id is None:
                summary["skipped"] += 1
                summary["details"].append(
                    {"user_id": user_id, "status": "skipped", "detail": result.error}
                )
                logger.info("Digest skipped for user=%s: %s", user_id, result.error)
                continue

            outcomes = deliver_digest(user_id, result.recommendation_id)
            delivered = any(outcome.ok for outcome in outcomes)
            if delivered:
                summary["sent"] += 1
            else:
                summary["failed"] += 1
            summary["details"].append(
                {
                    "user_id": user_id,
                    "status": "sent" if delivered else "failed",
                    "recommendation_id": result.recommendation_id,
                    "channels": [
                        {"channel": o.channel, "ok": o.ok, "error": o.error}
                        for o in outcomes
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001 - one user must not break the job
            summary["failed"] += 1
            summary["details"].append(
                {"user_id": user_id, "status": "error", "detail": str(exc)[:300]}
            )
            logger.exception("Digest failed for user=%s", user_id)

    logger.info(
        "Daily digest finished — candidates=%d sent=%d skipped=%d failed=%d",
        summary["candidates"], summary["sent"], summary["skipped"], summary["failed"],
    )
    return summary


def run_daily_digest_now() -> dict[str, Any]:
    """Run the digest job immediately (admin action / CLI)."""
    logger.info("Manual digest run requested")
    return daily_digest_job()


# --------------------------------------------------------------------------- #
# Housekeeping                                                                #
# --------------------------------------------------------------------------- #

def housekeeping_job() -> dict[str, Any]:
    """Clear stale pending/lock flags left behind by crashed runs.

    A run that dies between claiming the lock and storing its result would
    otherwise leave the UI showing a loading skeleton until the TTL expired. This
    job reconciles the flags against what is actually in the database.
    """
    cleared = 0
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=max(settings.agent_lock_ttl_seconds * 2, 600)
    )

    with session_scope() as db:
        user_ids = [
            int(row[0])
            for row in db.execute(select(User.id).where(User.is_active.is_(True))).all()
        ]
        for user_id in user_ids:
            if not get_agent_pending(user_id):
                continue
            latest = db.scalars(
                select(Recommendation)
                .where(Recommendation.user_id == user_id)
                .order_by(Recommendation.created_at.desc())
                .limit(1)
            ).first()
            created_at = latest.created_at if latest else None
            if created_at is not None and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at is None or created_at < cutoff:
                clear_agent_pending(user_id)
                release_agent_lock(user_id)
                cleared += 1

    if cleared:
        logger.info("Housekeeping cleared %d stale agent flag(s)", cleared)
    return {"cleared": cleared}
