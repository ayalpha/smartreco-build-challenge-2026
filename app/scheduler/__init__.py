"""Scheduling package: APScheduler jobs and proactive digest delivery (BONUS 2)."""

from __future__ import annotations

from app.scheduler.email_digest import (
    DeliveryOutcome,
    deliver_digest,
    render_digest_html,
    render_digest_text,
)
from app.scheduler.jobs import (
    daily_digest_job,
    enqueue_agent_run,
    get_scheduler,
    housekeeping_job,
    run_daily_digest_now,
    scheduler_status,
    shutdown_scheduler,
    start_scheduler,
)

__all__ = [
    "DeliveryOutcome",
    "daily_digest_job",
    "deliver_digest",
    "enqueue_agent_run",
    "get_scheduler",
    "housekeeping_job",
    "render_digest_html",
    "render_digest_text",
    "run_daily_digest_now",
    "scheduler_status",
    "shutdown_scheduler",
    "start_scheduler",
]
