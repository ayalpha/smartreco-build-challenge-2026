"""Centralised logging setup.

The whole project logs through :mod:`logging` — there is not a single ``print``
in application code.  :func:`configure_logging` is idempotent so it can safely
be called from the app factory, the scheduler and stand-alone scripts.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Optional

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Third-party loggers that are far too chatty at DEBUG level.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai._base_client",
    "urllib3",
    "qdrant_client",
    "apscheduler.executors.default",
    "apscheduler.scheduler",
    "multipart",
    "python_multipart",
)


def configure_logging(level: Optional[str] = None) -> None:
    """Install a single stdout handler with a consistent format.

    Args:
        level: Root log level name (``"INFO"``, ``"DEBUG"`` …).  When omitted the
            value of ``LOG_LEVEL`` from settings is used.
    """
    global _CONFIGURED

    if level is None:
        # Imported lazily to avoid a circular import at module load time.
        from app.config import get_settings

        level = get_settings().log_level

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    if _CONFIGURED:
        logging.getLogger().setLevel(numeric_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured at level %s", level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
