"""SQLAlchemy engine, session factory and declarative base.

Supports PostgreSQL (preferred) and SQLite (local-dev fallback) transparently:
the SQLite branch adds the ``check_same_thread`` flag required by the scheduler
threads and enables foreign-key enforcement, which SQLite disables by default.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model."""


def _engine_kwargs() -> dict[str, Any]:
    """Build driver-specific engine keyword arguments."""
    kwargs: dict[str, Any] = {
        "echo": settings.sql_echo,
        "future": True,
        "pool_pre_ping": True,
    }

    if settings.is_sqlite:
        # SQLite: allow cross-thread use (APScheduler runs jobs off-thread).
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in settings.database_url:
            # Keep one connection alive so an in-memory DB survives between
            # sessions (used by the test-suite).
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)

    return kwargs


engine: Engine = create_engine(settings.database_url, **_engine_kwargs())

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                            expire_on_commit=False, class_=Session)


if settings.is_sqlite:  # pragma: no branch - configuration dependent

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
        """Enable FK constraints + WAL journaling on every new SQLite connection."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception:  # pragma: no cover - exotic SQLite builds
            logger.debug("Could not apply SQLite PRAGMAs", exc_info=True)
        finally:
            cursor.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    The session is always closed, and rolled back if the request raised.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional context manager for non-request code (scheduler, scripts).

    Commits on clean exit, rolls back on exception, always closes.

    Example:
        >>> with session_scope() as db:  # doctest: +SKIP
        ...     db.add(User(email="a@b.c"))
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Rolling back session after unhandled exception")
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create every table declared on :class:`Base` if it does not yet exist.

    Alembic remains the source of truth for production schema changes; this is
    the convenience path for local dev, tests and the seed script.
    """
    # Import for side-effects: registers all mappers on ``Base.metadata``.
    from app import models  # noqa: F401  pylint: disable=unused-import

    logger.info("Ensuring database schema exists (%s)",
                "sqlite" if settings.is_sqlite else "postgresql")
    Base.metadata.create_all(bind=engine)


def healthcheck() -> bool:
    """Return True when a trivial query succeeds against the database."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("Database healthcheck failed", exc_info=True)
        return False
