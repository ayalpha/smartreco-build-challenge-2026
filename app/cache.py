"""Cache / distributed-lock layer with a transparent in-memory fallback.

Redis is used for two things:

1. **Trigger locks** — a ``SET NX EX`` lock keyed by user id prevents two
   concurrent agent runs for the same user (the classic duplicate-work bug when
   a burst of events arrives).
2. **Result cache** — the serialised active recommendation, so the 60-second
   frontend poll does not hit Postgres for every viewer.

If ``REDIS_URL`` is unset or Redis is unreachable, everything degrades to a
process-local dict with the same semantics (including TTL expiry).  That keeps
the app runnable with zero infrastructure while remaining correct in a
multi-process deployment when Redis *is* present.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheBackend(Protocol):
    """Minimal interface both backends implement."""

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def acquire_lock(self, key: str, ttl: int) -> bool: ...
    def release_lock(self, key: str) -> None: ...
    @property
    def name(self) -> str: ...


class InMemoryCache:
    """Thread-safe dict cache with TTL support (single-process fallback)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, Optional[float]]] = {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "memory"

    def _expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and expires_at <= time.monotonic()

    def get(self, key: str) -> Optional[str]:
        """Return the cached value, or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if self._expired(expires_at):
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store a value with an optional TTL in seconds."""
        with self._lock:
            expires_at = time.monotonic() + ttl if ttl else None
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Remove a key (no-op when absent)."""
        with self._lock:
            self._store.pop(key, None)

    def acquire_lock(self, key: str, ttl: int) -> bool:
        """Atomically acquire a lock; False if already held."""
        with self._lock:
            if self.get(key) is not None:
                return False
            self.set(key, "1", ttl=ttl)
            return True

    def release_lock(self, key: str) -> None:
        """Release a previously acquired lock."""
        self.delete(key)


class RedisCache:
    """Redis-backed implementation of :class:`CacheBackend`."""

    def __init__(self, url: str) -> None:
        import redis  # imported lazily so redis is optional at runtime

        self._client = redis.Redis.from_url(
            url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
        # Fail fast: the caller decides whether to fall back to memory.
        self._client.ping()

    @property
    def name(self) -> str:
        return "redis"

    def get(self, key: str) -> Optional[str]:
        """Return the cached value, or None on miss or transport error."""
        try:
            value = self._client.get(key)
            return str(value) if value is not None else None
        except Exception:
            logger.warning("Redis GET failed for %s", key, exc_info=True)
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store a value with an optional TTL in seconds."""
        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
        except Exception:
            logger.warning("Redis SET failed for %s", key, exc_info=True)

    def delete(self, key: str) -> None:
        """Remove a key (no-op when absent)."""
        try:
            self._client.delete(key)
        except Exception:
            logger.warning("Redis DEL failed for %s", key, exc_info=True)

    def acquire_lock(self, key: str, ttl: int) -> bool:
        """Atomic ``SET NX EX`` lock. Fails *open* so Redis outages never block work."""
        try:
            return bool(self._client.set(key, "1", nx=True, ex=ttl))
        except Exception:
            logger.warning("Redis lock acquisition failed for %s", key, exc_info=True)
            return True

    def release_lock(self, key: str) -> None:
        """Release a previously acquired lock."""
        self.delete(key)


_backend: Optional[CacheBackend] = None
_backend_lock = threading.Lock()


def get_cache() -> CacheBackend:
    """Return the process-wide cache backend, connecting to Redis on first use."""
    global _backend
    if _backend is not None:
        return _backend

    with _backend_lock:
        if _backend is not None:  # pragma: no cover - race guard
            return _backend
        if settings.redis_url:
            try:
                _backend = RedisCache(settings.redis_url)
                logger.info("Cache backend: Redis (%s)", settings.redis_url)
                return _backend
            except Exception as exc:
                logger.warning(
                    "Redis unavailable (%s) — falling back to in-memory cache.", exc
                )
        else:
            logger.info("REDIS_URL not set — using in-memory cache.")
        _backend = InMemoryCache()
        return _backend


def reset_cache_backend() -> None:
    """Drop the cached backend (used by tests to force re-selection)."""
    global _backend
    with _backend_lock:
        _backend = None


# --------------------------------------------------------------------------- #
# Domain-specific helpers                                                     #
# --------------------------------------------------------------------------- #

def _lock_key(user_id: int) -> str:
    return f"smartreco:agent:lock:{user_id}"


def _pending_key(user_id: int) -> str:
    return f"smartreco:agent:pending:{user_id}"


def _reco_key(user_id: int) -> str:
    return f"smartreco:reco:active:{user_id}"


def acquire_agent_lock(user_id: int, ttl: Optional[int] = None) -> bool:
    """Try to claim the right to run the agent for ``user_id``.

    Returns:
        True if the caller owns the lock and should proceed; False if another
        worker is already generating a recommendation for this user.
    """
    ttl = ttl or settings.agent_lock_ttl_seconds
    acquired = get_cache().acquire_lock(_lock_key(user_id), ttl=ttl)
    if not acquired:
        logger.info("Agent lock already held for user=%s — skipping duplicate run", user_id)
    return acquired


def release_agent_lock(user_id: int) -> None:
    """Release the agent lock for ``user_id``."""
    get_cache().release_lock(_lock_key(user_id))


def mark_agent_pending(user_id: int, reason: str, ttl: int = 600) -> None:
    """Flag that a run is queued so the UI can show a generating state."""
    get_cache().set(_pending_key(user_id), reason, ttl=ttl)


def get_agent_pending(user_id: int) -> Optional[str]:
    """Return the queued trigger reason, if a run is in flight."""
    return get_cache().get(_pending_key(user_id))


def clear_agent_pending(user_id: int) -> None:
    """Clear the queued-run flag."""
    get_cache().delete(_pending_key(user_id))


def cache_active_recommendation(user_id: int, payload: dict[str, Any], ttl: int = 120) -> None:
    """Cache the serialised active recommendation for the polling endpoint."""
    try:
        get_cache().set(_reco_key(user_id), json.dumps(payload, default=str), ttl=ttl)
    except (TypeError, ValueError):
        logger.warning("Recommendation payload for user=%s is not serialisable", user_id,
                       exc_info=True)


def get_cached_recommendation(user_id: int) -> Optional[dict[str, Any]]:
    """Return the cached recommendation payload, or None on miss."""
    raw = get_cache().get(_reco_key(user_id))
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        logger.warning("Discarding corrupt cached recommendation for user=%s", user_id)
        get_cache().delete(_reco_key(user_id))
        return None


def invalidate_recommendation_cache(user_id: int) -> None:
    """Drop the cached recommendation (called by ``recommendation_storer``)."""
    get_cache().delete(_reco_key(user_id))
