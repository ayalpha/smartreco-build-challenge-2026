"""Application configuration.

All runtime configuration is loaded from environment variables (or a local
``.env`` file) into a single immutable :class:`Settings` object.  Nothing in this
codebase reads ``os.environ`` directly except :func:`Settings` itself and the
Mesh client factory (which needs the raw key), so there is exactly one place to
look when auditing configuration.

Secrets are never logged: :meth:`Settings.safe_dump` redacts every field whose
name looks like a credential.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

#: Substrings that mark a settings field as secret for :meth:`Settings.safe_dump`.
_SECRET_HINTS = ("key", "token", "password", "secret", "dsn")


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Every field maps 1:1 to an environment variable of the same (upper-cased)
    name.  See ``.env.example`` for a documented template.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Nexora"
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    base_url: str = "http://localhost:8000"

    # ------------------------------------------------------- Mesh API (LLM)
    # MANDATORY: every LLM / embedding call in this project goes through Mesh.
    mesh_api_key: str = ""
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    mesh_model_reasoning: str = "openai/gpt-4o"
    mesh_model_writer: str = "anthropic/claude-3-5-sonnet"
    mesh_model_grader: str = "openai/gpt-4o-mini"
    mesh_embedding_model: str = "openai/text-embedding-3-small"
    mesh_embedding_dim: int = 1536
    mesh_max_retries: int = 3
    mesh_timeout_seconds: float = 60.0

    # ----------------------------------------------------------- hackathon
    submission_token: str = ""

    # ------------------------------------------------------------ database
    database_url: str = "sqlite:///./smartreco.db"
    sql_echo: bool = False

    # ----------------------------------------------------------- vector db
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "smartreco_products"
    vector_search_top_k: int = 12

    # ---------------------------------------------------------------- auth
    secret_key: str = "dev-only-insecure-secret-key-change-me-please-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080  # 7 days

    # --------------------------------------------------------------- agent
    agent_event_trigger_interval: int = 10
    agent_stale_hours: int = 2
    agent_stale_min_events: int = 5
    agent_max_retrieval_retries: int = 2
    agent_min_relevant_products: int = 3
    agent_recent_event_window: int = 60
    agent_final_product_count: int = 6
    agent_lock_ttl_seconds: int = 300

    # ----------------------------------------------------------- langsmith
    langsmith_tracing: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "smartreco-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --------------------------------------------------------------- email
    email_enabled: bool = True
    email_backend: Literal["console", "sendgrid", "smtp"] = "console"
    sendgrid_api_key: Optional[str] = None
    digest_from_email: str = "noreply@nexora.ai"
    digest_from_name: str = "Nexora"
    digest_schedule_hour: int = Field(default=18, ge=0, le=23)
    digest_schedule_minute: int = Field(default=0, ge=0, le=59)
    digest_min_events_today: int = 5
    digest_product_count: int = 3

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True

    # ------------------------------------------------------------ telegram
    telegram_bot_token: Optional[str] = None

    # --------------------------------------------------------------- redis
    redis_url: Optional[str] = "redis://localhost:6379/0"

    # ----------------------------------------------------------- scheduler
    scheduler_enabled: bool = True
    scheduler_timezone: str = "UTC"

    # ------------------------------------------------------------ validators
    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Upper-case the log level and reject unknown names early."""
        candidate = value.upper().strip()
        if candidate not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"Unsupported LOG_LEVEL: {value!r}")
        return candidate

    @field_validator("qdrant_api_key", "telegram_bot_token", "sendgrid_api_key",
                     "smtp_username", "smtp_password", "langsmith_api_key", "redis_url",
                     mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        """Treat an empty env var (``FOO=``) as "not configured"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalise_database_url(cls, value: Any) -> Any:
        """Pin an explicit SQLAlchemy driver onto PostgreSQL DSNs.

        Managed platforms hand out a bare DSN whose scheme SQLAlchemy 2.0 either
        rejects or resolves ambiguously:

        * Heroku-style providers emit ``postgres://``, which SQLAlchemy 2.0
          removed support for entirely — it raises ``NoSuchModuleError`` at
          engine creation, so the app would not boot at all.
        * Railway, Render and friends emit ``postgresql://``, which works but
          leaves the DBAPI implicit.

        Both are rewritten to ``postgresql+psycopg2://`` so the driver is
        unambiguous and injected DSNs work untouched. SQLite and any DSN that
        already names a driver are passed through unchanged.
        """
        if not isinstance(value, str):
            return value

        dsn = value.strip()
        if dsn.startswith("postgres://"):
            return "postgresql+psycopg2://" + dsn[len("postgres://"):]
        if dsn.startswith("postgresql://"):
            return "postgresql+psycopg2://" + dsn[len("postgresql://"):]
        return dsn

    @model_validator(mode="after")
    def _infer_platform_base_url(self) -> "Settings":
        """Adopt the hosting platform's public URL when ``BASE_URL`` is unset.

        ``BASE_URL`` only matters for absolute links in digest emails and Telegram
        messages, and on a PaaS the correct value is not knowable until the
        service has been created — which makes it exactly the kind of variable
        people forget to set, leaving ``localhost:8000`` links in a live email.

        Render injects ``RENDER_EXTERNAL_URL`` (a full URL) and Railway injects
        ``RAILWAY_PUBLIC_DOMAIN`` (a bare hostname), so both are picked up here.
        An explicitly configured ``BASE_URL`` always wins.
        """
        if "base_url" in self.model_fields_set:
            return self

        external = (os.environ.get("RENDER_EXTERNAL_URL") or "").strip()
        if not external:
            domain = (os.environ.get("RAILWAY_PUBLIC_DOMAIN") or "").strip()
            if domain:
                external = f"https://{domain}"

        if external:
            self.base_url = external.rstrip("/")
            logger.info("Detected platform public URL — BASE_URL=%s", self.base_url)

        return self

    # -------------------------------------------------------- derived props
    @property
    def is_sqlite(self) -> bool:
        """True when the configured database is SQLite (local-dev fallback)."""
        return self.database_url.startswith("sqlite")

    @property
    def is_production(self) -> bool:
        """True in production — used to harden cookies and disable debug output."""
        return self.environment == "production"

    @property
    def mesh_configured(self) -> bool:
        """True when a Mesh API key is present, so AI features can run for real."""
        return bool(self.mesh_api_key)

    @property
    def cookie_secure(self) -> bool:
        """Send auth cookies over HTTPS only outside of local development."""
        return self.is_production

    def safe_dump(self) -> dict[str, Any]:
        """Return all settings with secret-looking values redacted.

        Safe to log at start-up — used by :func:`app.main.create_app`.
        """
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if value and any(hint in name for hint in _SECRET_HINTS):
                out[name] = "***redacted***"
            else:
                out[name] = value
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton (cached).

    Using an LRU cache means the ``.env`` file is parsed exactly once and the
    same object is shared by FastAPI dependencies, the scheduler and the agent.
    """
    settings = Settings()
    if not settings.mesh_configured:
        logger.warning(
            "MESH_API_KEY is not set — AI features will run in degraded "
            "(heuristic) mode. Set MESH_API_KEY to enable live Mesh calls."
        )
    return settings


#: Convenience module-level singleton for non-DI call sites (scheduler, scripts).
settings: Settings = get_settings()
