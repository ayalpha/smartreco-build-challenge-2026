"""Configuration validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_production_rejects_default_secret_key() -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY must be set"):
        Settings(
            environment="production",
            debug=False,
            secret_key="dev-only-insecure-secret-key-change-me-please-32",
        )


def test_production_rejects_debug_mode() -> None:
    with pytest.raises(ValidationError, match="DEBUG must be false"):
        Settings(environment="production", secret_key="a-unique-production-secret")


def test_production_accepts_hardened_settings() -> None:
    settings = Settings(
        environment="production",
        debug=False,
        secret_key="a-unique-production-secret",
    )

    assert settings.is_production is True
