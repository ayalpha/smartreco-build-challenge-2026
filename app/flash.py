"""Flash messages for the server-rendered UI.

FastAPI has no built-in flash mechanism, so this module implements the classic
Flask pattern with a short-lived, signed cookie:

* :func:`flash` attaches a message to the outgoing redirect response;
* :func:`consume_flashes` reads and clears them when the next page renders.

Messages are signed with ``SECRET_KEY`` (via :mod:`itsdangerous`) so a user
cannot forge or tamper with UI copy, and they expire after 60 seconds.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FLASH_COOKIE_NAME = "smartreco_flash"
FLASH_MAX_AGE_SECONDS = 60
FLASH_SALT = "smartreco-flash-v1"

#: Bootstrap-ish semantic categories mapped to Tailwind styles in ``base.html``.
FlashCategory = Literal["success", "error", "warning", "info"]

_serializer = URLSafeTimedSerializer(settings.secret_key, salt=FLASH_SALT)


def flash(response: Response, message: str, category: FlashCategory = "info") -> None:
    """Queue a flash message on ``response``.

    Multiple calls on the same response accumulate — the cookie holds a list.

    Args:
        response: The (usually redirect) response being returned.
        message: Human-readable text to show the user.
        category: Semantic style key (``success``/``error``/``warning``/``info``).
    """
    existing: list[dict[str, str]] = getattr(response, "_smartreco_flashes", [])
    existing.append({"message": message, "category": category})
    setattr(response, "_smartreco_flashes", existing)

    response.set_cookie(
        key=FLASH_COOKIE_NAME,
        value=_serializer.dumps(existing),
        max_age=FLASH_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def read_flashes(request: Request) -> list[dict[str, str]]:
    """Read pending flash messages without mutating any response.

    Rendering needs the messages *before* the response object exists, so reading
    and clearing are separate operations (:func:`clear_flashes` does the latter).

    Args:
        request: The incoming request carrying the flash cookie.

    Returns:
        A list of ``{"message", "category"}`` dicts — empty when there is
        nothing pending or the cookie failed verification.
    """
    raw = request.cookies.get(FLASH_COOKIE_NAME)
    if not raw:
        return []

    try:
        payload = _serializer.loads(raw, max_age=FLASH_MAX_AGE_SECONDS)
    except SignatureExpired:
        return []
    except (BadSignature, json.JSONDecodeError):
        logger.warning("Discarded a flash cookie that failed signature verification")
        return []

    if not isinstance(payload, list):
        return []

    return [
        {"message": str(item.get("message", "")), "category": str(item.get("category", "info"))}
        for item in payload
        if isinstance(item, dict) and item.get("message")
    ]


def clear_flashes(response: Response) -> None:
    """Delete the flash cookie so messages are shown exactly once."""
    response.delete_cookie(FLASH_COOKIE_NAME, path="/")


def consume_flashes(request: Request, response: Response) -> list[dict[str, str]]:
    """Read pending flashes *and* clear the cookie in one call."""
    messages = read_flashes(request)
    if messages:
        clear_flashes(response)
    return messages
