"""Password hashing and JWT issuing/verification.

Auth model
----------
The app is server-rendered *and* exposes a JSON API, so a single JWT is used for
both:

* browsers receive it in an ``HttpOnly`` cookie (``smartreco_access``);
* API clients may send it as ``Authorization: Bearer <token>``.

The cookie is ``SameSite=Lax`` so top-level navigations keep the session while
cross-site POSTs cannot ride along, and ``Secure`` in production.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from jose import JWTError, jwt

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

#: Name of the cookie carrying the access token in browser sessions.
ACCESS_COOKIE_NAME = "smartreco_access"

#: bcrypt silently truncates beyond 72 bytes — we reject longer input instead.
MAX_PASSWORD_BYTES = 72

#: bcrypt cost factor. 12 is ~250ms on current hardware: slow enough to be
#: expensive to brute-force, fast enough for an interactive login.
BCRYPT_ROUNDS = 12

#: Iteration count for the PBKDF2 fallback (only used if bcrypt is unavailable).
PBKDF2_ITERATIONS = 480_000
PBKDF2_PREFIX = "pbkdf2_sha256"


class AuthError(Exception):
    """Raised when a token is missing, malformed or expired."""


# --------------------------------------------------------------------------- #
# Password hashing                                                            #
# --------------------------------------------------------------------------- #
#
# We call `bcrypt` directly rather than going through passlib. passlib 1.7.4 is
# unmaintained and its bcrypt backend probes the library with an 80-byte test
# secret, which bcrypt >= 4.1 rejects outright ("password cannot be longer than
# 72 bytes") — so `passlib.hash.bcrypt` raises on import-time backend detection
# against a modern bcrypt. Using the primitive directly removes that fragility
# and costs us nothing: the stored format is identical modular-crypt bcrypt, so
# existing hashes keep verifying.
#
# A PBKDF2-SHA256 fallback is retained for environments where the bcrypt C
# extension cannot be built; hashes are self-describing, so both schemes
# coexist and old hashes stay valid.


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with bcrypt.

    Args:
        plain_password: The user-supplied password.

    Returns:
        A modular-crypt-format hash safe to persist.

    Raises:
        ValueError: If the password exceeds bcrypt's 72-byte input limit.
    """
    encoded = plain_password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must not exceed {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
        )

    try:
        return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")
    except Exception:  # pragma: no cover - no bcrypt backend available
        logger.warning(
            "bcrypt hashing unavailable — falling back to PBKDF2-SHA256", exc_info=True
        )
        return _pbkdf2_hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its stored hash in constant time.

    Never raises: a corrupt, empty or unknown hash simply fails verification.

    Args:
        plain_password: The candidate password.
        hashed_password: The stored hash (bcrypt or PBKDF2).

    Returns:
        True only when the password matches.
    """
    if not hashed_password:
        return False

    try:
        if hashed_password.startswith(PBKDF2_PREFIX):
            return _pbkdf2_verify(plain_password, hashed_password)
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:MAX_PASSWORD_BYTES],
            hashed_password.encode("ascii"),
        )
    except Exception:
        logger.warning("Password verification failed for a malformed hash", exc_info=True)
        return False


@functools.lru_cache(maxsize=1)
def dummy_hash() -> str:
    """A valid throwaway hash used to equalise timing for unknown accounts.

    Verifying against this makes "no such user" cost roughly the same as "wrong
    password", so response timing does not disclose which emails are registered.
    Computed once and cached.
    """
    return hash_password("smartreco-timing-equaliser")


def _pbkdf2_hash(plain_password: str) -> str:
    """Hash with PBKDF2-SHA256, formatted as ``pbkdf2_sha256$iters$salt$hash``."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", plain_password.encode("utf-8"), salt.encode("ascii"), PBKDF2_ITERATIONS
    )
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def _pbkdf2_verify(plain_password: str, stored: str) -> bool:
    """Verify a PBKDF2-SHA256 hash produced by :func:`_pbkdf2_hash`."""
    try:
        _, iterations, salt, expected = stored.split("$", 3)
    except ValueError:
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256", plain_password.encode("utf-8"), salt.encode("ascii"), int(iterations)
    )
    return hmac.compare_digest(digest.hex(), expected)


def create_access_token(
    subject: str | int,
    *,
    role: str = "user",
    expires_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Mint a signed JWT.

    Args:
        subject: The user id (stored in the standard ``sub`` claim).
        role: The user's role, embedded to avoid a DB hit on cheap checks.
        expires_minutes: Override for the configured token lifetime.
        extra_claims: Additional non-reserved claims to embed.

    Returns:
        The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    lifetime = expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    payload: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=lifetime)).timestamp()),
        "iss": settings.app_name,
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT.

    Args:
        token: The encoded JWT.

    Returns:
        The decoded claim set.

    Raises:
        AuthError: If the signature is invalid or the token has expired.
    """
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise AuthError(f"Invalid or expired token: {exc}") from exc


def extract_user_id(token: str) -> int:
    """Return the integer user id encoded in ``sub``.

    Raises:
        AuthError: If the token is invalid or ``sub`` is not an integer.
    """
    claims = decode_access_token(token)
    subject = claims.get("sub")
    try:
        return int(str(subject))
    except (TypeError, ValueError) as exc:
        raise AuthError(f"Token subject is not a valid user id: {subject!r}") from exc
