"""Authentication schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.security import MAX_PASSWORD_BYTES

#: Minimum password length enforced at registration.
MIN_PASSWORD_LENGTH = 8


class RegisterRequest(BaseModel):
    """Payload for creating an account."""

    email: EmailStr = Field(..., description="Unique login email address")
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=160)

    @field_validator("password")
    @classmethod
    def _bcrypt_safe(cls, value: str) -> str:
        """Reject passwords bcrypt would silently truncate."""
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password must not exceed {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded"
            )
        return value

    @field_validator("full_name")
    @classmethod
    def _strip_name(cls, value: Optional[str]) -> Optional[str]:
        """Trim whitespace and normalise blanks to None."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class LoginRequest(BaseModel):
    """Payload for exchanging credentials for a token."""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """Issued access token."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class UserOut(BaseModel):
    """Public projection of a user."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    is_active: bool
    digest_opt_in: bool
    preferred_skill_level: Optional[str] = None
    created_at: Optional[datetime] = None

    @field_validator("role", mode="before")
    @classmethod
    def _role_to_str(cls, value: object) -> str:
        """Accept both the ``UserRole`` enum and a plain string."""
        return getattr(value, "value", str(value))
