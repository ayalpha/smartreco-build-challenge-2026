"""Authentication: HTML forms for the browser and JSON endpoints for API clients.

Both paths issue the same JWT.  Browser flows additionally set it as an
``HttpOnly`` cookie and use flash messages for feedback; JSON flows return the
token in the body.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user, render_page
from app.flash import flash
from app.models.user import User, UserRole
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.security import (
    ACCESS_COOKIE_NAME,
    create_access_token,
    dummy_hash,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["auth"])


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _set_auth_cookie(response: Response, token: str) -> None:
    """Attach the JWT to a response as a hardened cookie."""
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=token,
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def _safe_next(target: Optional[str]) -> str:
    """Validate a ``?next=`` redirect target to prevent open redirects."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


def _authenticate(db: Session, email: str, password: str) -> Optional[User]:
    """Return the user when credentials are valid, else None."""
    user = db.scalars(select(User).where(User.email == email.lower().strip())).first()
    if user is None:
        # Verify against a throwaway hash so a missing account costs roughly the
        # same as a wrong password — response timing must not disclose which
        # email addresses are registered.
        verify_password(password, dummy_hash())
        logger.info("Login failed: no account for %r", email)
        return None
    if not user.is_active:
        logger.info("Login failed: account %s is deactivated", user.id)
        return None
    if not verify_password(password, user.hashed_password):
        logger.info("Login failed: bad password for user %s", user.id)
        return None
    return user


def _create_user(db: Session, payload: RegisterRequest) -> User:
    """Create an account, promoting the very first user to admin.

    Bootstrapping the first account as admin means a fresh deployment has a way
    into the catalog UI without a separate CLI step.

    Raises:
        ValueError: If the email address is already registered.
    """
    email = str(payload.email).lower().strip()
    existing = db.scalars(select(User).where(User.email == email)).first()
    if existing is not None:
        raise ValueError("That email address is already registered.")

    is_first_user = db.scalars(select(User.id).limit(1)).first() is None
    user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=UserRole.ADMIN if is_first_user else UserRole.USER,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # concurrent duplicate registration
        db.rollback()
        raise ValueError("That email address is already registered.") from exc
    db.refresh(user)

    logger.info("Registered user id=%s email=%s role=%s", user.id, user.email, user.role.value)
    return user


def _token_for(user: User) -> str:
    """Mint an access token for a user."""
    return create_access_token(user.id, role=user.role.value)


# --------------------------------------------------------------------------- #
# HTML pages                                                                  #
# --------------------------------------------------------------------------- #

@router.get("/login", include_in_schema=False)
def login_page(request: Request, next: str = "/") -> Response:
    """Render the login form."""
    return render_page(request, "auth/login.html", next_url=_safe_next(next))


@router.post("/login", include_in_schema=False)
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    db: Session = Depends(get_db),
) -> Response:
    """Handle the login form: set the auth cookie and redirect."""
    user = _authenticate(db, email, password)
    if user is None:
        response = RedirectResponse(
            url=f"/login?next={_safe_next(next)}", status_code=status.HTTP_303_SEE_OTHER
        )
        flash(response, "Incorrect email or password. Please try again.", "error")
        return response

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    response = RedirectResponse(url=_safe_next(next), status_code=status.HTTP_303_SEE_OTHER)
    _set_auth_cookie(response, _token_for(user))
    flash(response, f"Welcome back, {user.display_name}.", "success")
    return response


@router.get("/register", include_in_schema=False)
def register_page(request: Request) -> Response:
    """Render the registration form."""
    return render_page(request, "auth/register.html")


@router.post("/register", include_in_schema=False)
def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
    db: Session = Depends(get_db),
) -> Response:
    """Handle the registration form, then sign the new user straight in."""
    try:
        payload = RegisterRequest(
            email=email, password=password, full_name=full_name or None
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        message = f"{'.'.join(str(p) for p in first.get('loc', []))}: {first.get('msg')}"
        response = RedirectResponse(url="/register", status_code=status.HTTP_303_SEE_OTHER)
        flash(response, f"Could not create your account — {message}", "error")
        return response

    try:
        user = _create_user(db, payload)
    except ValueError as exc:
        response = RedirectResponse(url="/register", status_code=status.HTTP_303_SEE_OTHER)
        flash(response, str(exc), "error")
        return response

    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    _set_auth_cookie(response, _token_for(user))
    flash(
        response,
        f"Account created — welcome to {settings.app_name}, {user.display_name}."
        + (" You are the first user, so you have admin access." if user.is_admin else ""),
        "success",
    )
    return response


@router.post("/logout", include_in_schema=False)
@router.get("/logout", include_in_schema=False)
def logout() -> Response:
    """Clear the auth cookie and return to the homepage."""
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(ACCESS_COOKIE_NAME, path="/")
    flash(response, "You have been signed out.", "info")
    return response


# --------------------------------------------------------------------------- #
# JSON API                                                                    #
# --------------------------------------------------------------------------- #

api_router = APIRouter(prefix="/api/auth", tags=["auth"])


@api_router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def api_register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Create an account and return an access token."""
    try:
        user = _create_user(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return TokenResponse(
        access_token=_token_for(user),
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@api_router.post("/login", response_model=TokenResponse)
def api_login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Exchange credentials for an access token."""
    user = _authenticate(db, str(payload.email), payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    return TokenResponse(
        access_token=_token_for(user),
        expires_in_minutes=settings.access_token_expire_minutes,
    )


@api_router.get("/me", response_model=UserOut)
def api_me(user: User = Depends(get_current_user)) -> User:
    """Return the authenticated user's profile."""
    return user
