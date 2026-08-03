"""Shared FastAPI dependencies: DB sessions, current user, role guards, templates.

Two flavours of "current user" exist deliberately:

* :func:`get_current_user_optional` — never raises; pages render for guests.
* :func:`get_current_user` / :func:`require_admin` — raise ``401``/``403``, and
  for browser navigations redirect to the login page instead of returning JSON.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.flash import clear_flashes, read_flashes
from app.models.user import User, UserRole
from app.security import ACCESS_COOKIE_NAME, AuthError, extract_user_id

logger = logging.getLogger(__name__)
settings = get_settings()

TEMPLATES_DIR = "app/templates"

templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _register_template_globals() -> None:
    """Expose a few helpers/constants to every template."""
    templates.env.globals.update(
        app_name=settings.app_name,
        environment=settings.environment,
        mesh_configured=settings.mesh_configured,
        langsmith_enabled=bool(settings.langsmith_tracing and settings.langsmith_api_key),
    )
    templates.env.filters["money"] = lambda value: (
        "Free" if not value else f"${float(value):,.2f}"
    )


_register_template_globals()


def get_templates() -> Jinja2Templates:
    """Return the shared Jinja2 environment."""
    return templates


def _token_from_request(request: Request) -> Optional[str]:
    """Extract a JWT from the auth cookie or an ``Authorization`` header."""
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        candidate = header.split(" ", 1)[1].strip()
        if candidate:
            return candidate
    cookie = request.cookies.get(ACCESS_COOKIE_NAME)
    return cookie or None


def get_current_user_optional(
    request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """Resolve the signed-in user, or None for anonymous visitors.

    Used by public pages and by the event-tracking endpoint (which accepts
    anonymous traffic keyed by ``session_id``).
    """
    token = _token_from_request(request)
    if not token:
        return None

    try:
        user_id = extract_user_id(token)
    except AuthError as exc:
        logger.debug("Ignoring invalid access token: %s", exc)
        return None

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    request: Request, user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Require an authenticated user.

    Raises:
        HTTPException: ``401`` for API clients, or a ``303`` redirect to
            ``/login`` when the caller is a browser navigation.
    """
    if user is not None:
        return user

    if _wants_html(request):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            detail="Authentication required",
            headers={"Location": f"/login?next={request.url.path}"},
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_admin(
    request: Request, user: User = Depends(get_current_user)
) -> User:
    """Require an authenticated **admin**.

    Raises:
        HTTPException: ``403`` (or a redirect to the catalog for browsers).
    """
    if user.role == UserRole.ADMIN:
        return user

    logger.warning("User %s attempted to access an admin-only resource", user.id)
    if _wants_html(request):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            detail="Admin access required",
            headers={"Location": "/catalog"},
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
    )


def _wants_html(request: Request) -> bool:
    """Is this a browser page load rather than an API/fetch call?

    Routing is the primary signal: everything under ``/api/`` is JSON, everything
    else is a server-rendered page.  An explicit ``X-Requested-With`` header lets
    ``fetch`` calls against page routes opt out of redirects.  The ``Accept``
    header is only consulted as a tie-breaker, because clients are inconsistent
    about sending it (``*/*`` is extremely common).
    """
    if request.url.path.startswith("/api/"):
        return False
    if request.headers.get("x-requested-with", "").lower() == "xmlhttprequest":
        return False
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return False
    return True


def base_context(request: Request, user: Optional[User] = None, **extra: Any) -> dict[str, Any]:
    """Build the template context shared by every page.

    Args:
        request: The active request (required by Jinja2Templates).
        user: The signed-in user, if any.
        **extra: Page-specific values merged into the context.
    """
    context: dict[str, Any] = {
        "request": request,
        "current_user": user,
        "settings": settings,
    }
    context.update(extra)
    return context


def render_page(
    request: Request,
    template_name: str,
    user: Optional[User] = None,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    """Render a Jinja2 page, injecting the shared context and pending flashes.

    Flashes are read before rendering and their cookie cleared on the way out, so
    each message is displayed exactly once.

    Args:
        request: The active request.
        template_name: Template path relative to ``app/templates``.
        user: The signed-in user, if any.
        status_code: HTTP status for the response.
        **extra: Page-specific template variables.

    Returns:
        The rendered :class:`~fastapi.responses.HTMLResponse`.
    """
    messages = read_flashes(request)
    context = base_context(request, user, flashes=messages, **extra)
    response = templates.TemplateResponse(
        request=request, name=template_name, context=context, status_code=status_code
    )
    if messages:
        clear_flashes(response)
    return response
