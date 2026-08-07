"""Proactive digest delivery: HTML email + optional Telegram (BONUS 2).

Four interchangeable email backends, selected by ``EMAIL_BACKEND``:

``console``
    Renders and logs the message. The default, so a fresh clone delivers digests
    with zero credentials and the feature is demonstrable offline.
``sendgrid``
    Uses the SendGrid v3 API (``SENDGRID_API_KEY``).
``smtp``
    Plain ``smtplib`` with STARTTLS — works with a Gmail app password.
``resend``
    Uses the Resend HTTP API (``RESEND_API_KEY``).

Every attempt, successful or not, is recorded in ``email_digests`` so the
scheduler's behaviour is fully auditable.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import get_settings
from app.database import session_scope
from app.models.email_digest import EmailDigest
from app.models.recommendation import Recommendation
from app.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

_environment: Optional[Environment] = None


def _jinja() -> Environment:
    """Return a standalone Jinja environment for rendering email templates.

    Separate from the request-scoped FastAPI templates because the scheduler runs
    with no request context.
    """
    global _environment
    if _environment is None:
        _environment = Environment(
            loader=FileSystemLoader("app/templates"),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _environment.filters["money"] = lambda value: (
            "Free" if not value else f"${float(value):,.2f}"
        )
    return _environment


@dataclass
class DeliveryOutcome:
    """Result of one delivery attempt on one channel."""

    channel: str
    backend: str
    ok: bool
    subject: Optional[str] = None
    error: Optional[str] = None


def render_digest_html(
    user_name: str,
    recommendation: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    base_url: Optional[str] = None,
) -> str:
    """Render the HTML email body."""
    template = _jinja().get_template("emails/digest.html")
    return template.render(
        app_name=settings.app_name,
        user_name=user_name,
        recommendation=recommendation,
        products=products,
        base_url=(base_url or settings.base_url).rstrip("/"),
        signals=(recommendation.get("interest_signals") or [])[:3],
    )


def render_digest_text(
    user_name: str, recommendation: dict[str, Any], products: list[dict[str, Any]]
) -> str:
    """Render a plain-text alternative (also reused for Telegram)."""
    lines = [
        f"Hi {user_name},",
        "",
        str(recommendation.get("headline") or "Your picks for today"),
        "",
        str(recommendation.get("narrative") or ""),
        "",
        "Recommended for you:",
    ]
    for index, product in enumerate(products, start=1):
        price = product.get("price")
        price_text = "Free" if not price else f"${float(price):,.2f}"
        lines.append(f"{index}. {product.get('title')} — {price_text}")
        pitch = product.get("pitch") or product.get("reason")
        if pitch:
            lines.append(f"   {pitch}")
        lines.append(f"   {settings.base_url.rstrip('/')}/product/{product.get('id')}")
    lines.extend(["", f"— The {settings.app_name} team"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Email backends                                                              #
# --------------------------------------------------------------------------- #

def _send_console(to_email: str, subject: str, html: str, text: str) -> None:
    """Log the message instead of sending it (default dev backend)."""
    logger.info(
        "[email:console] to=%s subject=%r\n%s\n(html body: %d chars)",
        to_email, subject, text, len(html),
    )


def _send_sendgrid(to_email: str, subject: str, html: str, text: str) -> None:
    """Send via the SendGrid v3 API."""
    if not settings.sendgrid_api_key:
        raise RuntimeError("EMAIL_BACKEND=sendgrid but SENDGRID_API_KEY is not set")

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Content, Email, Mail, To

    message = Mail(
        from_email=Email(settings.digest_from_email, settings.digest_from_name),
        to_emails=To(to_email),
        subject=subject,
    )
    message.add_content(Content("text/plain", text))
    message.add_content(Content("text/html", html))

    client = SendGridAPIClient(settings.sendgrid_api_key)
    response = client.send(message)
    if int(response.status_code) >= 300:
        raise RuntimeError(f"SendGrid returned HTTP {response.status_code}")
    logger.info("[email:sendgrid] delivered to %s (HTTP %s)", to_email, response.status_code)


def _send_smtp(to_email: str, subject: str, html: str, text: str) -> None:
    """Send via SMTP with STARTTLS."""
    if not settings.smtp_username or not settings.smtp_password:
        raise RuntimeError("EMAIL_BACKEND=smtp but SMTP_USERNAME/SMTP_PASSWORD are not set")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{settings.digest_from_name} <{settings.digest_from_email}>"
    message["To"] = to_email
    message.set_content(text)
    message.add_alternative(html, subtype="html")

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        server.ehlo()
        if settings.smtp_use_tls:
            server.starttls()
            server.ehlo()
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(message)

    logger.info("[email:smtp] delivered to %s via %s", to_email, settings.smtp_host)


def _send_resend(to_email: str, subject: str, html: str, text: str) -> None:
    """Send via the Resend HTTP API.

    Raises:
        RuntimeError: If the API key is missing or Resend rejects the request.
    """
    if not settings.resend_api_key:
        raise RuntimeError("EMAIL_BACKEND=resend but RESEND_API_KEY is not set")

    import httpx

    payload = {
        "from": f"{settings.digest_from_name} <{settings.digest_from_email}>",
        "to": [to_email],
        "subject": subject,
        "html": html,
        "text": text,
    }

    headers = {
        "Authorization": f"Bearer {settings.resend_api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.post(
        "https://api.resend.com/emails",
        headers=headers,
        json=payload,
        timeout=20.0,
    )

    if response.status_code >= 300:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text
        raise RuntimeError(
            f"Resend returned HTTP {response.status_code}: {error_body}"
        )

    logger.info("[email:resend] delivered to %s (HTTP %s)", to_email, response.status_code)


def send_email(to_email: str, subject: str, html: str, text: str) -> DeliveryOutcome:
    """Send an email through the configured backend."""
    backend = settings.email_backend
    try:
        if not settings.email_enabled:
            return DeliveryOutcome(
                channel="email",
                backend=backend,
                ok=False,
                subject=subject,
                error="email delivery is disabled (EMAIL_ENABLED=false)",
            )

        if backend == "sendgrid":
            _send_sendgrid(to_email, subject, html, text)
        elif backend == "resend":
            _send_resend(to_email, subject, html, text)
        elif backend == "smtp":
            _send_smtp(to_email, subject, html, text)
        else:
            _send_console(to_email, subject, html, text)

        return DeliveryOutcome(channel="email", backend=backend, ok=True, subject=subject)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Email delivery to %s failed via %s", to_email, backend)
        return DeliveryOutcome(
            channel="email",
            backend=backend,
            ok=False,
            subject=subject,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
