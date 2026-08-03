"""Proactive digest delivery: HTML email + optional Telegram (BONUS 2).

Three interchangeable email backends, selected by ``EMAIL_BACKEND``:

``console``
    Renders and logs the message. The default, so a fresh clone delivers digests
    with zero credentials and the feature is demonstrable offline.
``sendgrid``
    Uses the SendGrid v3 API (``SENDGRID_API_KEY``).
``smtp``
    Plain ``smtplib`` with STARTTLS — works with a Gmail app password.

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
    """Render the HTML email body.

    Args:
        user_name: Greeting name.
        recommendation: Serialised recommendation (``to_dict()``).
        products: The products to feature (already trimmed to N).
        base_url: Absolute base for links.  Defaults to ``BASE_URL``.

    Returns:
        The complete HTML document.
    """
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
    """Send via the SendGrid v3 API.

    Raises:
        RuntimeError: If the API key is missing or SendGrid rejects the request.
    """
    if not settings.sendgrid_api_key:
        raise RuntimeError("EMAIL_BACKEND=sendgrid but SENDGRID_API_KEY is not set")

    from sendgrid import SendGridAPIClient  # imported lazily: optional dependency
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
    """Send via SMTP with STARTTLS.

    Raises:
        RuntimeError: If SMTP credentials are missing.
    """
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


def send_email(to_email: str, subject: str, html: str, text: str) -> DeliveryOutcome:
    """Send an email through the configured backend.

    Returns:
        A :class:`DeliveryOutcome` — never raises, so one bad address cannot abort
        the nightly job for everybody else.
    """
    backend = settings.email_backend
    try:
        if not settings.email_enabled:
            return DeliveryOutcome(
                channel="email", backend=backend, ok=False, subject=subject,
                error="email delivery is disabled (EMAIL_ENABLED=false)",
            )
        if backend == "sendgrid":
            _send_sendgrid(to_email, subject, html, text)
        elif backend == "smtp":
            _send_smtp(to_email, subject, html, text)
        else:
            _send_console(to_email, subject, html, text)
        return DeliveryOutcome(channel="email", backend=backend, ok=True, subject=subject)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Email delivery to %s failed via %s", to_email, backend)
        return DeliveryOutcome(
            channel="email", backend=backend, ok=False, subject=subject,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )


def send_telegram(chat_id: str, text: str) -> DeliveryOutcome:
    """Send a digest to Telegram (BONUS 2b — enabled by ``TELEGRAM_BOT_TOKEN``).

    Returns:
        A :class:`DeliveryOutcome`; skipped cleanly when no token is configured.
    """
    if not settings.telegram_bot_token:
        return DeliveryOutcome(
            channel="telegram", backend="telegram", ok=False,
            error="TELEGRAM_BOT_TOKEN is not configured",
        )

    try:
        import httpx

        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        response = httpx.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text[:4000],  # Telegram hard-limits messages to 4096 chars
                "disable_web_page_preview": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        logger.info("[telegram] delivered to chat %s", chat_id)
        return DeliveryOutcome(channel="telegram", backend="telegram", ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram delivery to chat %s failed", chat_id)
        return DeliveryOutcome(
            channel="telegram", backend="telegram", ok=False,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def deliver_digest(user_id: int, recommendation_id: int) -> list[DeliveryOutcome]:
    """Render and deliver a digest for one user, recording every attempt.

    Args:
        user_id: Recipient.
        recommendation_id: The recommendation to feature.

    Returns:
        One :class:`DeliveryOutcome` per attempted channel.
    """
    outcomes: list[DeliveryOutcome] = []

    with session_scope() as db:
        user = db.get(User, user_id)
        recommendation = db.get(Recommendation, recommendation_id)

        if user is None or recommendation is None:
            logger.warning(
                "Cannot deliver digest: user=%s recommendation=%s not found",
                user_id, recommendation_id,
            )
            return outcomes

        payload = recommendation.to_dict()
        products = (payload.get("products") or [])[: settings.digest_product_count]
        subject = (
            payload.get("headline")
            or f"Your {settings.app_name} picks for today"
        )[:200]

        html = render_digest_html(user.display_name, payload, products)
        text = render_digest_text(user.display_name, payload, products)
        telegram_chat_id = user.telegram_chat_id
        recipient = user.email

        email_outcome = send_email(recipient, subject, html, text)
        outcomes.append(email_outcome)

        if telegram_chat_id:
            outcomes.append(send_telegram(telegram_chat_id, text))

        for outcome in outcomes:
            db.add(
                EmailDigest(
                    user_id=user_id,
                    recommendation_id=recommendation_id,
                    channel=outcome.channel,
                    backend=outcome.backend,
                    status="sent" if outcome.ok else "failed",
                    subject=outcome.subject or subject,
                    error=outcome.error,
                )
            )

    return outcomes
