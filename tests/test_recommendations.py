"""Tests for the recommendation API, pages and proactive digest delivery."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.runner import run_agent
from app.cache import get_cached_recommendation, mark_agent_pending
from app.config import get_settings
from app.models.email_digest import EmailDigest
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User

settings = get_settings()

LATEST = "/api/recommendations/latest"


class TestAuthorisation:
    """Recommendation endpoints are private."""

    def test_latest_requires_authentication(self, client: TestClient) -> None:
        response = client.get(LATEST)
        assert response.status_code == 401

    def test_refresh_requires_authentication(self, client: TestClient) -> None:
        response = client.post("/api/recommendations/refresh")
        assert response.status_code == 401

    def test_cannot_read_another_users_recommendation(
        self, client: TestClient, user: User, make_user: Any,
        products: list[Product], make_events: Any,
    ) -> None:
        from app.security import create_access_token

        make_events(user, products)
        result = run_agent(user.id, reason="manual")

        intruder = make_user(email="intruder@test.dev")
        headers = {"Authorization": f"Bearer {create_access_token(intruder.id, role='user')}"}

        response = client.get(
            f"/api/recommendations/{result.recommendation_id}", headers=headers
        )
        assert response.status_code == 404, "must not leak another user's recommendation"


class TestLatestEndpoint:
    """The endpoint the 60-second poller calls."""

    def test_reports_no_recommendation_initially(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.get(LATEST, headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["has_recommendation"] is False
        assert body["recommendation"] is None
        assert body["generating"] is False
        assert body["next_trigger"] is not None

    def test_returns_the_active_recommendation(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)
        result = run_agent(user.id, reason="manual")

        response = client.get(LATEST, headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["has_recommendation"] is True

        payload = body["recommendation"]
        assert payload["id"] == result.recommendation_id
        assert payload["narrative"]
        assert payload["products"]
        assert payload["interest_signals"]
        assert all("pitch" in product for product in payload["products"])

    def test_generating_flag_drives_the_skeleton(
        self, client: TestClient, user: User, auth_headers: dict[str, str]
    ) -> None:
        mark_agent_pending(user.id, "event_threshold")

        body = client.get(LATEST, headers=auth_headers).json()

        assert body["generating"] is True
        assert body["pending_reason"] == "event_threshold"

    def test_response_is_cached_for_the_poller(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)
        run_agent(user.id, reason="manual")

        # Priming call populates the cache…
        client.get(LATEST, headers=auth_headers)
        assert get_cached_recommendation(user.id) is not None

        # …and the poller's variant is served from it.
        body = client.get(f"{LATEST}?include_trigger=false", headers=auth_headers).json()
        assert body["served_from_cache"] is True
        assert body["recommendation"]["narrative"]


class TestRefreshEndpoint:
    """The manual "Refresh my picks" action."""

    def test_refresh_generates_a_recommendation(
        self, client: TestClient, db: Session, user: User,
        auth_headers: dict[str, str], products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)

        response = client.post("/api/recommendations/refresh", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["has_recommendation"] is True
        assert body["generating"] is False
        assert body["recommendation"]["trigger_reason"] == "manual"

        stored = int(
            db.scalar(
                select(func.count()).select_from(Recommendation)
                .where(Recommendation.user_id == user.id)
            )
            or 0
        )
        assert stored == 1

    def test_refresh_works_with_no_prior_events(
        self, client: TestClient, auth_headers: dict[str, str], products: list[Product]
    ) -> None:
        response = client.post("/api/recommendations/refresh", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["has_recommendation"] is True

    def test_refresh_conflicts_while_a_run_is_in_flight(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        from app.cache import acquire_agent_lock, release_agent_lock

        assert acquire_agent_lock(user.id) is True
        try:
            response = client.post("/api/recommendations/refresh", headers=auth_headers)
            assert response.status_code == 409
        finally:
            release_agent_lock(user.id)


class TestHistoryAndDiagnostics:
    """History and the trigger-explanation endpoint."""

    def test_history_is_newest_first(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)
        first = run_agent(user.id, reason="manual")
        second = run_agent(user.id, reason="manual")

        response = client.get("/api/recommendations/history?limit=5", headers=auth_headers)

        assert response.status_code == 200
        items = response.json()
        assert len(items) == 2
        assert items[0]["id"] == second.recommendation_id
        assert items[1]["id"] == first.recommendation_id

    def test_trigger_endpoint_explains_itself(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.get("/api/recommendations/trigger", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["should_run"] is False
        assert body["reason"] == "none"
        assert body["detail"]


class TestPages:
    """Server-rendered surfaces."""

    def test_homepage_renders_for_anonymous_visitors(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Nexora" in response.text
        assert "Sign in to see your personalised panel" in response.text

    def test_homepage_shows_the_recommendation_when_present(
        self, client: TestClient, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.security import ACCESS_COOKIE_NAME, create_access_token

        make_events(user, products)
        run_agent(user.id, reason="manual")

        client.cookies.set(ACCESS_COOKIE_NAME, create_access_token(user.id, role="user"))
        response = client.get("/")

        assert response.status_code == 200
        assert "Why this recommendation?" in response.text
        assert "Refresh my picks" in response.text

    def test_profile_requires_login(self, client: TestClient) -> None:
        response = client.get("/profile", follow_redirects=False)
        assert response.status_code in (302, 303)
        assert "/login" in response.headers["location"]

    def test_profile_renders_history(
        self, client: TestClient, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.security import ACCESS_COOKIE_NAME, create_access_token

        make_events(user, products)
        run_agent(user.id, reason="manual")

        client.cookies.set(ACCESS_COOKIE_NAME, create_access_token(user.id, role="user"))
        response = client.get("/profile")

        assert response.status_code == 200
        assert "Recommendation history" in response.text
        assert "Agent trigger status" in response.text

    def test_architecture_page_documents_the_graph(self, client: TestClient) -> None:
        response = client.get("/architecture")
        assert response.status_code == 200
        for node in ("activity_analyzer", "relevance_grader", "persuasion_writer"):
            assert node in response.text

    def test_graph_endpoint_exposes_the_topology(self, client: TestClient) -> None:
        response = client.get("/api/agent/graph")
        assert response.status_code == 200
        body = response.json()
        assert len(body["nodes"]) == 7
        assert "relevance_grader" in body["conditional_edges"]
        assert body["mermaid"]


class TestHealthEndpoints:
    """Probes must never be traced and must report dependency state."""

    def test_liveness(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_reports_every_dependency(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()

        assert body["checks"]["database"] is True
        # No key configured in tests.
        assert body["checks"]["mesh_configured"] is False
        assert body["vector_store_mode"] == "embedded"
        assert body["models"]["base_url"]


class TestDigestDelivery:
    """BONUS 2: proactive email/Telegram delivery."""

    def test_html_email_renders_with_products(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.scheduler.email_digest import render_digest_html

        make_events(user, products)
        result = run_agent(user.id, reason="scheduled_digest")

        from app.database import session_scope

        with session_scope() as session:
            record = session.get(Recommendation, result.recommendation_id)
            assert record is not None
            payload = record.to_dict()

        html = render_digest_html(user.display_name, payload, payload["products"][:3])

        assert "<!DOCTYPE html>" in html
        assert user.display_name in html
        assert payload["products"][0]["title"] in html
        # Email clients need table layout and inline styles, not flexbox.
        assert "<table" in html
        assert "flex" not in html.lower().split("<body")[1][:2000]

    def test_plain_text_alternative_is_generated(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.database import session_scope
        from app.scheduler.email_digest import render_digest_text

        make_events(user, products)
        result = run_agent(user.id, reason="scheduled_digest")
        with session_scope() as session:
            payload = session.get(Recommendation, result.recommendation_id).to_dict()

        text = render_digest_text(user.display_name, payload, payload["products"][:3])

        assert user.display_name in text
        assert payload["products"][0]["title"] in text
        assert "/product/" in text

    def test_delivery_is_recorded_for_audit(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.scheduler.email_digest import deliver_digest

        make_events(user, products)
        result = run_agent(user.id, reason="scheduled_digest")

        outcomes = deliver_digest(user.id, result.recommendation_id)

        assert outcomes
        assert outcomes[0].channel == "email"
        assert outcomes[0].backend == "console"
        assert outcomes[0].ok is True

        rows = list(db.scalars(select(EmailDigest).where(EmailDigest.user_id == user.id)))
        assert len(rows) == 1
        assert rows[0].status == "sent"
        assert rows[0].recommendation_id == result.recommendation_id

    def test_telegram_is_skipped_without_a_token(self) -> None:
        from app.scheduler.email_digest import send_telegram

        outcome = send_telegram("12345", "hello")
        assert outcome.ok is False
        assert "TELEGRAM_BOT_TOKEN" in (outcome.error or "")

    def test_digest_job_selects_only_active_users(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.scheduler.jobs import daily_digest_job

        make_events(user, products, count=settings.digest_min_events_today + 2)

        summary = daily_digest_job()

        assert summary["candidates"] == 1
        assert summary["sent"] == 1
        assert summary["failed"] == 0

    def test_digest_job_skips_users_below_the_threshold(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.scheduler.jobs import daily_digest_job

        make_events(user, products, count=1, include_cart=False)

        summary = daily_digest_job()

        assert summary["candidates"] == 0
        assert summary["sent"] == 0

    def test_opted_out_users_are_excluded(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.scheduler.jobs import daily_digest_job

        make_events(user, products, count=settings.digest_min_events_today + 2)
        user.digest_opt_in = False
        db.commit()

        assert daily_digest_job()["candidates"] == 0


class TestHousekeeping:
    """Stale flags must not leave the UI stuck on a skeleton."""

    def test_stale_pending_flag_is_cleared(self, user: User) -> None:
        from app.cache import get_agent_pending
        from app.scheduler.jobs import housekeeping_job

        mark_agent_pending(user.id, "event_threshold")
        assert get_agent_pending(user.id) == "event_threshold"

        summary = housekeeping_job()

        assert summary["cleared"] == 1
        assert get_agent_pending(user.id) is None
