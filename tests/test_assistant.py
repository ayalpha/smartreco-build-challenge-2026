"""Tests for the personalised agent chat and the saved-items feature.

Both run with ``MESH_API_KEY`` unset, so the chat exercises its graceful
degradation path — the reply is still personalised from the user's own signals,
just templated rather than model-written.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.runner import run_agent
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.user import User
from app.routers.assistant import build_profile


class TestChatAuthorisation:
    """The chat is private — there is no anonymous personalisation to ground it."""

    def test_chat_requires_authentication(self, client: TestClient) -> None:
        response = client.post("/api/assistant/chat", json={"message": "hi"})
        assert response.status_code == 401

    def test_profile_requires_authentication(self, client: TestClient) -> None:
        assert client.get("/api/assistant/profile").status_code == 401

    def test_empty_message_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/api/assistant/chat", json={"message": ""}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_oversized_message_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/api/assistant/chat", json={"message": "x" * 5000}, headers=auth_headers
        )
        assert response.status_code == 422


class TestPersonalisationProfile:
    """The profile is assembled from the agent's own computed signals."""

    def test_profile_reuses_agent_interest_signals(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        run_agent(user.id, reason="manual")

        profile = build_profile(db, user)

        assert profile["has_agent_run"] is True
        assert profile["signals"], "should reuse the recommendation's interest signals"
        assert profile["digest"], "should reuse the activity_analyzer digest"
        assert profile["event_count"] > 0
        assert all("topic" in s for s in profile["signals"])

    def test_profile_falls_back_to_events_before_any_agent_run(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        """A brand-new user with no recommendation is still personalised."""
        make_events(user, products)

        profile = build_profile(db, user)

        assert profile["has_agent_run"] is False
        assert profile["signals"], "should derive signals from raw events"
        assert profile["recent_titles"]

    def test_profile_is_empty_for_a_user_with_no_activity(
        self, db: Session, user: User
    ) -> None:
        profile = build_profile(db, user)
        assert profile["signals"] == []
        assert profile["event_count"] == 0


class TestChatEndpoint:
    """Replies are grounded in the caller's own behaviour."""

    def test_reply_is_grounded_and_returns_courses(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)
        run_agent(user.id, reason="manual")

        response = client.post(
            "/api/assistant/chat",
            json={"message": "What should I learn next?"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["reply"]
        assert body["grounded"] is True
        assert body["signals_used"], "the reply must name the signals it used"
        assert 1 <= len(body["courses"]) <= 3
        assert all(c["id"] and c["title"] for c in body["courses"])
        # Mesh is unset in tests, so this must be the degraded-but-personalised path.
        assert body["degraded"] is True

    def test_two_users_get_different_personalisation(
        self, client: TestClient, db: Session, user: User, make_user: Any,
        products: list[Product], make_events: Any,
    ) -> None:
        """The whole premise: identical question, different profiles."""
        from app.security import create_access_token

        agentic = [p for p in products if p.category == "Agentic AI"]
        other = [p for p in products if p.category != "Agentic AI"]

        second = make_user(email="second@test.dev")
        make_events(user, agentic)
        make_events(second, other)

        headers_one = {"Authorization": f"Bearer {create_access_token(user.id, role='user')}"}
        headers_two = {"Authorization": f"Bearer {create_access_token(second.id, role='user')}"}

        one = client.post("/api/assistant/chat", json={"message": "What next?"},
                          headers=headers_one).json()
        two = client.post("/api/assistant/chat", json={"message": "What next?"},
                          headers=headers_two).json()

        assert one["signals_used"] != two["signals_used"], (
            "two users with different behaviour must not share a profile"
        )

    def test_profile_endpoint_reports_signals(
        self, client: TestClient, user: User, auth_headers: dict[str, str],
        products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products)
        response = client.get("/api/assistant/profile", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["display_name"] == user.display_name
        assert body["event_count"] > 0
        assert isinstance(body["signals"], list)


class TestSavedItems:
    """Saves are add_to_cart events, so they also feed the agent."""

    def test_save_creates_an_add_to_cart_event(
        self, client: TestClient, db: Session, user: User,
        auth_headers: dict[str, str], products: list[Product],
    ) -> None:
        target = products[0]

        response = client.post(
            f"/api/assistant/saved/{target.id}", headers=auth_headers
        )

        assert response.status_code == 200
        body = response.json()
        assert body["saved"] is True
        assert body["count"] == 1

        events = list(
            db.scalars(
                select(Event).where(
                    Event.user_id == user.id,
                    Event.event_type == EventType.ADD_TO_CART.value,
                )
            )
        )
        assert len(events) == 1
        assert events[0].product_id == target.id

    def test_saving_is_idempotent(
        self, client: TestClient, auth_headers: dict[str, str], products: list[Product]
    ) -> None:
        target = products[0]
        client.post(f"/api/assistant/saved/{target.id}", headers=auth_headers)
        second = client.post(f"/api/assistant/saved/{target.id}", headers=auth_headers)

        assert second.json()["count"] == 1, "saving twice must not double-count"

    def test_unsave_removes_the_item(
        self, client: TestClient, auth_headers: dict[str, str], products: list[Product]
    ) -> None:
        target = products[0]
        client.post(f"/api/assistant/saved/{target.id}", headers=auth_headers)

        response = client.delete(f"/api/assistant/saved/{target.id}", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["saved"] is False
        assert response.json()["count"] == 0

    def test_list_returns_saved_items(
        self, client: TestClient, auth_headers: dict[str, str], products: list[Product]
    ) -> None:
        client.post(f"/api/assistant/saved/{products[0].id}", headers=auth_headers)
        client.post(f"/api/assistant/saved/{products[1].id}", headers=auth_headers)

        body = client.get("/api/assistant/saved", headers=auth_headers).json()

        assert body["count"] == 2
        assert {i["id"] for i in body["items"]} == {products[0].id, products[1].id}
        assert all(i["thumbnail_url"] for i in body["items"]), "covers must be wired"

    def test_saving_an_unknown_course_is_404(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        assert client.post("/api/assistant/saved/999999", headers=auth_headers).status_code == 404

    def test_anonymous_visitors_can_save_by_session(
        self, client: TestClient, products: list[Product]
    ) -> None:
        client.cookies.set("smartreco_session", "anon-session-1")

        response = client.post(f"/api/assistant/saved/{products[0].id}")

        assert response.status_code == 200
        assert response.json()["count"] == 1

    def test_saved_page_renders(
        self, client: TestClient, auth_headers: dict[str, str], products: list[Product]
    ) -> None:
        from app.security import ACCESS_COOKIE_NAME, create_access_token
        from app.models.user import User as UserModel  # noqa: F401

        client.post(f"/api/assistant/saved/{products[0].id}", headers=auth_headers)
        # The page is cookie-authenticated like every other rendered surface.
        token = auth_headers["Authorization"].split(" ", 1)[1]
        client.cookies.set(ACCESS_COOKIE_NAME, token)

        response = client.get("/saved")

        assert response.status_code == 200
        assert "Saved courses" in response.text
        assert products[0].title in response.text

    def test_saved_page_empty_state(self, client: TestClient) -> None:
        response = client.get("/saved")
        assert response.status_code == 200
        assert "No saved courses yet" in response.text
