"""Tests for the behavioural event ingest endpoint.

The ingest path is the hot path, so these tests pin down the properties that
matter operationally: it accepts both ``fetch`` and ``sendBeacon`` payloads, it
never loses a whole batch because of one bad item, it nulls unknown foreign keys
instead of erroring, and it returns fast without running the agent inline.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.user import User
from app.schemas.event import MAX_BATCH_SIZE, MAX_TIME_SPENT_SECONDS, EventIn

settings = get_settings()

ENDPOINT = "/api/events"


def _count_events(db: Session, user_id: int | None = None) -> int:
    """Count stored events, optionally for one user."""
    statement = select(func.count()).select_from(Event)
    if user_id is not None:
        statement = statement.where(Event.user_id == user_id)
    return int(db.scalar(statement) or 0)


class TestEventIngestion:
    """Happy-path ingestion for authenticated and anonymous callers."""

    def test_batch_is_persisted(
        self, client: TestClient, db: Session, user: User,
        auth_headers: dict[str, str], products: list[Product],
    ) -> None:
        payload = {
            "session_id": "session-abc",
            "events": [
                {"event_type": "page_view", "path": "/", "metadata": {"referrer": None}},
                {
                    "event_type": "product_click",
                    "product_id": products[0].id,
                    "path": f"/product/{products[0].id}",
                    "metadata": {"source": "catalog"},
                },
                {
                    "event_type": "search_query",
                    "path": "/catalog",
                    "metadata": {"query": "langgraph", "result_count": 3},
                },
            ],
        }

        response = client.post(ENDPOINT, json=payload, headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] == 3
        assert body["rejected"] == 0
        assert _count_events(db, user.id) == 3

        stored = list(db.scalars(select(Event).where(Event.user_id == user.id)))
        assert {row.event_type for row in stored} == {
            "page_view", "product_click", "search_query",
        }
        assert all(row.session_id == "session-abc" for row in stored)

    def test_flat_metadata_is_collected(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        """``tracker.js`` may send convenience fields flat rather than nested."""
        response = client.post(
            ENDPOINT,
            json={
                "session_id": "s1",
                "events": [
                    {"event_type": "search_query", "query": "pytorch", "result_count": 7}
                ],
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        event = db.scalars(select(Event).where(Event.user_id == user.id)).first()
        assert event is not None
        assert event.metadata_json["query"] == "pytorch"
        assert event.metadata_json["result_count"] == 7

    def test_anonymous_events_are_stored_without_a_user(
        self, client: TestClient, db: Session
    ) -> None:
        response = client.post(
            ENDPOINT,
            json={"session_id": "anon-1", "events": [{"event_type": "page_view", "path": "/"}]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] == 1
        # No user, so the agent cannot be triggered.
        assert body["triggered"] is False

        event = db.scalars(select(Event)).first()
        assert event is not None
        assert event.user_id is None
        assert event.session_id == "anon-1"

    def test_empty_batch_is_accepted(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            ENDPOINT, json={"session_id": "s", "events": []}, headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["accepted"] == 0


class TestSendBeaconCompatibility:
    """``navigator.sendBeacon`` does not send ``application/json``."""

    def test_text_plain_body_is_accepted(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        import json

        body = json.dumps(
            {"session_id": "beacon-1", "events": [{"event_type": "page_view", "path": "/"}]}
        )
        response = client.post(
            ENDPOINT,
            content=body,
            headers={**auth_headers, "Content-Type": "text/plain;charset=UTF-8"},
        )

        assert response.status_code == 200
        assert response.json()["accepted"] == 1
        assert _count_events(db, user.id) == 1

    def test_no_content_type_is_accepted(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        import json

        response = client.post(
            ENDPOINT,
            content=json.dumps(
                {"session_id": "b2", "events": [{"event_type": "add_to_cart"}]}
            ),
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert _count_events(db, user.id) == 1

    def test_garbage_body_does_not_error(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """A failed beacon must never surface an error in the user's console."""
        response = client.post(
            ENDPOINT, content=b"\x00\x01not json", headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["accepted"] == 0


class TestPartialFailureHandling:
    """One bad event must not cost the whole batch."""

    def test_unknown_event_type_is_rejected_individually(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            ENDPOINT,
            json={
                "session_id": "s",
                "events": [
                    {"event_type": "page_view", "path": "/"},
                    {"event_type": "definitely_not_a_real_event"},
                    {"event_type": "add_to_cart"},
                ],
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] == 2
        assert body["rejected"] == 1
        assert _count_events(db, user.id) == 2

    def test_unknown_product_id_is_nulled_not_rejected(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            ENDPOINT,
            json={
                "session_id": "s",
                "events": [{"event_type": "product_click", "product_id": 999_999}],
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["accepted"] == 1

        event = db.scalars(select(Event)).first()
        assert event is not None
        assert event.product_id is None, "a stale client id must not break ingestion"

    def test_non_object_entries_are_skipped(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            ENDPOINT,
            json={"session_id": "s", "events": ["nonsense", 42, {"event_type": "page_view"}]},
            headers=auth_headers,
        )
        body = response.json()
        assert body["accepted"] == 1
        assert body["rejected"] == 2


class TestSchemaGuards:
    """Validation rules that protect the database from abuse."""

    def test_time_spent_is_clamped(self) -> None:
        event = EventIn(
            event_type="time_spent", product_id=1, metadata={"seconds": 10**9}
        )
        assert event.collected_metadata()["seconds"] == MAX_TIME_SPENT_SECONDS

    def test_negative_time_spent_is_floored(self) -> None:
        event = EventIn(event_type="time_spent", metadata={"seconds": -50})
        assert event.collected_metadata()["seconds"] == 0

    def test_unparseable_seconds_is_dropped(self) -> None:
        event = EventIn(event_type="time_spent", metadata={"seconds": "not-a-number"})
        assert "seconds" not in event.collected_metadata()

    def test_oversized_batch_is_truncated(
        self, client: TestClient, db: Session, user: User, auth_headers: dict[str, str]
    ) -> None:
        events = [{"event_type": "page_view", "path": "/"} for _ in range(MAX_BATCH_SIZE + 50)]
        response = client.post(
            ENDPOINT, json={"session_id": "s", "events": events}, headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json()["accepted"] == MAX_BATCH_SIZE
        assert _count_events(db, user.id) == MAX_BATCH_SIZE

    def test_event_type_is_case_normalised(self) -> None:
        assert EventIn(event_type="PAGE_VIEW").event_type == "page_view"

    def test_all_six_event_types_are_accepted(
        self, client: TestClient, db: Session, user: User,
        auth_headers: dict[str, str], products: list[Product],
    ) -> None:
        events = [
            {"event_type": EventType.PAGE_VIEW.value, "path": "/"},
            {"event_type": EventType.PRODUCT_CLICK.value, "product_id": products[0].id,
             "metadata": {"source": "catalog"}},
            {"event_type": EventType.SEARCH_QUERY.value,
             "metadata": {"query": "x", "result_count": 1}},
            {"event_type": EventType.TIME_SPENT.value, "product_id": products[0].id,
             "metadata": {"seconds": 45}},
            {"event_type": EventType.ADD_TO_CART.value, "product_id": products[0].id},
            {"event_type": EventType.RECOMMENDATION_CLICK.value,
             "product_id": products[0].id, "metadata": {"recommendation_id": 1}},
        ]

        response = client.post(
            ENDPOINT, json={"session_id": "s", "events": events}, headers=auth_headers
        )

        assert response.status_code == 200
        assert response.json()["accepted"] == 6
        assert _count_events(db, user.id) == 6


class TestTriggerIntegration:
    """Ingest evaluates the trigger policy but never runs the agent inline."""

    def test_first_batch_triggers_the_agent(
        self, client: TestClient, user: User,
        auth_headers: dict[str, str], products: list[Product],
    ) -> None:
        response = client.post(
            ENDPOINT,
            json={
                "session_id": "s",
                "events": [
                    {"event_type": "product_click", "product_id": products[0].id,
                     "metadata": {"source": "catalog"}}
                ],
            },
            headers=auth_headers,
        )

        body = response.json()
        assert body["triggered"] is True
        assert body["trigger_reason"] == "first_time"

    def test_second_batch_does_not_retrigger(
        self, client: TestClient, user: User,
        auth_headers: dict[str, str], products: list[Product], make_events: Any,
    ) -> None:
        from app.agent.runner import run_agent

        make_events(user, products, count=2)
        run_agent(user.id, reason="manual")

        response = client.post(
            ENDPOINT,
            json={"session_id": "s", "events": [{"event_type": "page_view", "path": "/"}]},
            headers=auth_headers,
        )

        assert response.json()["triggered"] is False


class TestEventSummary:
    """The decorative activity summary used by the profile page."""

    def test_summary_for_authenticated_user(
        self, client: TestClient, user: User,
        auth_headers: dict[str, str], products: list[Product], make_events: Any,
    ) -> None:
        make_events(user, products, count=4)

        response = client.get(f"{ENDPOINT}/summary", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["authenticated"] is True
        assert body["total"] > 0
        assert isinstance(body["by_type"], dict)

    def test_summary_for_anonymous_visitor(self, client: TestClient) -> None:
        response = client.get(f"{ENDPOINT}/summary")
        assert response.status_code == 200
        assert response.json() == {"authenticated": False, "total": 0, "by_type": {}}
