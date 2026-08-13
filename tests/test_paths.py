"""Tests for personalized learning paths."""

from __future__ import annotations

from app.models.user import User
from app.routers.paths import behaviour_context, build_path, fallback_path


def test_path_page_requires_auth(client) -> None:
    """Guests are redirected to login rather than seeing an empty form."""
    response = client.get("/path", headers={"Accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 303
    assert "/login" in response.headers.get("location", "")


def test_path_page_renders_for_signed_in_user(client, auth_headers) -> None:
    """Authenticated learners get the builder form."""
    response = client.get(
        "/path",
        headers={**auth_headers, "Accept": "text/html"},
    )
    assert response.status_code == 200
    body = response.text
    assert "Become what you want" in body
    assert 'name="goal"' in body
    assert "Build my path" in body


def test_build_path_uses_catalog_and_behaviour(db, user: User, products, make_events) -> None:
    """Fallback path orders real catalog courses and cites the submitted goal."""
    make_events(user, products[:2], count=8)

    result = build_path(db, user, "Become an agentic AI engineer", weekly_hours=5)

    assert result["headline"]
    assert result["steps"]
    product_ids = {p.id for p in products}
    assert all(step["product_id"] in product_ids for step in result["steps"])
    assert [step["order"] for step in result["steps"]] == list(
        range(1, len(result["steps"]) + 1)
    )
    # Without Mesh keys in tests, the builder must still succeed via fallback.
    assert result["degraded"] is True
    assert "agentic AI engineer" in result["headline"] or "agentic AI engineer" in result["summary"]


def test_fallback_orders_by_skill_level(products) -> None:
    """Beginner material should appear before advanced when scores are equal."""
    catalog = [
        {
            "id": products[2].id,
            "title": products[2].title,
            "skill_level": "advanced",
            "duration": "16 hours",
            "category": products[2].category,
            "score": 0.5,
        },
        {
            "id": products[4].id,
            "title": products[4].title,
            "skill_level": "beginner",
            "duration": "6 hours",
            "category": products[4].category,
            "score": 0.5,
        },
        {
            "id": products[0].id,
            "title": products[0].title,
            "skill_level": "intermediate",
            "duration": "14 hours",
            "category": products[0].category,
            "score": 0.5,
        },
    ]
    path = fallback_path("Staff engineer", catalog, interests=["python"], weekly_hours=5)
    levels = [step["level"] for step in path["steps"]]
    assert levels[0] == "beginner"
    assert levels[-1] == "advanced"
    assert path["degraded"] is True
    assert path["steps"][0]["weeks"] >= 1


def test_behaviour_context_reads_product_categories(
    db, user: User, products, make_events
) -> None:
    """Recent product interactions should surface category interest signals."""
    make_events(user, products[:2], count=6)
    interests, searches = behaviour_context(db, user.id)
    assert isinstance(interests, list)
    assert isinstance(searches, str)
    # Sample products[0:2] are Agentic AI — clicks should lift that category.
    assert interests
    assert any("Agentic" in topic for topic in interests)


def test_create_path_post(client, auth_headers, products, user, make_events, db) -> None:
    """POST /path renders a path section and persists the career goal."""
    make_events(user, products[:3], count=5)
    response = client.post(
        "/path",
        data={"goal": "Become a multi-agent systems engineer", "weekly_hours": "8"},
        headers={**auth_headers, "Accept": "text/html"},
    )
    assert response.status_code == 200
    assert "personalized path" in response.text.lower() or "Your path" in response.text
    assert "View course" in response.text

    db.refresh(user)
    assert user.career_goal == "Become a multi-agent systems engineer"


def test_path_json_api(client, auth_headers, products, user, make_events, db) -> None:
    """POST /api/path returns structured steps and stores the goal."""
    make_events(user, products[:2], count=4)
    response = client.post(
        "/api/path",
        json={"goal": "Become an MLOps engineer", "weekly_hours": 10},
        headers=auth_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["goal"] == "Become an MLOps engineer"
    assert payload["weekly_hours"] == 10
    assert payload["steps"]
    assert payload["degraded"] is True  # no Mesh key in tests
    assert all("product_id" in step for step in payload["steps"])

    saved = client.get("/api/path", headers=auth_headers)
    assert saved.status_code == 200
    assert saved.json()["has_goal"] is True
    assert "MLOps" in saved.json()["goal"]

    db.refresh(user)
    assert user.career_goal == "Become an MLOps engineer"


def test_heuristic_interests_include_career_goal() -> None:
    """Career goals should bias the heuristic interest extractor."""
    from app.agent.nodes import _heuristic_interests

    signals, query = _heuristic_interests(
        [], {}, [], career_goal="Become a staff ML engineer"
    )
    assert any("staff ML engineer" in s["topic"] for s in signals)
    assert "staff ML engineer" in query
