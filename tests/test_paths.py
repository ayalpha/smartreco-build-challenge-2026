"""Tests for personalized learning paths."""

from __future__ import annotations

from app.models.user import User
from app.routers.paths import behaviour_context, build_path, fallback_path


def test_path_page_is_public_with_sign_in_cta(client) -> None:
    """Guests can view the Path teaser; building requires sign-in."""
    response = client.get("/path", headers={"Accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 200
    assert "Become what you want" in response.text
    assert "/login?next=/path" in response.text
    assert "Sign in to build your path" in response.text


def test_path_post_requires_auth(client) -> None:
    """POST /path must not build a path for anonymous visitors."""
    response = client.post(
        "/path",
        data={"goal": "Become an engineer", "weekly_hours": "5"},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
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


def test_merge_career_goal_into_mesh_signals() -> None:
    """Mesh interest output should still surface a Path career goal."""
    from app.agent.nodes import _merge_career_goal

    signals, query = _merge_career_goal(
        [{"topic": "python", "confidence": 0.7, "evidence": "clicks"}],
        "A course about python tooling",
        "Become an AI platform engineer",
    )
    assert signals[0]["topic"] == "Become an AI platform engineer"
    assert "AI platform engineer" in query
    # Idempotent when already present.
    again, q2 = _merge_career_goal(signals, query, "Become an AI platform engineer")
    assert sum(1 for s in again if "AI platform" in s["topic"]) == 1
    assert q2 == query


def test_nav_includes_path_link(client) -> None:
    """Global navigation exposes the Path option to every visitor."""
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert 'href="/path"' in response.text
    assert ">Path<" in response.text or "Build a path" in response.text


def test_path_page_prefills_saved_goal(client, auth_headers, user, db) -> None:
    """GET /path should show the learner's stored career goal."""
    user.career_goal = "Become a full-stack engineer"
    db.add(user)
    db.commit()

    response = client.get(
        "/path",
        headers={**auth_headers, "Accept": "text/html"},
    )
    assert response.status_code == 200
    assert "Become a full-stack engineer" in response.text


def test_path_routes_are_registered(client, auth_headers) -> None:
    """HTML and JSON path endpoints must be mounted on the app."""
    assert client.get("/path", headers={**auth_headers, "Accept": "text/html"}).status_code == 200
    assert client.get("/api/path", headers=auth_headers).status_code == 200
    openapi = client.get("/openapi.json").json()
    paths = openapi.get("paths") or {}
    assert "/api/path" in paths


def test_footer_and_signed_in_path_show_goal_chips(client, auth_headers) -> None:
    """Footer links Path; signed-in builder offers sample goal chips."""
    home = client.get("/", headers={"Accept": "text/html"})
    assert home.status_code == 200
    assert 'href="/path"' in home.text

    page = client.get("/path", headers={**auth_headers, "Accept": "text/html"})
    assert page.status_code == 200
    assert "path-goal-chip" in page.text
    assert "Become an agentic AI engineer" in page.text


def test_activity_digest_includes_career_goal(db, user, products, make_events) -> None:
    """Agent activity analysis should surface a saved Path goal."""
    from app.agent.nodes import activity_analyzer
    from app.agent.state import make_initial_state

    user.career_goal = "Become a data engineer"
    db.add(user)
    db.commit()
    make_events(user, products[:2], count=4)

    update = activity_analyzer(make_initial_state(user.id, trigger_reason="manual"))
    digest = str(update.get("behavior_digest") or "")
    assert "data engineer" in digest.lower()
