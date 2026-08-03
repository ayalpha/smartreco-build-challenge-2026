"""Tests for auth, the public catalog and the admin SQL ⇄ Qdrant dual-write.

The dual-write tests are the important ones: they assert that *both* stores move
together on create, update and delete, which is the property the whole retrieval
pipeline depends on.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.product import Product
from app.models.user import User, UserRole
from app.vector_store.qdrant_client import get_vector_store
from app.vector_store.sync import verify_sync


def _vector_ids() -> set[int]:
    """Return every product id currently present in the vector store."""
    store = get_vector_store()
    store.ensure_collection()
    points, _ = store.client().scroll(
        collection_name=store.collection, limit=500, with_payload=True
    )
    return {
        int((point.payload or {}).get("product_id") or point.id) for point in points
    }


# --------------------------------------------------------------------------- #
# Authentication                                                              #
# --------------------------------------------------------------------------- #

class TestAuthentication:
    """Registration, login and role enforcement."""

    def test_register_returns_a_token(self, client: TestClient) -> None:
        response = client.post(
            "/api/auth/register",
            json={"email": "new@test.dev", "password": "supersecret123", "full_name": "New"},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["access_token"]
        assert body["token_type"] == "bearer"

    def test_first_user_becomes_admin(self, client: TestClient, db: Session) -> None:
        client.post(
            "/api/auth/register",
            json={"email": "first@test.dev", "password": "supersecret123"},
        )
        client.post(
            "/api/auth/register",
            json={"email": "second@test.dev", "password": "supersecret123"},
        )

        first = db.scalars(select(User).where(User.email == "first@test.dev")).one()
        second = db.scalars(select(User).where(User.email == "second@test.dev")).one()

        assert first.role == UserRole.ADMIN
        assert second.role == UserRole.USER

    def test_duplicate_email_is_rejected(self, client: TestClient, user: User) -> None:
        response = client.post(
            "/api/auth/register",
            json={"email": user.email, "password": "supersecret123"},
        )
        assert response.status_code == 409

    def test_short_password_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/auth/register", json={"email": "x@test.dev", "password": "short"}
        )
        assert response.status_code == 422

    def test_login_with_valid_credentials(self, client: TestClient, make_user: Any) -> None:
        make_user(email="login@test.dev", password="correctpassword1")

        response = client.post(
            "/api/auth/login",
            json={"email": "login@test.dev", "password": "correctpassword1"},
        )

        assert response.status_code == 200
        assert response.json()["access_token"]

    def test_login_with_wrong_password_is_401(
        self, client: TestClient, make_user: Any
    ) -> None:
        make_user(email="login2@test.dev", password="correctpassword1")

        response = client.post(
            "/api/auth/login",
            json={"email": "login2@test.dev", "password": "wrongpassword1"},
        )
        assert response.status_code == 401

    def test_login_for_unknown_email_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/api/auth/login", json={"email": "ghost@test.dev", "password": "whatever123"}
        )
        assert response.status_code == 401

    def test_me_returns_the_profile(
        self, client: TestClient, user: User, auth_headers: dict[str, str]
    ) -> None:
        response = client.get("/api/auth/me", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["email"] == user.email
        assert body["role"] == "user"

    def test_html_login_sets_a_cookie(self, client: TestClient, make_user: Any) -> None:
        from app.security import ACCESS_COOKIE_NAME

        make_user(email="form@test.dev", password="correctpassword1")

        response = client.post(
            "/login",
            data={"email": "form@test.dev", "password": "correctpassword1", "next": "/"},
            follow_redirects=False,
        )

        assert response.status_code == 303
        assert ACCESS_COOKIE_NAME in response.cookies

    def test_open_redirect_is_blocked(self, client: TestClient, make_user: Any) -> None:
        make_user(email="redir@test.dev", password="correctpassword1")

        response = client.post(
            "/login",
            data={
                "email": "redir@test.dev",
                "password": "correctpassword1",
                "next": "https://evil.example.com/steal",
            },
            follow_redirects=False,
        )

        assert response.status_code == 303
        assert response.headers["location"] == "/"

    def test_password_hashing_is_not_reversible(self) -> None:
        from app.security import hash_password, verify_password

        hashed = hash_password("correct horse battery staple")

        assert "correct horse" not in hashed
        assert verify_password("correct horse battery staple", hashed) is True
        assert verify_password("wrong password", hashed) is False


# --------------------------------------------------------------------------- #
# Public catalog                                                              #
# --------------------------------------------------------------------------- #

class TestCatalog:
    """Listing, filtering and detail pages."""

    def test_list_returns_every_active_product(
        self, client: TestClient, products: list[Product]
    ) -> None:
        response = client.get("/api/products?page_size=50")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == len(products)
        assert len(body["items"]) == len(products)
        assert isinstance(body["items"][0]["tags"], list)

    def test_search_filters_results(
        self, client: TestClient, products: list[Product]
    ) -> None:
        body = client.get("/api/products?q=langgraph").json()

        assert body["total"] >= 1
        assert any("LangGraph" in item["title"] for item in body["items"])

    def test_category_and_level_filters(
        self, client: TestClient, products: list[Product]
    ) -> None:
        body = client.get("/api/products?category=Agentic%20AI&skill_level=advanced").json()

        assert body["total"] >= 1
        assert all(item["category"] == "Agentic AI" for item in body["items"])
        assert all(item["skill_level"] == "advanced" for item in body["items"])

    def test_max_price_filter(self, client: TestClient, products: list[Product]) -> None:
        body = client.get("/api/products?max_price=40").json()
        assert all(item["price"] <= 40 for item in body["items"])

    def test_categories_endpoint(self, client: TestClient, products: list[Product]) -> None:
        categories = client.get("/api/products/categories").json()
        assert "Agentic AI" in categories
        assert categories == sorted(categories)

    def test_detail_endpoint(self, client: TestClient, products: list[Product]) -> None:
        response = client.get(f"/api/products/{products[0].id}")

        assert response.status_code == 200
        assert response.json()["title"] == products[0].title

    def test_missing_product_is_404(self, client: TestClient) -> None:
        assert client.get("/api/products/999999").status_code == 404

    def test_catalog_page_renders(self, client: TestClient, products: list[Product]) -> None:
        response = client.get("/catalog?q=langgraph")

        assert response.status_code == 200
        assert "Course catalog" in response.text
        # The search meta tag is what makes tracker.js emit a real result count.
        assert 'name="smartreco-search-query"' in response.text

    def test_product_page_exposes_tracking_meta(
        self, client: TestClient, products: list[Product]
    ) -> None:
        response = client.get(f"/product/{products[0].id}")

        assert response.status_code == 200
        assert f'content="{products[0].id}"' in response.text
        # The save control replaced the old "Add to cart" button; it is wired to
        # the saved-items API and still emits an add_to_cart behavioural event.
        assert "Save for later" in response.text
        assert f'data-save-product="{products[0].id}"' in response.text

    def test_missing_product_page_is_404(self, client: TestClient) -> None:
        response = client.get("/product/999999")
        assert response.status_code == 404
        assert "couldn't find that page" in response.text


# --------------------------------------------------------------------------- #
# Admin dual-write                                                            #
# --------------------------------------------------------------------------- #

class TestAdminAuthorisation:
    """Admin surfaces reject non-admins."""

    def test_dashboard_redirects_regular_users(
        self, client: TestClient, user: User
    ) -> None:
        from app.security import ACCESS_COOKIE_NAME, create_access_token

        client.cookies.set(ACCESS_COOKIE_NAME, create_access_token(user.id, role="user"))
        response = client.get("/admin", follow_redirects=False)

        assert response.status_code in (302, 303)
        assert response.headers["location"] == "/catalog"

    def test_api_forbids_regular_users(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/api/admin/products",
            json={"title": "Sneaky course", "category": "Hacking"},
            headers=auth_headers,
        )
        assert response.status_code == 403

    def test_api_requires_authentication(self, client: TestClient) -> None:
        assert client.get("/api/admin/stats").status_code == 401

    def test_dashboard_renders_for_admins(
        self, client: TestClient, admin: User, products: list[Product]
    ) -> None:
        from app.security import ACCESS_COOKIE_NAME, create_access_token

        client.cookies.set(ACCESS_COOKIE_NAME, create_access_token(admin.id, role="admin"))
        response = client.get("/admin")

        assert response.status_code == 200
        assert "Dual-write status" in response.text
        assert "Mesh API" in response.text


class TestDualWrite:
    """SQL and Qdrant must move together on every mutation."""

    def test_create_writes_to_both_stores(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        response = client.post(
            "/api/admin/products",
            json={
                "title": "Retrieval Augmented Generation in Depth",
                "description": "Chunking, embeddings, reranking and evaluation for RAG systems.",
                "category": "Agentic AI",
                "tags": ["rag", "embeddings", "evaluation"],
                "price": 75.0,
                "skill_level": "intermediate",
                "duration": "9 hours",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        created_id = response.json()["id"]

        # SQL
        row = db.get(Product, created_id)
        assert row is not None
        assert row.tag_list == ["rag", "embeddings", "evaluation"]

        # Vector store
        assert created_id in _vector_ids()

    def test_created_product_is_immediately_retrievable(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        from app.vector_store.sync import hybrid_retrieve

        client.post(
            "/api/admin/products",
            json={
                "title": "Graph Neural Networks for Recommenders",
                "description": (
                    "Message passing over user-item interaction graphs for recommendation, "
                    "including sampling strategies and inductive inference."
                ),
                "category": "Deep Learning",
                "tags": ["gnn", "graphs", "recommenders"],
                "price": 95.0,
                "skill_level": "advanced",
            },
            headers=admin_headers,
        )

        hits = hybrid_retrieve(db, "graph neural networks for recommendation", limit=5)
        titles = [hit.payload.get("title") for hit in hits]
        assert "Graph Neural Networks for Recommenders" in titles

    def test_update_refreshes_the_vector_and_bumps_the_revision(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        target = products[0]
        original_revision = target.revision

        response = client.patch(
            f"/api/admin/products/{target.id}",
            json={"title": "Building Production Agents with LangGraph (2026 edition)",
                  "price": 99.0},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.expire_all()
        refreshed = db.get(Product, target.id)
        assert refreshed is not None
        assert refreshed.title.endswith("(2026 edition)")
        assert refreshed.price == 99.0
        assert refreshed.revision == original_revision + 1
        assert target.id in _vector_ids()

    def test_delete_removes_from_both_stores(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        target_id = products[-1].id
        assert target_id in _vector_ids()

        response = client.delete(f"/api/admin/products/{target_id}", headers=admin_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["deleted"] is True
        assert body["vector_deleted"] is True

        # This session cached the row when the fixture created it, so its identity
        # map has to be cleared before `get` will re-query.
        db.expire_all()
        assert db.get(Product, target_id) is None
        assert target_id not in _vector_ids()

    def test_deactivating_removes_the_vector_but_keeps_the_row(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        target_id = products[1].id

        response = client.patch(
            f"/api/admin/products/{target_id}",
            json={"is_active": False},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.expire_all()
        assert db.get(Product, target_id) is not None, "the SQL row must be retained"
        assert target_id not in _vector_ids(), "an inactive course must not be recommendable"

    def test_invalid_skill_level_is_rejected(
        self, client: TestClient, admin_headers: dict[str, str]
    ) -> None:
        response = client.post(
            "/api/admin/products",
            json={"title": "Bad course", "category": "Testing", "skill_level": "wizard"},
            headers=admin_headers,
        )
        assert response.status_code == 422

    def test_sync_verification_reports_agreement(
        self, db: Session, products: list[Product]
    ) -> None:
        status = verify_sync(db)

        assert status["sql_count"] == len(products)
        assert status["vector_count"] == len(products)
        assert status["in_sync"] is True

    def test_reindex_is_idempotent(
        self, client: TestClient, db: Session, admin_headers: dict[str, str],
        products: list[Product],
    ) -> None:
        first = client.post("/api/admin/reindex", headers=admin_headers).json()
        second = client.post("/api/admin/reindex", headers=admin_headers).json()

        assert first["ok"] is True
        assert second["ok"] is True
        assert first["written"] == second["written"] == len(products)
        assert verify_sync(db)["in_sync"] is True

    def test_stats_endpoint_summarises_everything(
        self, client: TestClient, admin_headers: dict[str, str], products: list[Product]
    ) -> None:
        stats = client.get("/api/admin/stats", headers=admin_headers).json()

        assert stats["products"] == len(products)
        assert stats["vector"]["in_sync"] is True
        assert stats["mesh"]["configured"] is False
        assert "langsmith" in stats


class TestSeedScript:
    """The seeder must be genuinely idempotent."""

    def test_seeding_twice_does_not_duplicate(self, db: Session) -> None:
        from scripts.seed_products import CATALOG, seed_catalog

        created_first, updated_first = seed_catalog(db)
        db.commit()
        created_second, updated_second = seed_catalog(db)
        db.commit()

        assert created_first == len(CATALOG)
        assert created_second == 0
        assert updated_second == 0

        total = len(list(db.scalars(select(Product))))
        assert total == len(CATALOG)

    def test_catalog_covers_every_required_category(self) -> None:
        from scripts.seed_products import CATALOG

        categories = {entry["category"] for entry in CATALOG}
        required = {
            "Machine Learning", "Deep Learning", "Agentic AI", "Data Engineering",
            "Web Development", "DevOps", "Cloud", "Python", "JavaScript", "Career Skills",
        }

        assert required <= categories, f"missing categories: {required - categories}"
        assert len(CATALOG) >= 30, "the brief requires at least 30 seeded courses"

    def test_every_seeded_course_is_well_formed(self) -> None:
        from scripts.seed_products import CATALOG

        for entry in CATALOG:
            assert entry["title"]
            assert len(entry["description"]) > 80, f"{entry['title']} needs a real description"
            assert entry["skill_level"] in {"beginner", "intermediate", "advanced"}
            assert entry["price"] >= 0
            assert entry["tags"]
