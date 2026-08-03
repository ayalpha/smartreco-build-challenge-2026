"""Shared pytest fixtures.

Isolation strategy
------------------
Environment variables are set **before** any ``app.*`` import, because several
modules capture ``get_settings()`` at import time.  The test environment is
deliberately dependency-free:

* SQLite in a temp directory instead of PostgreSQL;
* an unreachable Qdrant URL, so :class:`~app.vector_store.qdrant_client.VectorStore`
  falls back to its embedded in-memory mode;
* no ``REDIS_URL``, so the cache falls back to the in-process backend;
* no ``MESH_API_KEY``, so every agent node exercises its graceful-degradation
  path.  The full graph therefore runs end to end in CI with zero credentials —
  which is exactly what makes these tests meaningful rather than mocked.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import pytest

# --------------------------------------------------------------------------- #
# Environment — must be configured before importing the application           #
# --------------------------------------------------------------------------- #

_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="smartreco-tests-"))
_TEST_DB_PATH = _TEST_DB_DIR / "test.db"

os.environ.update(
    ENVIRONMENT="test",
    DEBUG="true",
    LOG_LEVEL="WARNING",
    DATABASE_URL=f"sqlite:///{_TEST_DB_PATH}",
    # Unreachable on purpose: exercises the embedded-Qdrant fallback.
    QDRANT_URL="http://127.0.0.1:6399",
    QDRANT_COLLECTION=f"smartreco_test_{uuid.uuid4().hex[:8]}",
    REDIS_URL="",
    MESH_API_KEY="",
    SECRET_KEY="test-secret-key-that-is-long-enough-for-hs256",
    SCHEDULER_ENABLED="false",
    LANGSMITH_TRACING="false",
    EMAIL_ENABLED="true",
    EMAIL_BACKEND="console",
    TELEGRAM_BOT_TOKEN="",
    AGENT_EVENT_TRIGGER_INTERVAL="10",
    AGENT_FINAL_PRODUCT_COUNT="4",
    VECTOR_SEARCH_TOP_K="8",
)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.cache import reset_cache_backend  # noqa: E402
from app.database import Base, SessionLocal, engine, init_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models.email_digest import EmailDigest  # noqa: E402
from app.models.event import Event  # noqa: E402
from app.models.product import Product  # noqa: E402
from app.models.recommendation import Recommendation  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.security import create_access_token, hash_password  # noqa: E402
from app.vector_store.qdrant_client import get_vector_store  # noqa: E402
from app.vector_store.sync import sync_products  # noqa: E402


# --------------------------------------------------------------------------- #
# Session-scoped setup                                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Iterator[None]:
    """Create the schema once for the whole test session."""
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def app_instance() -> Any:
    """The FastAPI application under test."""
    return create_app()


@pytest.fixture()
def client(app_instance: Any) -> Iterator[TestClient]:
    """A ``TestClient`` with the app's lifespan executed."""
    with TestClient(app_instance) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Per-test isolation                                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Truncate every table and reset the caches between tests."""
    reset_cache_backend()
    _truncate_all()
    yield
    _truncate_all()
    reset_cache_backend()


def _truncate_all() -> None:
    """Delete all rows, children first to respect foreign keys."""
    with SessionLocal() as session:
        for model in (EmailDigest, Recommendation, Event, Product, User):
            session.execute(delete(model))
        session.commit()


@pytest.fixture()
def db() -> Iterator[Session]:
    """A plain database session for arranging and asserting test state."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Data factories                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def make_user(db: Session) -> Any:
    """Return a factory that creates users.

    Example:
        >>> user = make_user(email="a@b.c", role=UserRole.ADMIN)  # doctest: +SKIP
    """

    def factory(
        email: str = "learner@test.dev",
        password: str = "testpassword123",
        role: UserRole = UserRole.USER,
        full_name: str = "Test Learner",
    ) -> User:
        user = User(
            email=email,
            full_name=full_name,
            hashed_password=hash_password(password),
            role=role,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    return factory


@pytest.fixture()
def user(make_user: Any) -> User:
    """A standard (non-admin) user."""
    return make_user()


@pytest.fixture()
def admin(make_user: Any) -> User:
    """An admin user."""
    return make_user(email="admin@test.dev", role=UserRole.ADMIN, full_name="Test Admin")


def _auth_headers(target: User) -> dict[str, str]:
    """Build a Bearer header for a user."""
    token = create_access_token(target.id, role=target.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def auth_headers(user: User) -> dict[str, str]:
    """Bearer auth headers for the standard user."""
    return _auth_headers(user)


@pytest.fixture()
def admin_headers(admin: User) -> dict[str, str]:
    """Bearer auth headers for the admin user."""
    return _auth_headers(admin)


#: Small, deliberately lopsided catalog (with real generated covers, matching
#: what the seed script writes): three clearly agentic-AI courses plus
#: unrelated distractors, so retrieval quality is actually observable.
SAMPLE_PRODUCTS: list[dict[str, Any]] = [
    {
        "title": "Building Production Agents with LangGraph",
        "thumbnail_url": "/static/img/courses/agentic-ai.jpg",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 89.0,
        "duration": "14 hours",
        "rating": 4.9,
        "tags": "langgraph, agents, state machines, python",
        "description": (
            "Build agents as explicit state machines with conditional edges, checkpointing "
            "and refinement loops that retry retrieval when results are thin."
        ),
    },
    {
        "title": "Agentic RAG: Retrieval That Reasons",
        "thumbnail_url": "/static/img/courses/rag-retrieval.jpg",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "10 hours",
        "rating": 4.8,
        "tags": "rag, retrieval, agents, reranking",
        "description": (
            "Grade retrieved documents, rewrite queries when relevance is low, and fuse dense "
            "and keyword rankings for agentic retrieval augmented generation."
        ),
    },
    {
        "title": "Multi-Agent Systems: Coordination Patterns",
        "thumbnail_url": "/static/img/courses/multi-agent.jpg",
        "category": "Agentic AI",
        "skill_level": "advanced",
        "price": 129.0,
        "duration": "16 hours",
        "rating": 4.7,
        "tags": "multi-agent, orchestration, planning, agents",
        "description": (
            "Supervisor architectures, hierarchical delegation and message passing between "
            "cooperating agents that plan and critique their own output."
        ),
    },
    {
        "title": "Vector Databases and Semantic Search at Scale",
        "thumbnail_url": "/static/img/courses/vector-database.jpg",
        "category": "Data Engineering",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "10 hours",
        "rating": 4.8,
        "tags": "qdrant, vector database, embeddings, hybrid search",
        "description": (
            "Approximate nearest neighbour indexes, metadata filtering and hybrid dense plus "
            "sparse retrieval with reciprocal rank fusion over a vector database."
        ),
    },
    {
        "title": "Writing for Engineers: Design Docs and Postmortems",
        "thumbnail_url": "/static/img/courses/technical-writing.jpg",
        "category": "Career Skills",
        "skill_level": "beginner",
        "price": 35.0,
        "duration": "6 hours",
        "rating": 4.8,
        "tags": "writing, communication, design docs",
        "description": (
            "Structure a design document so reviewers engage with the decision, and run a "
            "blameless postmortem that produces real action items."
        ),
    },
    {
        "title": "Kubernetes for Application Teams",
        "thumbnail_url": "/static/img/courses/kubernetes.jpg",
        "category": "DevOps",
        "skill_level": "advanced",
        "price": 119.0,
        "duration": "21 hours",
        "rating": 4.6,
        "tags": "kubernetes, helm, scaling",
        "description": (
            "Deployments, rollout strategies, resource limits, probes and autoscaling for teams "
            "that need to run a service well without becoming platform engineers."
        ),
    },
]


@pytest.fixture()
def products(db: Session) -> list[Product]:
    """Insert the sample catalog into SQL *and* the embedded vector store."""
    store = get_vector_store()
    store.reset_collection()

    rows = [Product(**payload) for payload in SAMPLE_PRODUCTS]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)

    result = sync_products(rows)
    assert result.ok, f"vector sync failed during fixture setup: {result.error}"
    return rows


@pytest.fixture()
def make_events(db: Session) -> Any:
    """Return a factory that fabricates behavioural events for a user.

    The generated session is weighted toward the first product supplied, so tests
    can assert that the agent picks up the dominant signal.
    """
    from datetime import datetime, timedelta, timezone

    from app.models.event import EventType

    def factory(
        target_user: User,
        target_products: list[Product],
        *,
        count: int = 12,
        include_cart: bool = True,
        start: Any = None,
    ) -> list[Event]:
        """Create events.

        Args:
            target_user: Owner of the events.
            target_products: Catalog rows to reference.
            count: Number of click/dwell events (a search and a cart add are added
                on top, so the total is ``count + 2``).
            include_cart: Whether to append the high-intent cart event.
            start: Timestamp to begin from. Defaults to an hour ago; pass
                ``datetime.now(timezone.utc)`` when the events must be newer than a
                recommendation that was just generated.
        """
        cursor = start or (datetime.now(timezone.utc) - timedelta(hours=1))
        rows: list[Event] = []

        for index in range(count):
            product = target_products[index % max(1, min(2, len(target_products)))]
            cursor = cursor + timedelta(seconds=40)
            event_type = (
                EventType.PRODUCT_CLICK.value if index % 2 == 0 else EventType.TIME_SPENT.value
            )
            metadata: dict[str, Any] = {"product_title": product.title}
            if event_type == EventType.TIME_SPENT.value:
                metadata["seconds"] = 150
            else:
                metadata["source"] = "search"
            rows.append(
                Event(
                    user_id=target_user.id,
                    session_id="test-session",
                    event_type=event_type,
                    product_id=product.id,
                    path=f"/product/{product.id}",
                    metadata_json=metadata,
                    timestamp=cursor,
                )
            )

        cursor = cursor + timedelta(seconds=30)
        rows.append(
            Event(
                user_id=target_user.id,
                session_id="test-session",
                event_type=EventType.SEARCH_QUERY.value,
                path="/catalog",
                metadata_json={"query": "langgraph agents", "result_count": 3},
                timestamp=cursor,
            )
        )

        if include_cart:
            cursor = cursor + timedelta(seconds=25)
            rows.append(
                Event(
                    user_id=target_user.id,
                    session_id="test-session",
                    event_type=EventType.ADD_TO_CART.value,
                    product_id=target_products[0].id,
                    path=f"/product/{target_products[0].id}",
                    metadata_json={"product_title": target_products[0].title},
                    timestamp=cursor,
                )
            )

        db.add_all(rows)
        db.commit()
        return rows

    return factory
