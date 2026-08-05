"""FastAPI application factory and lifespan management.

Start-up sequence
-----------------
1. configure logging and log a redacted settings snapshot;
2. ensure the database schema exists (Alembic owns production migrations, this is
   the dev/test convenience path);
3. configure LangSmith tracing (BONUS 3);
4. ensure the Qdrant collection exists;
5. start APScheduler (BONUS 2 digest + the agent's background queue).

Every step is individually fault-tolerant: a missing Qdrant server or Redis must
not stop the web app from serving, because the retrieval and cache layers both
degrade gracefully by design.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agent.mesh_client import describe_models, mesh_available
from app.agent.observability import configure_langsmith, observability_status
from app.config import get_settings
from app.database import healthcheck as db_healthcheck
from app.database import init_db
from app.dependencies import get_current_user_optional, render_page
from app.logging_config import configure_logging
from app.models.user import User
from app.routers import admin, assistant, auth, cart, events, products, recommendations
from app.scheduler.jobs import scheduler_status, shutdown_scheduler, start_scheduler
from app.vector_store.qdrant_client import get_vector_store

settings = get_settings()
configure_logging()
logger = logging.getLogger(__name__)

DESCRIPTION = """\
**Nexora** is a behavioural AI recommendation agent for a learning marketplace,
built by AY Systum for the SmartReco Build Challenge 2026.

It watches what a learner does, understands what they are pursuing, and writes a
persuasive, evidence-grounded recommendation — via a seven-node LangGraph state
machine with hybrid (dense ⊕ BM25) retrieval over Qdrant.

Every LLM and embedding call is routed through the **Mesh API** gateway.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage start-up and shutdown of every subsystem."""
    logger.info("=" * 78)
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)
    logger.info("=" * 78)
    logger.debug("Effective settings: %s", settings.safe_dump())

    try:
        init_db()
    except Exception:
        logger.exception("Database initialisation failed — the app may not work correctly")

    configure_langsmith()

    try:
        get_vector_store().ensure_collection()
        _rehydrate_embedded_vectors()
    except Exception:
        logger.warning(
            "Could not prepare the Qdrant collection at start-up; retrieval will "
            "retry on demand.", exc_info=True,
        )

    if mesh_available():
        logger.info("Mesh API configured — models: %s", describe_models())
    else:
        logger.warning(
            "MESH_API_KEY is not set. The agent will run in degraded (heuristic) "
            "mode: everything works, but the copy is templated rather than written."
        )

    try:
        start_scheduler()
    except Exception:
        logger.exception("Scheduler failed to start — digests and queued runs are off")

    yield

    logger.info("Shutting down %s", settings.app_name)
    shutdown_scheduler(wait=False)
    from app.agent.runner import shutdown_executor

    shutdown_executor(wait=False)


def _rehydrate_embedded_vectors() -> None:
    """Rebuild the vector index at start-up when it is empty but SQL is not.

    The embedded Qdrant fallback lives in process memory, so it starts empty on
    every boot — including after a seed script populated SQL in a *different*
    process.  Without this, a zero-infrastructure run would come up with a catalog
    the agent cannot retrieve from.  Re-indexing is idempotent and only runs when
    there is an actual gap to close, so it is a no-op against a real Qdrant server
    whose storage already persists.
    """
    from app.database import session_scope
    from app.vector_store.sync import reindex_all, verify_sync

    store = get_vector_store()
    with session_scope() as db:
        status = verify_sync(db)
        if status["vector_count"] > 0 or status["sql_count"] == 0:
            logger.info(
                "Vector store ready — %s SQL row(s), %s vector(s)",
                status["sql_count"], status["vector_count"],
            )
            return

        logger.info(
            "Vector index is empty but SQL holds %s product(s) — re-indexing "
            "(%s Qdrant).", status["sql_count"],
            "embedded" if store.embedded_mode else "server",
        )
        result = reindex_all(db)
        if result.ok:
            logger.info("Re-indexed %s product(s) at start-up", result.written)
        else:
            logger.error("Start-up re-index failed: %s", result.error)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        The wired-up application instance.
    """
    app = FastAPI(
        title=f"{settings.app_name} — Behavioural AI Recommendation Agent",
        description=DESCRIPTION,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # --- server-rendered pages -------------------------------------------
    app.include_router(recommendations.router)
    app.include_router(products.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(assistant.router)
    app.include_router(cart.router)

    # --- JSON API ---------------------------------------------------------
    app.include_router(auth.api_router)
    app.include_router(products.api_router)
    app.include_router(events.router)
    app.include_router(recommendations.api_router)
    app.include_router(admin.api_router)
    app.include_router(assistant.api_router)

    _register_error_handlers(app)
    _register_system_routes(app)

    logger.info("Application configured with %d routes", len(app.routes))
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Install handlers that render HTML for browsers and JSON for API clients."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(  # type: ignore[misc]
        request: Request, exc: StarletteHTTPException
    ) -> Response:
        """Redirect browser auth failures, otherwise return JSON or an error page."""
        location = (exc.headers or {}).get("Location")
        if exc.status_code in (status.HTTP_303_SEE_OTHER, status.HTTP_307_TEMPORARY_REDIRECT) and location:
            return RedirectResponse(url=location, status_code=status.HTTP_303_SEE_OTHER)

        wants_html = "text/html" in request.headers.get(
            "accept", ""
        ) and not request.url.path.startswith("/api/")

        if exc.status_code == status.HTTP_404_NOT_FOUND and wants_html:
            return render_page(
                request, "404.html", None, status_code=404, missing=request.url.path
            )

        if wants_html and exc.status_code >= 500:
            return render_page(
                request, "500.html", None, status_code=exc.status_code,
                detail=str(exc.detail),
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(  # type: ignore[misc]
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a compact, JSON-safe 422 payload.

        ``exc.errors()`` is *not* directly serialisable: when a validator raises
        ``ValueError`` (as the ``skill_level`` validator does), Pydantic puts the
        exception object itself in ``ctx``, and ``json.dumps`` then fails — turning
        a 422 into a 500. So the errors are projected onto plain strings here, which
        also avoids echoing internal context back to the client.
        """
        cleaned = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": str(error.get("msg", "invalid value")),
                "type": str(error.get("type", "value_error")),
            }
            for error in exc.errors()
        ]
        logger.info(
            "Validation error on %s %s: %s", request.method, request.url.path, cleaned[:3]
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": cleaned}
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(  # type: ignore[misc]
        request: Request, exc: Exception
    ) -> Response:
        """Log the traceback and return a safe error response."""
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        if "text/html" in request.headers.get("accept", "") and not request.url.path.startswith("/api/"):
            return render_page(
                request, "500.html", None, status_code=500,
                detail=str(exc) if settings.debug else None,
            )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error": str(exc) if settings.debug else None,
            },
        )


def _register_system_routes(app: FastAPI) -> None:
    """Add health, readiness and agent-introspection endpoints."""

    @app.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        """Liveness probe.

        Deliberately *not* traced to LangSmith — health checks would otherwise
        dominate the trace view (selective tracing, BONUS 3).
        """
        return {"status": "ok", "app": settings.app_name, "environment": settings.environment}

    @app.get("/health/ready", tags=["system"])
    def readiness() -> JSONResponse:
        """Readiness probe reporting every dependency's state."""
        store = get_vector_store()
        checks = {
            "database": db_healthcheck(),
            "vector_store": store.is_healthy(),
            "mesh_configured": mesh_available(),
        }
        payload: dict[str, Any] = {
            "status": "ready" if checks["database"] else "degraded",
            "checks": checks,
            "vector_store_mode": "embedded" if store.embedded_mode else "server",
            "scheduler": scheduler_status(),
            "observability": observability_status(),
            "models": describe_models(),
        }
        code = status.HTTP_200_OK if checks["database"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=code, content=payload)

    @app.get("/api/agent/graph", tags=["system"])
    def agent_graph() -> dict[str, Any]:
        """Return the compiled graph topology (ASCII + Mermaid) for inspection."""
        from app.agent.graph import render_ascii, render_mermaid

        return {
            "nodes": [
                "activity_analyzer",
                "interest_extractor",
                "retrieval_node",
                "relevance_grader",
                "retrieval_refiner",
                "persuasion_writer",
                "recommendation_storer",
            ],
            "conditional_edges": {
                "relevance_grader": {
                    f">= {settings.agent_min_relevant_products} relevant": "persuasion_writer",
                    f"< {settings.agent_min_relevant_products} and retries "
                    f"< {settings.agent_max_retrieval_retries}": "retrieval_refiner",
                    "retry budget exhausted": "persuasion_writer",
                }
            },
            "ascii": render_ascii(),
            "mermaid": render_mermaid(),
        }

    @app.get("/architecture", include_in_schema=False)
    def architecture_page(
        request: Request,
        user: Optional[User] = Depends(get_current_user_optional),
    ) -> Response:
        """Render a page explaining the agent architecture (great for demos)."""
        from app.agent.graph import render_mermaid

        return render_page(
            request,
            "architecture.html",
            user,
            mermaid=render_mermaid(),
            models=describe_models(),
            observability=observability_status(),
            scheduler=scheduler_status(),
        )


#: ASGI entry point — ``uvicorn app.main:app --reload``
app = create_app()
