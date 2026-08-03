"""SmartReco — a behavioural AI recommendation agent for a learning marketplace.

Package map
-----------
==========================  ===================================================
Module                      Responsibility
==========================  ===================================================
:mod:`app.main`             FastAPI factory, lifespan, health, error handling
:mod:`app.config`           Typed settings loaded from the environment
:mod:`app.database`         SQLAlchemy engine, sessions, declarative base
:mod:`app.security`         Password hashing and JWT issuing/verification
:mod:`app.cache`            Redis (or in-memory) locks and result cache
:mod:`app.models`           ORM models
:mod:`app.schemas`          Pydantic request/response models
:mod:`app.routers`          HTTP layer (pages + JSON API)
:mod:`app.agent`            LangGraph recommendation engine + Mesh gateway
:mod:`app.vector_store`     Embeddings, Qdrant, BM25, dual-write sync
:mod:`app.scheduler`        APScheduler jobs and digest delivery
==========================  ===================================================

★ Architectural invariant: every LLM and embedding call goes through the Mesh API
(:mod:`app.agent.mesh_client`).  No provider SDK is called directly anywhere.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
