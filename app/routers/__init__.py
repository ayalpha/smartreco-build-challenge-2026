"""HTTP routers.

Each module exposes ``router`` (server-rendered pages / form posts) and, where
applicable, ``api_router`` (JSON endpoints under ``/api``).  They are wired up in
:func:`app.main.create_app`.
"""

from __future__ import annotations

from app.routers import admin, assistant, auth, cart, events, products, recommendations

__all__ = ["admin", "assistant", "auth", "cart", "events", "products", "recommendations"]
