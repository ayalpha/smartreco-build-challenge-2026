"""Cart and checkout — UI-only commerce flow.

Deliberately **not** a payment integration. There is no Stripe SDK, no charge,
no card data anywhere on the server: the checkout form validates client-side,
the "Pay" action renders a styled confirmation, and nothing is transmitted or
stored. The card fields never leave the browser — the form has no ``action`` and
its submit handler calls ``preventDefault``.

Cart state lives in ``localStorage`` (see ``static/js/cart.js``) so it survives
navigation with no backend session work. These routes only render the shells;
the item list, totals and validation are hydrated in the browser.

The one server-side touch is behavioural: adding to the cart also emits the
existing ``add_to_cart`` event via the saved-items endpoint, so commerce intent
still feeds the recommendation agent rather than bypassing it.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user_optional, render_page
from app.models.product import Product
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cart"])


def _catalog_index(db: Session) -> list[dict[str, object]]:
    """Serialise the active catalog for client-side cart hydration.

    The cart stores only ids and quantities; titles, prices and covers are
    resolved from this index so a price change is never stale in a saved cart.
    """
    rows = db.scalars(select(Product).where(Product.is_active.is_(True))).all()
    return [
        {
            "id": row.id,
            "title": row.title,
            "price": row.price,
            "category": row.category,
            "skill_level": row.skill_level,
            "duration": row.duration,
            "instructor": row.instructor,
            "thumbnail_url": row.thumbnail_url,
        }
        for row in rows
    ]


@router.get("/cart", include_in_schema=False)
def cart_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the cart. Items are hydrated from localStorage by cart.js."""
    return render_page(request, "cart.html", user, catalog=_catalog_index(db))


@router.get("/checkout", include_in_schema=False)
def checkout_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the checkout form (UI-only — no payment processing)."""
    return render_page(request, "checkout.html", user, catalog=_catalog_index(db))


@router.get("/checkout/success", include_in_schema=False)
def checkout_success_page(
    request: Request,
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the order-confirmation screen."""
    return render_page(request, "checkout_success.html", user)
