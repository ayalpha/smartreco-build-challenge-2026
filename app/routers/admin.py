"""Admin catalog management with strict SQL ⇄ Qdrant dual-write.

Every mutation here follows the same shape:

1. write SQL and commit (system of record);
2. mirror the change into Qdrant via :mod:`app.vector_store.sync`;
3. report the vector outcome to the admin.

A vector failure never rolls back the SQL write — the mirror is repairable with
the "Re-index catalog" action — but it is always surfaced rather than swallowed,
so drift can't accumulate silently.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.mesh_client import describe_models, mesh_available
from app.agent.observability import observability_status
from app.config import get_settings
from app.database import get_db
from app.dependencies import render_page, require_admin
from app.flash import flash
from app.models.event import Event
from app.models.product import SKILL_LEVELS, Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.schemas.common import OptionalRatingForm
from app.schemas.product import ProductCreate, ProductOut, ProductUpdate
from app.vector_store.qdrant_client import get_vector_store
from app.vector_store.sync import reindex_all, remove_product, sync_product, verify_sync

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin", tags=["admin"])
api_router = APIRouter(prefix="/api/admin", tags=["admin"])


# --------------------------------------------------------------------------- #
# Shared write helpers                                                        #
# --------------------------------------------------------------------------- #

def _apply_payload(product: Product, payload: dict[str, Any]) -> None:
    """Copy validated fields onto a product row, normalising tags."""
    for field, value in payload.items():
        if field == "tags":
            product.tags = Product.normalise_tags(value)
        elif hasattr(product, field):
            setattr(product, field, value)


def _create_product(db: Session, payload: ProductCreate) -> tuple[Product, Any]:
    """Insert a product into SQL, then mirror it into Qdrant.

    Returns:
        ``(product, sync_result)``.
    """
    product = Product()
    _apply_payload(product, payload.model_dump())
    db.add(product)
    db.commit()
    db.refresh(product)

    result = sync_product(product)
    logger.info("Created product id=%s (vector sync ok=%s)", product.id, result.ok)
    return product, result


def _update_product(db: Session, product: Product, payload: ProductUpdate) -> tuple[Product, Any]:
    """Apply a partial update in SQL, bump the revision, re-embed and upsert."""
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    # `is_active` is meaningful when explicitly set to False, so re-add it.
    if "is_active" in payload.model_fields_set:
        changes["is_active"] = payload.is_active

    _apply_payload(product, changes)
    product.revision = (product.revision or 1) + 1
    db.commit()
    db.refresh(product)

    if product.is_active:
        result = sync_product(product)
    else:
        # Deactivated products must leave the retrievable index entirely.
        result = remove_product(product.id)

    logger.info(
        "Updated product id=%s fields=%s (vector sync ok=%s)",
        product.id, sorted(changes), result.ok,
    )
    return product, result


def _delete_product(db: Session, product: Product) -> Any:
    """Delete from SQL and Qdrant, in that order."""
    product_id = product.id
    db.delete(product)
    db.commit()
    result = remove_product(product_id)
    logger.info("Deleted product id=%s (vector delete ok=%s)", product_id, result.ok)
    return result


def _dashboard_stats(db: Session) -> dict[str, Any]:
    """Aggregate the numbers shown on the admin dashboard."""
    def scalar_count(model: Any, *conditions: Any) -> int:
        statement = select(func.count()).select_from(model)
        for condition in conditions:
            statement = statement.where(condition)
        return int(db.scalar(statement) or 0)

    top_categories = [
        {"category": str(row[0]), "count": int(row[1])}
        for row in db.execute(
            select(Product.category, func.count(Product.id))
            .where(Product.is_active.is_(True))
            .group_by(Product.category)
            .order_by(func.count(Product.id).desc())
            .limit(8)
        ).all()
    ]

    event_breakdown = [
        {"event_type": str(row[0]), "count": int(row[1])}
        for row in db.execute(
            select(Event.event_type, func.count(Event.id))
            .group_by(Event.event_type)
            .order_by(func.count(Event.id).desc())
        ).all()
    ]

    store = get_vector_store()
    return {
        "products": scalar_count(Product),
        "active_products": scalar_count(Product, Product.is_active.is_(True)),
        "users": scalar_count(User),
        "events": scalar_count(Event),
        "recommendations": scalar_count(Recommendation),
        "active_recommendations": scalar_count(
            Recommendation, Recommendation.is_active.is_(True)
        ),
        "degraded_recommendations": scalar_count(
            Recommendation, Recommendation.degraded.is_(True)
        ),
        "top_categories": top_categories,
        "event_breakdown": event_breakdown,
        "vector": verify_sync(db),
        "vector_embedded_mode": store.embedded_mode,
        "mesh": {"configured": mesh_available(), **describe_models()},
        "langsmith": observability_status(),
    }


# --------------------------------------------------------------------------- #
# HTML: dashboard + forms                                                     #
# --------------------------------------------------------------------------- #

@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Render the admin dashboard: catalog table, stats and sync status."""
    products = list(db.scalars(select(Product).order_by(Product.id.desc()).limit(200)))
    recent_recommendations = list(
        db.scalars(
            select(Recommendation)
            .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
            .limit(10)
        )
    )
    return render_page(
        request,
        "admin/dashboard.html",
        admin,
        products=products,
        stats=_dashboard_stats(db),
        recent_recommendations=recent_recommendations,
    )


@router.get("/products/new", include_in_schema=False)
def new_product_form(
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Render the blank product form."""
    categories = [
        str(row[0])
        for row in db.execute(select(Product.category).distinct().order_by(Product.category)).all()
        if row[0]
    ]
    return render_page(
        request,
        "admin/product_form.html",
        admin,
        product=None,
        categories=categories,
        skill_levels=list(SKILL_LEVELS),
        action="/admin/products",
    )


@router.get("/products/{product_id}/edit", include_in_schema=False)
def edit_product_form(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Render the product form pre-filled for editing."""
    product = db.get(Product, product_id)
    if product is None:
        response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
        flash(response, f"Course #{product_id} no longer exists.", "error")
        return response

    categories = [
        str(row[0])
        for row in db.execute(select(Product.category).distinct().order_by(Product.category)).all()
        if row[0]
    ]
    return render_page(
        request,
        "admin/product_form.html",
        admin,
        product=product,
        categories=categories,
        skill_levels=list(SKILL_LEVELS),
        action=f"/admin/products/{product.id}",
    )


def _form_payload(
    title: str,
    description: str,
    category: str,
    tags: str,
    price: float,
    skill_level: str,
    duration: str,
    thumbnail_url: str,
    instructor: str,
    rating: Optional[float],
    is_active: bool,
) -> dict[str, Any]:
    """Normalise raw form fields into a dict for Pydantic validation."""
    return {
        "title": title.strip(),
        "description": description.strip(),
        "category": category.strip(),
        "tags": tags,
        "price": price,
        "skill_level": skill_level,
        "duration": duration.strip() or None,
        "thumbnail_url": thumbnail_url.strip() or None,
        "instructor": instructor.strip() or None,
        "rating": rating,
        "is_active": is_active,
    }


@router.post("/products", include_in_schema=False)
def create_product_form(
    title: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    tags: str = Form(""),
    price: float = Form(0.0),
    skill_level: str = Form("beginner"),
    duration: str = Form(""),
    thumbnail_url: str = Form(""),
    instructor: str = Form(""),
    rating: OptionalRatingForm = None,
    is_active: bool = Form(True),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Handle the create-product form (SQL + vector dual-write)."""
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    try:
        payload = ProductCreate.model_validate(
            _form_payload(title, description, category, tags, price, skill_level,
                          duration, thumbnail_url, instructor, rating, is_active)
        )
    except ValidationError as exc:
        flash(response, f"Could not save the course — {_first_error(exc)}", "error")
        return response

    product, result = _create_product(db, payload)
    flash(
        response,
        f"Created “{product.title}”. {result.message}",
        "success" if result.ok else "warning",
    )
    return response


@router.post("/products/{product_id}", include_in_schema=False)
def update_product_form(
    product_id: int,
    title: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    tags: str = Form(""),
    price: float = Form(0.0),
    skill_level: str = Form("beginner"),
    duration: str = Form(""),
    thumbnail_url: str = Form(""),
    instructor: str = Form(""),
    rating: OptionalRatingForm = None,
    is_active: bool = Form(False),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Handle the edit-product form (SQL + vector dual-write)."""
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    product = db.get(Product, product_id)
    if product is None:
        flash(response, f"Course #{product_id} no longer exists.", "error")
        return response

    try:
        payload = ProductUpdate.model_validate(
            _form_payload(title, description, category, tags, price, skill_level,
                          duration, thumbnail_url, instructor, rating, is_active)
        )
    except ValidationError as exc:
        flash(response, f"Could not save the course — {_first_error(exc)}", "error")
        return response

    product, result = _update_product(db, product, payload)
    flash(
        response,
        f"Updated “{product.title}”. {result.message}",
        "success" if result.ok else "warning",
    )
    return response


@router.post("/products/{product_id}/delete", include_in_schema=False)
def delete_product_form(
    product_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Handle the delete action (removes the row *and* its vector)."""
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    product = db.get(Product, product_id)
    if product is None:
        flash(response, f"Course #{product_id} no longer exists.", "error")
        return response

    title = product.title
    result = _delete_product(db, product)
    flash(
        response,
        f"Deleted “{title}” from SQL and the vector store. {result.message}",
        "success" if result.ok else "warning",
    )
    return response


@router.post("/reindex", include_in_schema=False)
def reindex_form(
    reset: bool = Form(False),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> Response:
    """Re-embed and re-upsert the whole catalog (self-heal the vector mirror)."""
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    result = reindex_all(db, reset=reset)
    flash(
        response,
        f"Re-indexed the catalog. {result.message}",
        "success" if result.ok else "error",
    )
    return response


@router.post("/digest/run", include_in_schema=False)
def run_digest_form(admin: User = Depends(require_admin)) -> Response:
    """Run the daily digest job immediately (BONUS 2 manual trigger)."""
    from app.scheduler.jobs import run_daily_digest_now

    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    try:
        summary = run_daily_digest_now()
        flash(
            response,
            f"Digest job finished: {summary['sent']} sent, {summary['skipped']} skipped, "
            f"{summary['failed']} failed.",
            "success" if summary["failed"] == 0 else "warning",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual digest run failed")
        flash(response, f"Digest job failed: {exc}", "error")
    return response


def _first_error(exc: ValidationError) -> str:
    """Render the first validation error as a short human sentence."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
    return f"{location or 'input'}: {error.get('msg')}"


# --------------------------------------------------------------------------- #
# JSON API                                                                    #
# --------------------------------------------------------------------------- #

@api_router.get("/stats", response_model=dict)
def api_stats(
    db: Session = Depends(get_db), admin: User = Depends(require_admin)
) -> dict[str, Any]:
    """Return dashboard statistics as JSON."""
    return _dashboard_stats(db)


@api_router.post("/products", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def api_create_product(
    payload: ProductCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> ProductOut:
    """Create a product (SQL + vector dual-write)."""
    product, result = _create_product(db, payload)
    if not result.ok:
        logger.error("Vector mirror failed for new product %s: %s", product.id, result.error)
    return ProductOut.model_validate(product.to_public_dict())


@api_router.patch("/products/{product_id}", response_model=ProductOut)
def api_update_product(
    product_id: int,
    payload: ProductUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> ProductOut:
    """Partially update a product (SQL + vector dual-write)."""
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Product {product_id} not found"
        )
    product, result = _update_product(db, product, payload)
    if not result.ok:
        logger.error("Vector mirror failed for product %s: %s", product_id, result.error)
    return ProductOut.model_validate(product.to_public_dict())


@api_router.delete("/products/{product_id}", status_code=status.HTTP_200_OK)
def api_delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    """Delete a product from SQL **and** the vector store."""
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Product {product_id} not found"
        )
    result = _delete_product(db, product)
    return {
        "deleted": True,
        "product_id": product_id,
        "vector_deleted": result.ok,
        "detail": result.message,
    }


@api_router.post("/reindex", response_model=dict)
def api_reindex(
    reset: bool = False,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> dict[str, Any]:
    """Re-index the catalog into the vector store."""
    result = reindex_all(db, reset=reset)
    return {
        "ok": result.ok,
        "written": result.written,
        "degraded_embeddings": result.degraded_embeddings,
        "detail": result.message,
        "sync": verify_sync(db),
    }
