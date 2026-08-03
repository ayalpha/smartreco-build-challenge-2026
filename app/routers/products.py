"""Public catalog: browse, search and product detail (HTML + JSON)."""

from __future__ import annotations

import logging
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.dependencies import get_current_user_optional, render_page
from app.models.product import SKILL_LEVELS, Product
from app.models.user import User
from app.schemas.common import OptionalPriceQuery
from app.schemas.product import ProductListResponse, ProductOut

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["catalog"])
api_router = APIRouter(prefix="/api/products", tags=["catalog"])

#: Catalog page size (kept small so the demo shows pagination working).
PAGE_SIZE = 12


def _apply_filters(
    statement: Select,
    *,
    query: Optional[str],
    category: Optional[str],
    skill_level: Optional[str],
    max_price: Optional[float],
) -> Select:
    """Apply catalog filters to a SELECT.

    All filtering goes through SQLAlchemy expressions — there is no string
    concatenation anywhere near the query, so the search box cannot be used for
    SQL injection.
    """
    statement = statement.where(Product.is_active.is_(True))

    if query:
        pattern = f"%{query.strip().lower()}%"
        statement = statement.where(
            or_(
                func.lower(Product.title).like(pattern),
                func.lower(Product.description).like(pattern),
                func.lower(Product.tags).like(pattern),
                func.lower(Product.category).like(pattern),
            )
        )
    if category:
        statement = statement.where(Product.category == category)
    if skill_level and skill_level in SKILL_LEVELS:
        statement = statement.where(Product.skill_level == skill_level)
    if max_price is not None:
        statement = statement.where(Product.price <= max_price)

    return statement


def _categories(db: Session) -> list[str]:
    """Distinct active categories, alphabetically."""
    rows = db.execute(
        select(Product.category)
        .where(Product.is_active.is_(True))
        .distinct()
        .order_by(Product.category)
    ).all()
    return [str(row[0]) for row in rows if row[0]]


def _count(db: Session, statement: Select) -> int:
    """Count rows matching a filtered SELECT."""
    subquery = statement.with_only_columns(Product.id).order_by(None).subquery()
    return int(db.scalar(select(func.count()).select_from(subquery)) or 0)


# --------------------------------------------------------------------------- #
# HTML pages                                                                  #
# --------------------------------------------------------------------------- #

@router.get("/catalog", include_in_schema=False)
def catalog_page(
    request: Request,
    q: Optional[str] = Query(default=None, max_length=200),
    category: Optional[str] = Query(default=None, max_length=80),
    skill_level: Optional[str] = Query(default=None, max_length=32),
    max_price: OptionalPriceQuery = None,
    page: int = Query(default=1, ge=1, le=500),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render the browsable, filterable catalog."""
    statement = _apply_filters(
        select(Product), query=q, category=category, skill_level=skill_level,
        max_price=max_price,
    )
    total = _count(db, statement)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(page, pages)

    products = list(
        db.scalars(
            statement.order_by(Product.rating.desc().nullslast(), Product.id.desc())
            .offset((page - 1) * PAGE_SIZE)
            .limit(PAGE_SIZE)
        )
    )

    return render_page(
        request,
        "catalog.html",
        user,
        products=products,
        categories=_categories(db),
        skill_levels=list(SKILL_LEVELS),
        total=total,
        page=page,
        pages=pages,
        filters={
            "q": q or "",
            "category": category or "",
            "skill_level": skill_level or "",
            "max_price": max_price,
        },
    )


@router.get("/product/{product_id}", include_in_schema=False)
def product_detail_page(
    product_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
) -> Response:
    """Render one product, plus a few related courses from the same category."""
    product = db.get(Product, product_id)
    if product is None or not product.is_active:
        return render_page(
            request, "404.html", user, status_code=status.HTTP_404_NOT_FOUND,
            missing=f"Course #{product_id}",
        )

    related = list(
        db.scalars(
            select(Product)
            .where(
                Product.is_active.is_(True),
                Product.category == product.category,
                Product.id != product.id,
            )
            .order_by(Product.rating.desc().nullslast())
            .limit(4)
        )
    )

    return render_page(request, "product_detail.html", user, product=product, related=related)


# --------------------------------------------------------------------------- #
# JSON API                                                                    #
# --------------------------------------------------------------------------- #

@api_router.get("", response_model=ProductListResponse)
def list_products(
    q: Optional[str] = Query(default=None, max_length=200),
    category: Optional[str] = Query(default=None, max_length=80),
    skill_level: Optional[str] = Query(default=None, max_length=32),
    max_price: OptionalPriceQuery = None,
    page: int = Query(default=1, ge=1, le=500),
    page_size: int = Query(default=PAGE_SIZE, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ProductListResponse:
    """Paginated, filterable catalog listing."""
    statement = _apply_filters(
        select(Product), query=q, category=category, skill_level=skill_level,
        max_price=max_price,
    )
    total = _count(db, statement)
    rows = list(
        db.scalars(
            statement.order_by(Product.rating.desc().nullslast(), Product.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )

    return ProductListResponse(
        items=[ProductOut.model_validate(row.to_public_dict()) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        query=q,
        category=category,
    )


@api_router.get("/categories", response_model=list[str])
def list_categories(db: Session = Depends(get_db)) -> list[str]:
    """Distinct active categories."""
    return _categories(db)


@api_router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)) -> ProductOut:
    """Fetch a single active product."""
    product = db.get(Product, product_id)
    if product is None or not product.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Product {product_id} not found"
        )
    return ProductOut.model_validate(product.to_public_dict())
