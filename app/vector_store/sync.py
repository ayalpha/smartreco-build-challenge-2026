"""Dual-write orchestration: keep SQL and Qdrant perfectly in sync.

Contract
--------
SQL is the system of record; Qdrant is a derived mirror.  Every catalog mutation
funnels through this module so there is exactly one code path capable of writing
vectors:

===============  =================================================
SQL operation    Vector-side effect
===============  =================================================
``INSERT``       embed + upsert point, invalidate BM25 index
``UPDATE``       re-embed + upsert point (same id), invalidate BM25
``DELETE``       delete point by id, invalidate BM25
===============  =================================================

Vector failures never roll back a committed SQL write — the mirror is
self-healing via :func:`reindex_all` — but they are logged loudly and reported
back to the admin UI, and :func:`verify_sync` exposes drift for the health probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.product import Product
from app.vector_store.bm25 import BM25Document, get_index, invalidate_index
from app.vector_store.embeddings import embed_documents
from app.vector_store.qdrant_client import (
    RetrievedProduct,
    SearchFilters,
    get_vector_store,
)

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Outcome of a dual-write operation, surfaced to the admin UI."""

    ok: bool
    written: int = 0
    deleted: int = 0
    degraded_embeddings: bool = False
    error: Optional[str] = None

    @property
    def message(self) -> str:
        """Short human-readable status line."""
        if not self.ok:
            return f"Vector sync failed: {self.error}"
        if self.degraded_embeddings:
            return (
                f"Synced {self.written or self.deleted} item(s) to the vector store "
                "using fallback embeddings (Mesh unavailable)."
            )
        return f"Synced {self.written or self.deleted} item(s) to the vector store."


def _vector_payload(product: Product) -> dict[str, Any]:
    """Build the Qdrant payload mirror for a product row."""
    payload = product.to_public_dict()
    payload.update(
        product_id=product.id,
        revision=product.revision,
        is_active=product.is_active,
        document=product.embedding_text(),
        keyword_text=product.keyword_text(),
        updated_at=product.updated_at.isoformat() if product.updated_at else None,
    )
    # Qdrant keyword filters work on lists of strings, which `tags` already is.
    payload["tags"] = product.tag_list
    return payload


def sync_products(products: Sequence[Product]) -> SyncResult:
    """Embed and upsert a batch of products into Qdrant.

    Args:
        products: Freshly committed product rows.

    Returns:
        A :class:`SyncResult` describing what happened (never raises).
    """
    if not products:
        return SyncResult(ok=True, written=0)

    try:
        payloads = [_vector_payload(product) for product in products]
        embedding = embed_documents([payload["document"] for payload in payloads])
        written = get_vector_store().upsert_products(payloads, embedding)
        invalidate_index()
        return SyncResult(
            ok=True, written=written, degraded_embeddings=embedding.degraded
        )
    except Exception as exc:  # noqa: BLE001 - never let the mirror break the API
        logger.exception("Dual-write to Qdrant failed for %d product(s)", len(products))
        return SyncResult(ok=False, error=str(exc)[:500])


def sync_product(product: Product) -> SyncResult:
    """Embed and upsert a single product (convenience wrapper)."""
    return sync_products([product])


def remove_product(product_id: int) -> SyncResult:
    """Delete a product's vector so SQL deletes propagate to Qdrant.

    Args:
        product_id: The id of the product removed from SQL.

    Returns:
        A :class:`SyncResult` (never raises).
    """
    try:
        get_vector_store().delete_product(product_id)
        invalidate_index()
        return SyncResult(ok=True, deleted=1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to delete product %s from the vector store", product_id)
        return SyncResult(ok=False, error=str(exc)[:500])


def reindex_all(db: Session, *, reset: bool = False) -> SyncResult:
    """Re-embed and re-upsert the whole active catalog.

    This is the self-healing path: run it after a Qdrant outage, after switching
    embedding models, or once Mesh becomes reachable following a degraded seed.

    Args:
        db: Open SQLAlchemy session.
        reset: When True, drop and recreate the collection first (required when
            the embedding dimensionality changes).

    Returns:
        A :class:`SyncResult` for the whole batch.
    """
    if reset:
        try:
            get_vector_store().reset_collection()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not reset the Qdrant collection")
            return SyncResult(ok=False, error=str(exc)[:500])

    products = list(
        db.scalars(select(Product).where(Product.is_active.is_(True)).order_by(Product.id))
    )
    logger.info("Re-indexing %d active product(s) into the vector store", len(products))
    return sync_products(products)


def verify_sync(db: Session) -> dict[str, Any]:
    """Compare SQL and vector counts to expose drift.

    Returns:
        ``{"sql_count", "vector_count", "in_sync", "embedded_mode"}``.
    """
    sql_count = int(
        db.scalar(
            select(func.count())
            .select_from(Product)
            .where(Product.is_active.is_(True))
        )
        or 0
    )
    store = get_vector_store()
    vector_count = store.count()
    return {
        "sql_count": sql_count,
        "vector_count": vector_count,
        "in_sync": sql_count == vector_count,
        "embedded_mode": store.embedded_mode,
    }


# --------------------------------------------------------------------------- #
# Retrieval helpers (used by the agent's retrieval_node)                      #
# --------------------------------------------------------------------------- #

def _catalog_documents(db: Session) -> list[BM25Document]:
    """Load the active catalog as BM25 documents."""
    rows = db.scalars(select(Product).where(Product.is_active.is_(True))).all()
    return [BM25Document(doc_id=row.id, text=row.keyword_text()) for row in rows]


def keyword_search(db: Session, query: str, limit: int = 24) -> list[tuple[int, float]]:
    """Rank the catalog against ``query`` with BM25.

    Args:
        db: Open session (only used if the index needs rebuilding).
        query: Free-text query.
        limit: Max hits.

    Returns:
        ``[(product_id, score)]`` best first.
    """
    index = get_index(lambda: _catalog_documents(db))
    return index.search(query, limit=limit)


def sql_payload_hydrator(db: Session) -> Any:
    """Return a hydrator that fills payloads from SQL rather than Qdrant.

    Keyword-only hits are, by definition, absent from the dense result set; SQL
    already has the authoritative row, so this avoids a second vector round-trip.
    """

    def hydrate(product_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        rows = db.scalars(select(Product).where(Product.id.in_(list(product_ids)))).all()
        return {row.id: _vector_payload(row) for row in rows}

    return hydrate


def hybrid_retrieve(
    db: Session,
    query: str,
    *,
    limit: Optional[int] = None,
    filters: Optional[SearchFilters] = None,
) -> list[RetrievedProduct]:
    """Full hybrid retrieval: dense (Mesh + Qdrant) ⊕ BM25, fused with RRF.

    Args:
        db: Open session.
        query: The retrieval query written by the agent.
        limit: Max fused results.
        filters: Optional metadata constraints.

    Returns:
        Fused, filtered hits — best first.
    """
    embedding = embed_documents([query])
    query_vector = embedding.vectors[0] if embedding.vectors else []
    keyword_hits = keyword_search(db, query, limit=(limit or 12) * 2)

    return get_vector_store().hybrid_search(
        query=query,
        query_vector=query_vector,
        keyword_hits=keyword_hits,
        limit=limit,
        filters=filters,
        payload_hydrator=sql_payload_hydrator(db),
    )
