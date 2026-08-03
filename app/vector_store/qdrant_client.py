"""Qdrant vector store: dual-write target and retrieval engine.

Responsibilities
----------------
* **Collection lifecycle** — create the collection with the configured
  dimensionality and the payload indexes needed for metadata filtering.
* **Dual-write primitives** — :meth:`VectorStore.upsert_products` and
  :meth:`VectorStore.delete_product`, called from the catalog routers so SQL and
  vectors never diverge.
* **Retrieval (BONUS 4)** — dense search with metadata filters, plus
  :meth:`VectorStore.hybrid_search` which fuses dense results with an in-process
  BM25 ranking via Reciprocal Rank Fusion.

Availability
------------
If the configured Qdrant server is unreachable, the client transparently falls
back to ``qdrant-client``'s embedded local mode (``:memory:``).  The API surface
is identical, so the app, the tests and CI all work with no Docker daemon — while
a real Qdrant is used the moment ``QDRANT_URL`` is live.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.vector_store.embeddings import EmbeddingResult

logger = logging.getLogger(__name__)
settings = get_settings()

#: Payload fields that get a Qdrant index so filters stay fast.
_KEYWORD_INDEX_FIELDS = ("category", "skill_level", "tags")
_NUMERIC_INDEX_FIELDS = ("price", "product_id", "revision")

#: RRF constant from Cormack et al. (2009); 60 is the canonical default.
RRF_K = 60


@dataclass
class RetrievedProduct:
    """One retrieval hit with its provenance and fused score."""

    product_id: int
    payload: dict[str, Any]
    dense_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    dense_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    retrieval_mode: str = "dense"

    def as_dict(self) -> dict[str, Any]:
        """Flatten into the plain dict shape the agent state carries around."""
        data = dict(self.payload)
        data.update(
            id=self.product_id,
            dense_score=round(self.dense_score, 5),
            keyword_score=round(self.keyword_score, 5),
            fused_score=round(self.fused_score, 5),
            dense_rank=self.dense_rank,
            keyword_rank=self.keyword_rank,
            retrieval_mode=self.retrieval_mode,
        )
        return data


@dataclass
class SearchFilters:
    """Optional metadata constraints applied to dense retrieval (BONUS 4).

    Populated by the agent from inferred behaviour: a user who only ever opens
    beginner courses should not be pitched an advanced one, and a user who never
    clicks anything above $50 should not be shown a $400 bootcamp.
    """

    skill_levels: Optional[Sequence[str]] = None
    categories: Optional[Sequence[str]] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    exclude_product_ids: Optional[Sequence[int]] = None

    def describe(self) -> dict[str, Any]:
        """Human/JSON readable summary for the agent trace."""
        return {
            "skill_levels": list(self.skill_levels) if self.skill_levels else None,
            "categories": list(self.categories) if self.categories else None,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "excluded": len(self.exclude_product_ids or []),
        }

    def is_empty(self) -> bool:
        """True when no constraint at all is set."""
        return not any(
            [
                self.skill_levels,
                self.categories,
                self.max_price is not None,
                self.min_price is not None,
                self.exclude_product_ids,
            ]
        )

    def to_qdrant_filter(self) -> Optional[qmodels.Filter]:
        """Translate into a Qdrant ``Filter``, or None when empty."""
        must: list[Any] = []
        must_not: list[Any] = []

        if self.skill_levels:
            must.append(
                qmodels.FieldCondition(
                    key="skill_level",
                    match=qmodels.MatchAny(any=[s.lower() for s in self.skill_levels]),
                )
            )
        if self.categories:
            must.append(
                qmodels.FieldCondition(
                    key="category", match=qmodels.MatchAny(any=list(self.categories))
                )
            )
        if self.max_price is not None or self.min_price is not None:
            must.append(
                qmodels.FieldCondition(
                    key="price",
                    range=qmodels.Range(gte=self.min_price, lte=self.max_price),
                )
            )
        if self.exclude_product_ids:
            must_not.append(
                qmodels.FieldCondition(
                    key="product_id",
                    match=qmodels.MatchAny(any=list(self.exclude_product_ids)),
                )
            )

        if not must and not must_not:
            return None
        return qmodels.Filter(must=must or None, must_not=must_not or None)


class VectorStore:
    """Thin, well-behaved wrapper around a Qdrant collection."""

    def __init__(self, collection: Optional[str] = None) -> None:
        self.collection = collection or settings.qdrant_collection
        self._client: Optional[QdrantClient] = None
        self._embedded_mode = False
        self._collection_ready = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------ plumbing
    @property
    def embedded_mode(self) -> bool:
        """True when running against the in-process fallback instead of a server."""
        return self._embedded_mode

    def client(self) -> QdrantClient:
        """Return a connected client, falling back to embedded mode if needed."""
        with self._lock:
            if self._client is not None:
                return self._client

            try:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=10.0,
                    prefer_grpc=False,
                )
                client.get_collections()  # fail fast on an unreachable server
                logger.info("Connected to Qdrant server at %s", settings.qdrant_url)
                self._client = client
                self._embedded_mode = False
            except Exception as exc:
                logger.warning(
                    "Qdrant server at %s unreachable (%s) — using embedded in-memory "
                    "Qdrant. Start it with: docker run -p 6333:6333 qdrant/qdrant",
                    settings.qdrant_url, exc,
                )
                self._client = QdrantClient(location=":memory:")
                self._embedded_mode = True
            return self._client

    def is_healthy(self) -> bool:
        """True when the store answers a trivial request."""
        try:
            self.client().get_collections()
            return True
        except Exception:
            logger.warning("Qdrant healthcheck failed", exc_info=True)
            return False

    def ensure_collection(self, dimension: Optional[int] = None) -> None:
        """Create the collection and payload indexes if they do not exist.

        Idempotent — safe to call on every write path and at start-up.

        Args:
            dimension: Vector size.  Defaults to ``MESH_EMBEDDING_DIM``.
        """
        with self._lock:
            if self._collection_ready:
                return

            size = dimension or settings.mesh_embedding_dim
            client = self.client()

            try:
                exists = client.collection_exists(self.collection)
            except Exception:  # pragma: no cover - very old client builds
                exists = self.collection in {
                    c.name for c in client.get_collections().collections
                }

            if not exists:
                logger.info("Creating Qdrant collection %r (dim=%d, cosine)",
                            self.collection, size)
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=size, distance=qmodels.Distance.COSINE
                    ),
                )

            self._create_payload_indexes()
            self._collection_ready = True

    def _create_payload_indexes(self) -> None:
        """Create payload indexes used by metadata filtering (best-effort)."""
        if self._embedded_mode:
            # Local/embedded Qdrant filters by brute force; payload indexes are a
            # no-op there and only emit warnings.
            logger.debug("Skipping payload indexes: embedded Qdrant does not use them")
            return

        client = self.client()
        for field in _KEYWORD_INDEX_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug("Payload index for %r already present or unsupported", field)
        for field in _NUMERIC_INDEX_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.FLOAT
                    if field == "price"
                    else qmodels.PayloadSchemaType.INTEGER,
                )
            except Exception:
                logger.debug("Payload index for %r already present or unsupported", field)

    def reset_collection(self) -> None:
        """Drop and recreate the collection (used by ``--reindex`` and tests)."""
        client = self.client()
        try:
            client.delete_collection(self.collection)
            logger.info("Dropped Qdrant collection %r", self.collection)
        except Exception:
            logger.debug("Collection %r did not exist", self.collection)
        with self._lock:
            self._collection_ready = False
        self.ensure_collection()

    # -------------------------------------------------------------- writes
    def upsert_products(
        self,
        products: Sequence[dict[str, Any]],
        embedding: EmbeddingResult,
    ) -> int:
        """Upsert product mirrors into Qdrant (the vector half of the dual-write).

        Args:
            products: Payload dicts, each containing at least ``id``.
            embedding: Vectors + provenance, aligned index-wise with ``products``.

        Returns:
            The number of points written.

        Raises:
            ValueError: If the vector count does not match the product count.
        """
        if not products:
            return 0
        if len(products) != len(embedding.vectors):
            raise ValueError(
                f"Vector/product length mismatch: {len(embedding.vectors)} vectors "
                f"for {len(products)} products"
            )

        self.ensure_collection(dimension=embedding.dimension)

        points: list[qmodels.PointStruct] = []
        for product, vector in zip(products, embedding.vectors):
            payload = dict(product)
            payload["product_id"] = int(product["id"])
            # Auditability requirement: stamp the embedding model + version.
            payload.update(embedding.provenance)
            points.append(
                qmodels.PointStruct(
                    id=int(product["id"]), vector=list(vector), payload=payload
                )
            )

        self.client().upsert(collection_name=self.collection, points=points, wait=True)
        logger.info(
            "Upserted %d point(s) into %r using model=%s (degraded=%s)",
            len(points), self.collection, embedding.model, embedding.degraded,
        )
        return len(points)

    def delete_product(self, product_id: int) -> None:
        """Remove a product's vector — the delete half of the dual-write."""
        self.ensure_collection()
        self.client().delete(
            collection_name=self.collection,
            points_selector=qmodels.PointIdsList(points=[int(product_id)]),
            wait=True,
        )
        logger.info("Deleted point %s from %r", product_id, self.collection)

    def count(self) -> int:
        """Number of points currently stored (0 when the collection is absent)."""
        try:
            self.ensure_collection()
            return int(self.client().count(self.collection, exact=True).count)
        except Exception:
            logger.warning("Could not count points in %r", self.collection, exc_info=True)
            return 0

    # -------------------------------------------------------------- search
    def _query_points(
        self,
        query_vector: list[float],
        limit: int,
        query_filter: Optional[qmodels.Filter],
    ) -> list[Any]:
        """Run an ANN query, supporting both modern and legacy client APIs.

        ``qdrant-client`` 1.10 introduced ``query_points`` and 1.16 removed the
        old ``search`` method, so both are probed here rather than pinning the
        client to a narrow version range.

        Returns:
            A list of scored points, each exposing ``id``, ``score`` and ``payload``.
        """
        client = self.client()

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(getattr(response, "points", response) or [])

        # Legacy path for qdrant-client < 1.10.
        return list(
            client.search(  # type: ignore[attr-defined]
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        )

    def dense_search(
        self,
        query_vector: list[float],
        limit: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> list[RetrievedProduct]:
        """Cosine ANN search with optional metadata filtering.

        Args:
            query_vector: The embedded query.
            limit: Max hits.  Defaults to ``VECTOR_SEARCH_TOP_K``.
            filters: Optional metadata constraints.

        Returns:
            Hits ordered by descending similarity (empty list on failure — the
            agent then relies on the keyword half of hybrid search).
        """
        top_k = limit or settings.vector_search_top_k
        self.ensure_collection()

        query_filter = filters.to_qdrant_filter() if filters else None

        try:
            raw_hits = self._query_points(query_vector, top_k, query_filter)
        except Exception:
            logger.error("Dense search failed against %r", self.collection, exc_info=True)
            return []

        results: list[RetrievedProduct] = []
        for rank, hit in enumerate(raw_hits, start=1):
            payload = dict(hit.payload or {})
            product_id = int(payload.get("product_id") or payload.get("id") or hit.id)
            results.append(
                RetrievedProduct(
                    product_id=product_id,
                    payload=payload,
                    dense_score=float(hit.score or 0.0),
                    fused_score=float(hit.score or 0.0),
                    dense_rank=rank,
                    retrieval_mode="dense",
                )
            )

        logger.debug("Dense search returned %d hit(s) (filters=%s)", len(results),
                     filters.describe() if filters else None)
        return results

    def fetch_payloads(self, product_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        """Retrieve stored payloads for specific product ids.

        Used to hydrate BM25 hits that the dense search did not return.
        """
        if not product_ids:
            return {}
        self.ensure_collection()
        try:
            records = self.client().retrieve(
                collection_name=self.collection,
                ids=[int(pid) for pid in product_ids],
                with_payload=True,
            )
        except Exception:
            logger.warning("Could not hydrate payloads from Qdrant", exc_info=True)
            return {}

        out: dict[int, dict[str, Any]] = {}
        for record in records:
            payload = dict(record.payload or {})
            product_id = int(payload.get("product_id") or payload.get("id") or record.id)
            out[product_id] = payload
        return out

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        keyword_hits: Sequence[tuple[int, float]],
        limit: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        payload_hydrator: Optional[Any] = None,
    ) -> list[RetrievedProduct]:
        """Dense + BM25 retrieval fused with Reciprocal Rank Fusion (BONUS 4).

        RRF score for a document ``d``::

            score(d) = Σ_r  1 / (RRF_K + rank_r(d))

        over each ranking ``r`` in which ``d`` appears.  RRF is rank-based, so it
        needs no score normalisation between the (cosine) dense ranking and the
        (unbounded) BM25 ranking — which is exactly why it is the standard choice
        for this fusion.

        Args:
            query: The raw query text (logging/diagnostics only).
            query_vector: Embedded query for the dense half.
            keyword_hits: ``[(product_id, bm25_score)]``, best first.
            limit: Max fused results.  Defaults to ``VECTOR_SEARCH_TOP_K``.
            filters: Metadata constraints for the dense half; also applied
                post-hoc to keyword hits so both halves obey the same rules.
            payload_hydrator: Optional ``callable(ids) -> {id: payload}`` used to
                fill in payloads for keyword-only hits (the SQL catalog is passed
                in by the caller so Qdrant is not queried twice).

        Returns:
            Fused hits ordered by descending RRF score.
        """
        top_k = limit or settings.vector_search_top_k
        dense_results = self.dense_search(query_vector, limit=top_k * 2, filters=filters)

        merged: dict[int, RetrievedProduct] = {}
        for result in dense_results:
            result.fused_score = 1.0 / (RRF_K + (result.dense_rank or top_k))
            merged[result.product_id] = result

        missing_ids: list[int] = []
        for rank, (product_id, score) in enumerate(keyword_hits[: top_k * 2], start=1):
            contribution = 1.0 / (RRF_K + rank)
            existing = merged.get(int(product_id))
            if existing is not None:
                existing.keyword_score = float(score)
                existing.keyword_rank = rank
                existing.fused_score += contribution
                existing.retrieval_mode = "hybrid"
            else:
                merged[int(product_id)] = RetrievedProduct(
                    product_id=int(product_id),
                    payload={},
                    keyword_score=float(score),
                    keyword_rank=rank,
                    fused_score=contribution,
                    retrieval_mode="keyword",
                )
                missing_ids.append(int(product_id))

        # Hydrate keyword-only hits so downstream nodes always see full payloads.
        if missing_ids:
            hydrated: dict[int, dict[str, Any]] = {}
            if payload_hydrator is not None:
                try:
                    hydrated = payload_hydrator(missing_ids) or {}
                except Exception:
                    logger.warning("Payload hydrator failed", exc_info=True)
            if not hydrated:
                hydrated = self.fetch_payloads(missing_ids)
            for product_id in missing_ids:
                payload = hydrated.get(product_id)
                if payload:
                    merged[product_id].payload = payload
                else:
                    merged.pop(product_id, None)

        candidates = list(merged.values())
        if filters is not None:
            candidates = [c for c in candidates if _payload_matches(c.payload, filters)]

        candidates.sort(key=lambda item: item.fused_score, reverse=True)
        logger.info(
            "Hybrid search %r -> %d dense + %d keyword => %d fused (top_k=%d)",
            query[:80], len(dense_results), len(keyword_hits), len(candidates), top_k,
        )
        return candidates[:top_k]


def _payload_matches(payload: dict[str, Any], filters: SearchFilters) -> bool:
    """Apply :class:`SearchFilters` to a payload dict in Python.

    Keyword-only hits never pass through Qdrant's filter engine, so the same
    constraints are enforced here to keep both halves consistent.
    """
    if not payload:
        return False

    if filters.exclude_product_ids:
        product_id = payload.get("product_id") or payload.get("id")
        if product_id is not None and int(product_id) in set(filters.exclude_product_ids):
            return False

    if filters.skill_levels:
        allowed = {s.lower() for s in filters.skill_levels}
        if str(payload.get("skill_level", "")).lower() not in allowed:
            return False

    if filters.categories:
        if payload.get("category") not in set(filters.categories):
            return False

    price = payload.get("price")
    if price is not None:
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            price_value = 0.0
        if filters.max_price is not None and price_value > filters.max_price:
            return False
        if filters.min_price is not None and price_value < filters.min_price:
            return False

    return True


_store: Optional[VectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """Return the process-wide :class:`VectorStore` singleton."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:  # pragma: no branch - race guard
            _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Drop the singleton (tests use this to isolate collections)."""
    global _store
    with _store_lock:
        _store = None
