"""Vector-store package: embeddings, Qdrant access, BM25 and dual-write sync.

Public surface::

    from app.vector_store import hybrid_retrieve, sync_product, SearchFilters

Names are resolved lazily (PEP 562) for the same reason as in
:mod:`app.agent`: ``embeddings`` depends on ``app.agent.mesh_client`` while
``app.agent.nodes`` depends on ``qdrant_client``, so eager re-exports here would
create an import cycle depending on which package was imported first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from app.vector_store.embeddings import EmbeddingResult, embed_documents, embed_query
    from app.vector_store.qdrant_client import (
        RetrievedProduct,
        SearchFilters,
        VectorStore,
        get_vector_store,
    )
    from app.vector_store.sync import (
        SyncResult,
        hybrid_retrieve,
        keyword_search,
        reindex_all,
        remove_product,
        sync_product,
        sync_products,
        verify_sync,
    )

#: Public name -> defining submodule.
_EXPORTS: dict[str, str] = {
    "EmbeddingResult": "app.vector_store.embeddings",
    "RetrievedProduct": "app.vector_store.qdrant_client",
    "SearchFilters": "app.vector_store.qdrant_client",
    "SyncResult": "app.vector_store.sync",
    "VectorStore": "app.vector_store.qdrant_client",
    "embed_documents": "app.vector_store.embeddings",
    "embed_query": "app.vector_store.embeddings",
    "get_vector_store": "app.vector_store.qdrant_client",
    "hybrid_retrieve": "app.vector_store.sync",
    "keyword_search": "app.vector_store.sync",
    "reindex_all": "app.vector_store.sync",
    "remove_product": "app.vector_store.sync",
    "sync_product": "app.vector_store.sync",
    "sync_products": "app.vector_store.sync",
    "verify_sync": "app.vector_store.sync",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include the lazy exports in ``dir()`` output."""
    return sorted(set(globals()) | set(_EXPORTS))
