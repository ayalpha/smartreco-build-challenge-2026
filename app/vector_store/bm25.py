"""Okapi BM25 keyword index — the sparse half of hybrid retrieval (BONUS 4).

Why implement BM25 in-process instead of using a sparse-vector service?
The catalog is small (hundreds of courses) and fully resident in SQL, so a
correct BM25 ranking costs microseconds and adds zero infrastructure.  The index
is rebuilt lazily behind a version stamp, so catalog writes invalidate it without
any explicit coupling from the router code.

The output is *ranked ids*, which :func:`app.vector_store.qdrant_client.hybrid_search`
fuses with the dense Qdrant ranking using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from app.vector_store.embeddings import tokenize

logger = logging.getLogger(__name__)

#: Standard Okapi BM25 parameters.
K1 = 1.5
B = 0.75

#: How long a built index is trusted before rebuilding (seconds).
INDEX_TTL_SECONDS = 300


@dataclass
class BM25Document:
    """One indexable catalog entry."""

    doc_id: int
    text: str


@dataclass
class BM25Index:
    """An immutable, queryable BM25 index over the catalog."""

    doc_ids: list[int] = field(default_factory=list)
    term_frequencies: list[dict[str, int]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    document_frequency: dict[str, int] = field(default_factory=dict)
    average_length: float = 0.0
    built_at: float = field(default_factory=time.monotonic)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self.doc_ids)

    @property
    def is_stale(self) -> bool:
        """True once the index has outlived :data:`INDEX_TTL_SECONDS`."""
        return (time.monotonic() - self.built_at) > INDEX_TTL_SECONDS

    def search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """Rank documents against ``query`` using Okapi BM25.

        Args:
            query: Free-text query.
            limit: Maximum results to return.

        Returns:
            ``[(doc_id, score)]`` sorted by descending score; only positive
            scores are returned.
        """
        terms = tokenize(query)
        if not terms or self.size == 0:
            return []

        scores: dict[int, float] = {}
        total_docs = self.size

        for term in set(terms):
            doc_freq = self.document_frequency.get(term, 0)
            if doc_freq == 0:
                continue
            # BM25 IDF with the +1 numerator/denominator smoothing that keeps
            # very common terms from going negative.
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for position, frequencies in enumerate(self.term_frequencies):
                freq = frequencies.get(term)
                if not freq:
                    continue
                doc_length = self.doc_lengths[position] or 1
                denominator = freq + K1 * (
                    1.0 - B + B * (doc_length / (self.average_length or 1.0))
                )
                contribution = idf * (freq * (K1 + 1.0)) / denominator
                doc_id = self.doc_ids[position]
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked if score > 0.0][:limit]


def build_index(documents: Iterable[BM25Document]) -> BM25Index:
    """Build a BM25 index from an iterable of documents.

    Args:
        documents: The catalog entries to index.

    Returns:
        A ready-to-query :class:`BM25Index`.
    """
    index = BM25Index()
    total_length = 0

    for document in documents:
        tokens = tokenize(document.text)
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

        index.doc_ids.append(document.doc_id)
        index.term_frequencies.append(frequencies)
        index.doc_lengths.append(len(tokens))
        total_length += len(tokens)

        for term in frequencies:
            index.document_frequency[term] = index.document_frequency.get(term, 0) + 1

    index.average_length = (total_length / index.size) if index.size else 0.0
    logger.debug("Built BM25 index over %d documents (avg length %.1f tokens)",
                 index.size, index.average_length)
    return index


# --------------------------------------------------------------------------- #
# Process-wide cached index                                                   #
# --------------------------------------------------------------------------- #

_index: Optional[BM25Index] = None
_index_version = 0
_built_version = -1
_index_lock = threading.Lock()


def invalidate_index() -> None:
    """Mark the cached index stale.

    Called by the dual-write layer on every catalog create/update/delete so
    keyword search never ranks against a deleted course.
    """
    global _index_version
    with _index_lock:
        _index_version += 1
    logger.debug("BM25 index invalidated (version=%d)", _index_version)


def get_index(documents_provider: Callable[[], list[BM25Document]]) -> BM25Index:
    """Return the cached index, rebuilding it when stale or invalidated.

    Args:
        documents_provider: Zero-argument callable returning the current
            documents.  Only invoked when a rebuild is actually required, so the
            caller's database query is skipped on cache hits.

    Returns:
        A fresh-enough :class:`BM25Index`.
    """
    global _index, _built_version

    with _index_lock:
        needs_rebuild = (
            _index is None or _built_version != _index_version or _index.is_stale
        )
        if not needs_rebuild and _index is not None:
            return _index

        documents = documents_provider()
        _index = build_index(documents)
        _built_version = _index_version
        return _index
