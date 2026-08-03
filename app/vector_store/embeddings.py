"""Embedding generation — Mesh-first, with a deterministic offline fallback.

Primary path
------------
``openai/text-embedding-3-small`` **via Mesh** (see :mod:`app.agent.mesh_client`).

Fallback path
-------------
If Mesh is unreachable or unconfigured we still need *some* vector so the
catalog remains searchable (and so the test-suite and CI can exercise the full
retrieval pipeline without network access).  :func:`hashing_embedding` provides a
deterministic feature-hashed bag-of-n-grams vector of the same dimensionality.
It is genuinely useful for lexical overlap and clearly *worse* than a real
embedding — which is why the model identifier is stamped into every Qdrant point
(:data:`FALLBACK_MODEL_NAME`) so degraded vectors are auditable and can be
re-embedded later with :func:`app.vector_store.sync.reindex_all`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

from app.agent.mesh_client import MeshTelemetry, MeshUnavailableError, embed_texts
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

#: Version tag for the primary embedding pipeline, stored in point metadata.
EMBEDDING_PIPELINE_VERSION = "v1"

#: Identifier written to vector metadata when the offline fallback was used.
FALLBACK_MODEL_NAME = "smartreco/hashing-bow-fallback-v1"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EmbeddingResult:
    """A batch of vectors plus provenance metadata for auditability."""

    vectors: list[list[float]]
    model: str
    dimension: int
    pipeline_version: str
    degraded: bool

    @property
    def provenance(self) -> dict[str, object]:
        """Metadata dict merged into every Qdrant point payload."""
        return {
            "embedding_model": self.model,
            "embedding_dim": self.dimension,
            "embedding_pipeline_version": self.pipeline_version,
            "embedding_degraded": self.degraded,
        }


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer shared by the fallback and BM25."""
    return _TOKEN_RE.findall((text or "").lower())


def hashing_embedding(text: str, dimension: Optional[int] = None) -> list[float]:
    """Deterministic L2-normalised feature-hashed embedding.

    Hashes unigrams and bigrams into ``dimension`` buckets with sub-linear term
    weighting, then normalises.  Cosine similarity over these vectors behaves
    like a smoothed lexical-overlap score.

    Args:
        text: Text to encode.
        dimension: Output dimension.  Defaults to ``MESH_EMBEDDING_DIM`` so
            fallback vectors are interchangeable with real ones in Qdrant.

    Returns:
        A vector of length ``dimension``.  All-zero input yields a unit vector on
        the first axis so Qdrant never rejects a zero-norm point.
    """
    dim = dimension or settings.mesh_embedding_dim
    buckets = [0.0] * dim

    tokens = tokenize(text)
    if not tokens:
        buckets[0] = 1.0
        return buckets

    grams = list(tokens)
    grams.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

    counts: dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1

    for gram, count in counts.items():
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in buckets))
    if norm == 0.0:  # pragma: no cover - astronomically unlikely
        buckets[0] = 1.0
        return buckets
    return [value / norm for value in buckets]


def embed_documents(
    texts: list[str], *, telemetry: Optional[MeshTelemetry] = None
) -> EmbeddingResult:
    """Embed a batch of documents through Mesh, degrading gracefully.

    Args:
        texts: Documents to embed.
        telemetry: Optional Mesh call collector.

    Returns:
        An :class:`EmbeddingResult` carrying the vectors and their provenance.
    """
    if not texts:
        return EmbeddingResult(
            vectors=[],
            model=settings.mesh_embedding_model,
            dimension=settings.mesh_embedding_dim,
            pipeline_version=EMBEDDING_PIPELINE_VERSION,
            degraded=False,
        )

    try:
        vectors = embed_texts(texts, telemetry=telemetry)
        dimension = len(vectors[0]) if vectors else settings.mesh_embedding_dim
        if dimension != settings.mesh_embedding_dim:
            logger.warning(
                "Mesh returned %d-dim embeddings but MESH_EMBEDDING_DIM=%d. "
                "Update the env var and re-index to keep Qdrant consistent.",
                dimension, settings.mesh_embedding_dim,
            )
        return EmbeddingResult(
            vectors=vectors,
            model=settings.mesh_embedding_model,
            dimension=dimension,
            pipeline_version=EMBEDDING_PIPELINE_VERSION,
            degraded=False,
        )
    except MeshUnavailableError as exc:
        logger.warning(
            "Embedding via Mesh unavailable (%s) — using deterministic hashing "
            "fallback (%s). Re-run `python -m scripts.seed_products --reindex` "
            "once Mesh is reachable.", exc, FALLBACK_MODEL_NAME,
        )
        return EmbeddingResult(
            vectors=[hashing_embedding(text) for text in texts],
            model=FALLBACK_MODEL_NAME,
            dimension=settings.mesh_embedding_dim,
            pipeline_version=EMBEDDING_PIPELINE_VERSION,
            degraded=True,
        )


def embed_query(text: str, *, telemetry: Optional[MeshTelemetry] = None) -> EmbeddingResult:
    """Embed a single search query (thin wrapper over :func:`embed_documents`)."""
    return embed_documents([text], telemetry=telemetry)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 for mismatched lengths or zero-norm inputs rather than raising,
    because this runs inside scoring loops where a hard failure is unhelpful.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)
