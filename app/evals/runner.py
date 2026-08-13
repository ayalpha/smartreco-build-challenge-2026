"""Retrieval evaluation runner.

Runs golden queries through hybrid / keyword retrieval, scores them with
ranking metrics (and optional multi-label classification view), and returns a
single metrics dict suitable for tests, JSON dumps, or stdout tables.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from sqlalchemy.orm import Session

from app.evals.config import DEFAULT_EVAL_PARAMS, EvalParams
from app.evals.datasets import (
    GOLDEN_RETRIEVAL_CASES,
    RetrievalCase,
    filter_cases,
    load_products,
    resolve_cases,
)
from app.evals.metrics import (
    classification_metrics,
    multilabel_sets_to_binary_vectors,
    per_query_ranking,
    ranking_metrics_at_k,
    scores_to_labels,
)
from app.evals.report import format_metrics_report
from app.vector_store.qdrant_client import SearchFilters
from app.vector_store.sync import hybrid_retrieve, keyword_search

logger = logging.getLogger(__name__)

RetrieverFn = Callable[[Session, str, int, Optional[SearchFilters]], list[int]]


def _default_retriever(mode: str) -> RetrieverFn:
    """Build a retriever that returns ranked product ids."""

    def retrieve(
        db: Session,
        query: str,
        limit: int,
        filters: Optional[SearchFilters],
    ) -> list[int]:
        if mode == "keyword":
            hits = keyword_search(db, query, limit=limit)
            return [product_id for product_id, _score in hits]
        # hybrid (default) goes through production hybrid_retrieve.
        hits = hybrid_retrieve(db, query, limit=limit, filters=filters)
        return [hit.product_id for hit in hits]

    return retrieve


def run_retrieval_eval(
    db: Session,
    *,
    params: Optional[EvalParams] = None,
    cases: Optional[Sequence[RetrievalCase]] = None,
    retriever: Optional[RetrieverFn] = None,
    filters: Optional[SearchFilters] = None,
) -> dict[str, Any]:
    """Evaluate retrieval quality against golden cases.

    Returns a metrics dict with:

    * primary ranking metrics at ``params.k`` (precision/recall/f1/hit/accuracy/mrr)
    * ``by_k`` blocks for each cutoff in ``params.effective_ks()``
    * multi-label micro/macro classification over (query × catalog) labels
    * optional ``per_case`` breakdowns
    * ``params``, ``passed_gates``, ``gate_failures``
    """
    params = params or DEFAULT_EVAL_PARAMS
    raw_cases = list(cases) if cases is not None else list(GOLDEN_RETRIEVAL_CASES)
    selected = filter_cases(
        raw_cases,
        split=params.split,
        case_ids=params.case_ids,
        limit=params.limit_cases,
        tag=(params.extra or {}).get("tag"),
    )
    resolved = resolve_cases(db, selected)
    products = load_products(db)
    catalog_size = len(products)
    catalog_ids = [p.id for p in products]

    max_k = max(params.effective_ks())
    retrieve = retriever or _default_retriever(params.retrieval_mode)

    ranked_lists: list[list[int]] = []
    gold_lists: list[list[int]] = []
    per_case: list[dict[str, Any]] = []

    for case in resolved:
        ranked = retrieve(db, case["query"], max_k, filters)
        gold = list(case["relevant_ids"])
        ranked_lists.append(ranked)
        gold_lists.append(gold)

        if params.include_per_case:
            case_metrics = {
                "id": case["id"],
                "query": case["query"],
                "relevant_ids": gold,
                "retrieved_ids": ranked[: params.k],
                "split": case["split"],
            }
            case_metrics.update(
                per_query_ranking(
                    ranked,
                    gold,
                    k=params.k,
                    catalog_size=catalog_size,
                    zero_division=params.zero_division,
                )
            )
            per_case.append(case_metrics)

    by_k: dict[str, dict[str, Any]] = {}
    primary: dict[str, Any] = {}
    for cutoff in params.effective_ks():
        report = ranking_metrics_at_k(
            ranked_lists,
            gold_lists,
            k=cutoff,
            catalog_size=catalog_size,
            zero_division=params.zero_division,
        )
        block = report.to_dict()
        by_k[str(cutoff)] = block
        if cutoff == params.k:
            primary = block

    # Multi-label classification view: top-k predicted relevant vs gold set.
    pred_sets = [ranked[: params.k] for ranked in ranked_lists]
    y_true, y_pred = multilabel_sets_to_binary_vectors(
        pred_sets, gold_lists, catalog_ids
    )
    multi_micro = classification_metrics(
        y_true, y_pred, average="micro", zero_division=params.zero_division
    )
    multi_macro = classification_metrics(
        y_true, y_pred, average="macro", zero_division=params.zero_division
    )
    multi_weighted = classification_metrics(
        y_true, y_pred, average="weighted", zero_division=params.zero_division
    )
    multi_binary = classification_metrics(
        y_true,
        y_pred,
        average="binary",
        positive_label=1,
        zero_division=params.zero_division,
    )

    metrics: dict[str, Any] = {
        "task": "retrieval",
        "params": params.to_dict(),
        "n_cases": len(resolved),
        "n_cases_requested": len(selected),
        "catalog_size": catalog_size,
        "k": params.k,
        **primary,
        "by_k": by_k,
        "classification": {
            "binary": multi_binary.to_dict(),
            "micro": multi_micro.to_dict(),
            "macro": multi_macro.to_dict(),
            "weighted": multi_weighted.to_dict(),
        },
        "accuracy_micro": multi_micro.accuracy,
        "precision_micro": multi_micro.precision,
        "recall_micro": multi_micro.recall,
        "f1_micro": multi_micro.f1,
        "precision_macro": multi_macro.precision,
        "recall_macro": multi_macro.recall,
        "f1_macro": multi_macro.f1,
    }
    if params.include_per_case:
        metrics["per_case"] = per_case

    passed, failures = params.passes_gates(metrics)
    metrics["passed_gates"] = passed
    metrics["gate_failures"] = failures
    return metrics


def run_classification_eval(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    params: Optional[EvalParams] = None,
    scores: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Evaluate hard labels (or thresholded scores) with full metric suite.

    When ``scores`` is provided, labels are derived via
    ``params.relevance_threshold`` before scoring — useful for grader outputs.
    """
    params = params or EvalParams(average="binary")
    predictions: Sequence[Any] = y_pred
    if scores is not None:
        predictions = scores_to_labels(
            scores,
            threshold=params.relevance_threshold,
            positive_label=params.positive_label,
            negative_label=0 if params.positive_label != 0 else -1,
        )

    averages = ("binary", "micro", "macro", "weighted")
    blocks: dict[str, Any] = {}
    for average in averages:
        report = classification_metrics(
            y_true,
            predictions,
            average=average,
            positive_label=params.positive_label,
            zero_division=params.zero_division,
        )
        blocks[average] = report.to_dict()

    primary_avg = params.average if params.average in blocks else "binary"
    primary = blocks.get(primary_avg, next(iter(blocks.values()), {}))

    metrics: dict[str, Any] = {
        "task": "classification",
        "params": params.to_dict(),
        "n_samples": len(y_true),
        **{
            key: primary[key]
            for key in ("accuracy", "precision", "recall", "f1", "support")
            if key in primary
        },
        "average": primary_avg,
        "by_average": blocks,
    }
    if "confusion" in primary:
        metrics["confusion"] = primary["confusion"]
    if "per_class" in primary:
        metrics["per_class"] = primary["per_class"]

    passed, failures = params.passes_gates(metrics)
    metrics["passed_gates"] = passed
    metrics["gate_failures"] = failures
    return metrics


def report_retrieval_eval(
    db: Session,
    *,
    params: Optional[EvalParams] = None,
    as_json: bool = False,
) -> str:
    """Run retrieval eval and format for stdout."""
    metrics = run_retrieval_eval(db, params=params)
    return format_metrics_report(metrics, title="Retrieval eval", as_json=as_json)
