"""Retrieval evaluation runner.

Runs golden queries through hybrid / keyword retrieval, scores them with
ranking metrics (and optional multi-label classification view), and returns a
single metrics dict suitable for tests, JSON dumps, or stdout tables.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

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
    beats_baseline,
    best_threshold_by_metric,
    blend_rerank_scores,
    bootstrap_metric_ci,
    classification_metrics,
    classification_metrics_bundle,
    confusion_matrix_labels,
    k_sweep_table,
    matthews_corrcoef,
    mcnemar_test,
    mean_average_precision_at_k,
    mean_ndcg_at_k,
    metrics_delta,
    multilabel_sets_to_binary_vectors,
    per_query_ranking,
    precision_recall_auc,
    random_ranking_baseline,
    rank_by_scores,
    ranking_metrics_at_k,
    scores_to_labels,
    success_at_k,
    summarize_numeric_fields,
    threshold_sweep_metrics,
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
    legacy_tag = (params.extra or {}).get("tag")
    selected = filter_cases(
        raw_cases,
        split=params.split,
        case_ids=params.case_ids,
        exclude_case_ids=params.exclude_case_ids,
        limit=params.limit_cases,
        tag=legacy_tag,
        tags=params.tags,
        tag_any=params.tag_any,
        shuffle=params.shuffle_cases,
        seed=params.seed,
    )
    resolved = resolve_cases(db, selected)
    products = load_products(db)
    catalog_size = len(products)
    catalog_ids = [p.id for p in products]
    ranking_catalog_size = catalog_size if params.use_catalog_accuracy else None

    max_k = max(params.effective_ks())
    retrieve = retriever or _default_retriever(params.retrieval_mode)
    active_filters = filters if filters is not None else params.search_filters()

    ranked_lists: list[list[int]] = []
    gold_lists: list[list[int]] = []
    per_case: list[dict[str, Any]] = []

    for case in resolved:
        ranked = retrieve(db, case["query"], max_k, active_filters)
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
                    catalog_size=ranking_catalog_size,
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
            catalog_size=ranking_catalog_size,
            zero_division=params.zero_division,
        )
        block = report.to_dict()
        if params.include_ndcg:
            block["ndcg_at_k"] = mean_ndcg_at_k(
                ranked_lists,
                gold_lists,
                k=cutoff,
                zero_division=params.zero_division,
            )
        if params.include_map:
            block["map_at_k"] = mean_average_precision_at_k(
                ranked_lists,
                gold_lists,
                k=cutoff,
                zero_division=params.zero_division,
            )
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
    binary_bundle = classification_metrics_bundle(
        y_true,
        y_pred,
        positive_label=1,
        zero_division=params.zero_division,
        betas=(1.0, *params.f_betas),
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
            "binary_bundle": binary_bundle,
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
        if params.include_per_case_summary:
            metrics["per_case_summary"] = summarize_numeric_fields(per_case)

    if params.leave_one_out and len(resolved) >= 2:
        metrics["leave_one_out"] = _leave_one_out_ranking(
            ranked_lists,
            gold_lists,
            k=params.k,
            catalog_size=ranking_catalog_size,
            zero_division=params.zero_division,
        )

    metrics["success_at_k"] = success_at_k(
        ranked_lists,
        gold_lists,
        k=params.k,
        min_relevant=params.min_relevant,
    )
    metrics["min_relevant"] = params.min_relevant
    metrics["k_sweep"] = k_sweep_table(
        ranked_lists,
        gold_lists,
        ks=params.effective_ks(),
        catalog_size=ranking_catalog_size,
        min_relevant=params.min_relevant,
        zero_division=params.zero_division,
    )

    if params.random_baseline_trials > 0 and gold_lists:
        baseline = random_ranking_baseline(
            gold_lists,
            catalog_ids=catalog_ids,
            k=params.k,
            n_trials=params.random_baseline_trials,
            seed=params.seed,
            min_relevant=params.min_relevant,
            zero_division=params.zero_division,
        )
        metrics["random_baseline"] = baseline
        # Compare primary ranking aliases (precision/recall/f1 map to @k).
        ranking_view = {
            "precision": metrics.get("precision_at_k", metrics.get("precision")),
            "recall": metrics.get("recall_at_k", metrics.get("recall")),
            "f1": metrics.get("f1_at_k", metrics.get("f1")),
            "hit_at_k": metrics.get("hit_at_k"),
            "mrr": metrics.get("mrr"),
            "success_at_k": metrics.get("success_at_k"),
        }
        metrics["vs_random"] = beats_baseline(ranking_view, baseline)
        if params.require_beat_random and not metrics["vs_random"]["all_beat"]:
            # Soft-fail via gate list after passes_gates below.
            metrics["_require_beat_random"] = True

    if active_filters is not None:
        metrics["filters"] = active_filters.describe()

    passed, failures = params.passes_gates(metrics)
    if metrics.pop("_require_beat_random", False):
        passed = False
        failures = list(failures) + ["random_baseline: did not beat null model on all keys"]
    metrics["passed_gates"] = passed
    metrics["gate_failures"] = failures
    return metrics


def _leave_one_out_ranking(
    ranked_lists: Sequence[Sequence[int]],
    gold_lists: Sequence[Sequence[int]],
    *,
    k: int,
    catalog_size: Optional[int],
    zero_division: float,
) -> dict[str, Any]:
    """Recompute primary ranking metrics leaving each query out once."""
    n = len(ranked_lists)
    folds: list[dict[str, Any]] = []
    for idx in range(n):
        retained_r = [r for i, r in enumerate(ranked_lists) if i != idx]
        retained_g = [g for i, g in enumerate(gold_lists) if i != idx]
        report = ranking_metrics_at_k(
            retained_r,
            retained_g,
            k=k,
            catalog_size=catalog_size,
            zero_division=zero_division,
        )
        folds.append(report.to_dict())
    summary = summarize_numeric_fields(
        folds,
        keys=(
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "hit_at_k",
            "accuracy_at_k",
            "mrr",
        ),
    )
    return {"n_folds": n, "summary": summary, "folds": folds}


def compare_retrieval_modes(
    db: Session,
    *,
    modes: Sequence[str] = ("hybrid", "keyword"),
    params: Optional[EvalParams] = None,
    cases: Optional[Sequence[RetrievalCase]] = None,
    baseline: str = "keyword",
) -> dict[str, Any]:
    """Run the same golden set under multiple retrieval modes and diff metrics.

    Returns::

        {
          "modes": {mode: metrics_dict, ...},
          "delta_vs_baseline": {mode: {metric: delta, ...}, ...},
          "baseline": baseline,
          "params": ...,
        }
    """
    base_params = params or DEFAULT_EVAL_PARAMS
    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_params = base_params.with_updates(
            retrieval_mode=mode,  # type: ignore[arg-type]
            include_per_case=False,
        )
        mode_metrics[mode] = run_retrieval_eval(
            db, params=mode_params, cases=cases
        )

    if baseline not in mode_metrics:
        baseline = next(iter(mode_metrics))

    deltas: dict[str, dict[str, float]] = {}
    for mode, metrics in mode_metrics.items():
        if mode == baseline:
            continue
        deltas[mode] = metrics_delta(mode_metrics[baseline], metrics)

    return {
        "task": "retrieval_mode_compare",
        "baseline": baseline,
        "modes": mode_metrics,
        "delta_vs_baseline": deltas,
        "params": base_params.to_dict(),
    }


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
    negative_label: Any = 0 if params.positive_label != 0 else -1
    predictions: Sequence[Any] = y_pred
    if scores is not None:
        predictions = scores_to_labels(
            scores,
            threshold=params.relevance_threshold,
            positive_label=params.positive_label,
            negative_label=negative_label,
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

    betas = (1.0, *params.f_betas)
    metrics["bundle"] = classification_metrics_bundle(
        y_true,
        predictions,
        positive_label=params.positive_label,
        zero_division=params.zero_division,
        betas=betas,
    )
    # Surface F-beta aliases at top level for easy asserts / gates.
    for key in ("f0_5", "f2", "specificity", "balanced_accuracy"):
        if key in metrics["bundle"]:
            metrics[key] = metrics["bundle"][key]

    metrics["mcc"] = matthews_corrcoef(
        y_true,
        predictions,
        positive_label=params.positive_label,
        zero_division=params.zero_division,
    )
    metrics["confusion_matrix"] = confusion_matrix_labels(y_true, predictions)

    if params.n_bootstrap > 0:
        metrics["bootstrap"] = {}
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            metrics["bootstrap"][metric_name] = bootstrap_metric_ci(
                y_true,
                predictions,
                metric=metric_name,
                n_bootstrap=params.n_bootstrap,
                seed=params.seed,
                confidence=params.bootstrap_confidence,
                positive_label=params.positive_label,
                zero_division=params.zero_division,
            )

    sweep_thresholds = params.thresholds
    if scores is not None and sweep_thresholds:
        metrics["by_threshold"] = threshold_sweep_metrics(
            y_true,
            scores,
            thresholds=sweep_thresholds,
            positive_label=params.positive_label,
            negative_label=negative_label,
            zero_division=params.zero_division,
        )
        metrics["best_threshold_by_f1"] = best_threshold_by_metric(
            metrics["by_threshold"], metric="f1"
        )
        metrics["pr_auc"] = precision_recall_auc(
            metrics["by_threshold"], zero_division=params.zero_division
        )

    passed, failures = params.passes_gates(metrics)
    metrics["passed_gates"] = passed
    metrics["gate_failures"] = failures
    return metrics


def run_grader_threshold_eval(
    scored_items: Sequence[Mapping[str, Any]],
    *,
    params: Optional[EvalParams] = None,
    score_key: str = "relevance_score",
    label_key: str = "relevant",
) -> dict[str, Any]:
    """Evaluate binary relevance labels vs continuous grader scores.

    Designed for LLM-as-judge / heuristic grader outputs:
    each item is ``{score_key: float, label_key: bool|0/1}``.

    Uses ``params.relevance_threshold`` (default 0.5; agent heuristic uses 0.35).
    """
    params = params or EvalParams(
        relevance_threshold=0.35,  # matches app.agent.nodes._HEURISTIC_RELEVANCE_THRESHOLD
        thresholds=(0.25, 0.35, 0.5, 0.65, 0.8),
        n_bootstrap=50,
        seed=0,
    )
    y_true = [
        1 if bool(item.get(label_key)) else 0 for item in scored_items
    ]
    scores = [float(item.get(score_key, 0.0)) for item in scored_items]
    return run_classification_eval(
        y_true,
        y_pred=[],
        scores=scores,
        params=params,
    )


def run_rerank_eval(
    candidates: Sequence[Mapping[str, Any]],
    *,
    params: Optional[EvalParams] = None,
    id_key: str = "id",
    judge_key: str = "relevance_score",
    retrieval_key: str = "fused_score",
    label_key: str = "relevant",
) -> dict[str, Any]:
    """Evaluate agent-style re-rank blend vs pure judge / pure retrieval order.

    Each candidate mapping needs ``id``, judge score, retrieval score, and a
    binary relevance label. Returns ranking metrics for three orderings:

    * ``judge`` — sort by judge score only
    * ``retrieval`` — sort by fused retrieval score only
    * ``blend`` — 65/35 blend (configurable via ``params.judge_weight``)

    Plus classification accuracy/precision/recall/F1 when blend scores are
    thresholded with ``params.relevance_threshold``.
    """
    params = params or EvalParams(
        k=3,
        relevance_threshold=0.35,
        judge_weight=0.65,
        retrieval_weight=0.35,
        min_relevant=3,
        thresholds=(0.25, 0.35, 0.5, 0.65),
    )
    if not candidates:
        return {
            "task": "rerank",
            "params": params.to_dict(),
            "n_candidates": 0,
            "passed_gates": True,
            "gate_failures": [],
        }

    ids = [item[id_key] for item in candidates]
    judge = [float(item.get(judge_key, 0.0)) for item in candidates]
    retrieval = [float(item.get(retrieval_key, 0.0)) for item in candidates]
    relevant_ids = [
        item[id_key] for item in candidates if bool(item.get(label_key))
    ]
    blend = blend_rerank_scores(
        judge,
        retrieval,
        judge_weight=params.judge_weight,
        retrieval_weight=params.retrieval_weight,
    )

    orderings = {
        "judge": rank_by_scores(ids, judge),
        "retrieval": rank_by_scores(ids, retrieval),
        "blend": rank_by_scores(ids, blend),
    }

    by_order: dict[str, Any] = {}
    for name, ranked in orderings.items():
        report = ranking_metrics_at_k(
            [ranked],
            [relevant_ids],
            k=params.k,
            catalog_size=len(ids),
            zero_division=params.zero_division,
        )
        block = report.to_dict()
        block["success_at_k"] = success_at_k(
            [ranked],
            [relevant_ids],
            k=params.k,
            min_relevant=min(params.min_relevant, max(1, len(relevant_ids))),
        )
        by_order[name] = block

    # Threshold blend scores as a binary grader.
    y_true = [1 if bool(item.get(label_key)) else 0 for item in candidates]
    cls_metrics = run_classification_eval(
        y_true,
        y_pred=[],
        scores=blend,
        params=params.with_updates(
            average="binary",
            n_bootstrap=params.n_bootstrap,
        ),
    )

    primary = by_order["blend"]
    metrics: dict[str, Any] = {
        "task": "rerank",
        "params": params.to_dict(),
        "n_candidates": len(candidates),
        "n_relevant": len(relevant_ids),
        "orderings": by_order,
        "blend_vs_judge": metrics_delta(by_order["judge"], by_order["blend"]),
        "blend_vs_retrieval": metrics_delta(
            by_order["retrieval"], by_order["blend"]
        ),
        **{
            key: primary[key]
            for key in (
                "precision_at_k",
                "recall_at_k",
                "f1_at_k",
                "hit_at_k",
                "accuracy_at_k",
                "mrr",
                "success_at_k",
            )
            if key in primary
        },
        "accuracy": cls_metrics.get("accuracy"),
        "precision": cls_metrics.get("precision"),
        "recall": cls_metrics.get("recall"),
        "f1": cls_metrics.get("f1"),
        "mcc": cls_metrics.get("mcc"),
        "classification": {
            key: cls_metrics[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "by_threshold",
                "best_threshold_by_f1",
                "pr_auc",
                "bundle",
            )
            if key in cls_metrics
        },
        "blend_scores": blend,
        "order_blend": orderings["blend"],
    }
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


def run_param_grid(
    db: Session,
    *,
    grid: Sequence[Mapping[str, Any]],
    base_params: Optional[EvalParams] = None,
    cases: Optional[Sequence[RetrievalCase]] = None,
) -> dict[str, Any]:
    """Run retrieval eval for each param override in ``grid``.

    Each grid entry is a dict of ``EvalParams`` field overrides, e.g.::

        [{"k": 1}, {"k": 3}, {"k": 5, "retrieval_mode": "keyword"}]

    Returns a list of ``{overrides, metrics}`` rows plus a compact APRF table.
    """
    base = base_params or DEFAULT_EVAL_PARAMS
    rows: list[dict[str, Any]] = []
    for index, overrides in enumerate(grid):
        params = base.with_updates(**dict(overrides))
        metrics = run_retrieval_eval(db, params=params, cases=cases)
        rows.append(
            {
                "index": index,
                "overrides": dict(overrides),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "hit_at_k": metrics.get("hit_at_k"),
                "mrr": metrics.get("mrr"),
                "success_at_k": metrics.get("success_at_k"),
                "passed_gates": metrics.get("passed_gates"),
                "n_cases": metrics.get("n_cases"),
                "metrics": metrics,
            }
        )
    return {
        "task": "param_grid",
        "n": len(rows),
        "base_params": base.to_dict(),
        "rows": rows,
        "best_by_f1": max(rows, key=lambda r: float(r.get("f1") or 0.0)) if rows else None,
    }


def compare_thresholds(
    y_true: Sequence[Any],
    scores: Sequence[float],
    *,
    threshold_a: float,
    threshold_b: float,
    positive_label: Any = 1,
) -> dict[str, Any]:
    """Compare two score thresholds with full metrics + McNemar discordant counts.

    Useful when tuning the agent heuristic threshold (0.35) vs a stricter 0.5.
    """
    neg = 0 if positive_label != 0 else -1
    pred_a = scores_to_labels(
        scores, threshold=threshold_a, positive_label=positive_label, negative_label=neg
    )
    pred_b = scores_to_labels(
        scores, threshold=threshold_b, positive_label=positive_label, negative_label=neg
    )
    metrics_a = run_classification_eval(
        y_true,
        pred_a,
        params=EvalParams(
            average="binary",
            positive_label=positive_label,
            relevance_threshold=threshold_a,
        ),
    )
    metrics_b = run_classification_eval(
        y_true,
        pred_b,
        params=EvalParams(
            average="binary",
            positive_label=positive_label,
            relevance_threshold=threshold_b,
        ),
    )
    return {
        "task": "threshold_compare",
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "a": {
            key: metrics_a[key]
            for key in ("accuracy", "precision", "recall", "f1", "mcc")
            if key in metrics_a
        },
        "b": {
            key: metrics_b[key]
            for key in ("accuracy", "precision", "recall", "f1", "mcc")
            if key in metrics_b
        },
        "delta_b_minus_a": metrics_delta(metrics_a, metrics_b),
        "mcnemar": mcnemar_test(y_true, pred_a, pred_b),
    }


def run_eval_suite(
    db: Session,
    *,
    params: Optional[EvalParams] = None,
    include_classification_fixtures: bool = True,
    include_grader_fixtures: bool = True,
    include_rerank_fixtures: bool = True,
    include_mode_compare: bool = False,
) -> dict[str, Any]:
    """Run the full offline suite and return a single nested metrics report.

    Sections
    --------
    * ``retrieval`` — golden retrieval cases
    * ``classification_fixtures`` — label fixtures with expected APRF
    * ``grader_fixtures`` — score-threshold grader cases
    * ``rerank_fixtures`` — blend vs judge/retrieval orderings
    * ``mode_compare`` — optional hybrid vs keyword
    """
    from app.evals.datasets import (
        GOLDEN_CLASSIFICATION_FIXTURES,
        GOLDEN_GRADER_SCORE_FIXTURES,
        GOLDEN_RERANK_FIXTURES,
    )

    params = params or DEFAULT_EVAL_PARAMS
    suite: dict[str, Any] = {
        "task": "eval_suite",
        "params": params.to_dict(),
    }

    suite["retrieval"] = run_retrieval_eval(db, params=params)

    if include_classification_fixtures:
        cls_rows = []
        for fixture in GOLDEN_CLASSIFICATION_FIXTURES:
            if fixture.scores:
                metrics = run_classification_eval(
                    list(fixture.y_true),
                    y_pred=[],
                    scores=list(fixture.scores),
                    params=EvalParams(
                        average="binary",
                        relevance_threshold=0.5,
                        thresholds=(0.3, 0.5, 0.8),
                    ),
                )
            else:
                metrics = run_classification_eval(
                    list(fixture.y_true),
                    list(fixture.y_pred),
                    params=EvalParams(average="binary"),
                )
            row = {
                "id": fixture.id,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "expected": fixture.expected,
                "match": all(
                    abs(float(metrics.get(k, -1)) - float(v)) < 1e-9
                    for k, v in fixture.expected.items()
                ),
            }
            cls_rows.append(row)
        suite["classification_fixtures"] = {
            "n": len(cls_rows),
            "n_match": sum(1 for r in cls_rows if r["match"]),
            "rows": cls_rows,
        }

    if include_grader_fixtures:
        grader_rows = []
        for fixture in GOLDEN_GRADER_SCORE_FIXTURES:
            metrics = run_grader_threshold_eval(
                fixture["items"],
                params=EvalParams(
                    relevance_threshold=fixture["threshold"],
                    thresholds=(0.25, 0.35, 0.5, 0.65),
                    n_bootstrap=0,
                ),
            )
            grader_rows.append(
                {
                    "id": fixture["id"],
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "match": all(
                        abs(float(metrics.get(k, -1)) - float(v)) < 1e-9
                        for k, v in fixture["expected"].items()
                    ),
                }
            )
        suite["grader_fixtures"] = {
            "n": len(grader_rows),
            "n_match": sum(1 for r in grader_rows if r["match"]),
            "rows": grader_rows,
        }

    if include_rerank_fixtures:
        rerank_rows = []
        for fixture in GOLDEN_RERANK_FIXTURES:
            metrics = run_rerank_eval(
                fixture["candidates"],
                params=EvalParams(
                    k=fixture["k"],
                    judge_weight=params.judge_weight,
                    retrieval_weight=params.retrieval_weight,
                    relevance_threshold=params.relevance_threshold,
                    min_relevant=1,
                ),
            )
            rerank_rows.append(
                {
                    "id": fixture["id"],
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "blend": metrics.get("orderings", {}).get("blend", {}),
                }
            )
        suite["rerank_fixtures"] = {"n": len(rerank_rows), "rows": rerank_rows}

    if include_mode_compare:
        suite["mode_compare"] = compare_retrieval_modes(
            db,
            modes=("hybrid", "keyword"),
            params=params.with_updates(include_per_case=False),
            baseline="keyword",
        )

    # Suite-level pass: retrieval gates + all classification/grader fixtures match.
    failures: list[str] = []
    if not suite["retrieval"].get("passed_gates", True):
        failures.extend(
            f"retrieval:{f}" for f in suite["retrieval"].get("gate_failures", [])
        )
    for section in ("classification_fixtures", "grader_fixtures"):
        block = suite.get(section)
        if not block:
            continue
        if block["n_match"] != block["n"]:
            failures.append(
                f"{section}: {block['n_match']}/{block['n']} fixtures matched"
            )
    suite["passed_gates"] = not failures
    suite["gate_failures"] = failures
    return suite
