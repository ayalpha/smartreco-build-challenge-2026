#!/usr/bin/env python3
"""Run offline retrieval evals from the CLI.

Examples
--------
    python scripts/run_evals.py
    python scripts/run_evals.py --k 5 --split train --mode keyword
    python scripts/run_evals.py --json --min-hit-rate 0.5
    python scripts/run_evals.py --case-id langgraph-agents --case-id kubernetes

Uses the application database URL from the environment (or SQLite default).
For a hermetic run against the sample catalog, prefer::

    pytest tests/test_evals.py -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_evals.py` from the repo root without installing.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nexora offline retrieval evals")
    parser.add_argument("--k", type=int, default=3, help="Primary cutoff k (default 3)")
    parser.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=None,
        help="Optional multi-cutoff sweep, e.g. --ks 1 3 5",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Golden-case split filter",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "keyword", "dense"),
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Restrict to case id (repeatable)",
    )
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Filter cases by single tag")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Require all of these tags (AND)",
    )
    parser.add_argument(
        "--tag-any",
        nargs="*",
        default=None,
        help="Require any of these tags (OR)",
    )
    parser.add_argument(
        "--exclude-case-id",
        action="append",
        dest="exclude_case_ids",
        default=None,
        help="Exclude case id (repeatable)",
    )
    parser.add_argument("--min-hit-rate", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-mrr", type=float, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Relevance threshold stored on params (for score-based evals)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="Score thresholds for classification sweeps",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for shuffle")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle filtered cases before limit",
    )
    parser.add_argument(
        "--no-catalog-accuracy",
        action="store_true",
        help="Make accuracy@k alias hit@k instead of full-catalog binary accuracy",
    )
    parser.add_argument(
        "--no-ndcg",
        action="store_true",
        help="Skip nDCG@k in the ranking report",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip MAP@k in the ranking report",
    )
    parser.add_argument("--min-ndcg", type=float, default=None)
    parser.add_argument("--min-map", type=float, default=None)
    parser.add_argument(
        "--f-beta",
        type=float,
        action="append",
        dest="f_betas",
        default=None,
        help="Extra F-beta value to report (repeatable; default 0.5 and 2.0)",
    )
    parser.add_argument(
        "--compare-modes",
        nargs="*",
        default=None,
        help="Compare retrieval modes, e.g. --compare-modes hybrid keyword",
    )
    parser.add_argument(
        "--baseline-mode",
        default="keyword",
        help="Baseline mode for --compare-modes deltas (default keyword)",
    )
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="Report leave-one-case-out ranking stability",
    )
    parser.add_argument(
        "--skill-level",
        action="append",
        dest="skill_levels",
        default=None,
        help="Metadata filter: skill level (repeatable)",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Metadata filter: category (repeatable)",
    )
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap resamples for classification CI (classification-only path)",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "agent", "strict"),
        default=None,
        help="Start from a named EvalParams preset before applying flags",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=None,
        help="Success@k requires this many golds in top-k (agent default 3)",
    )
    parser.add_argument("--min-success-at-k", type=float, default=None)
    parser.add_argument("--judge-weight", type=float, default=None)
    parser.add_argument("--retrieval-weight", type=float, default=None)
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of a table")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    parser.add_argument(
        "--no-per-case",
        action="store_true",
        help="Omit per-query breakdown from the metrics dict",
    )
    parser.add_argument(
        "--seed-sample",
        action="store_true",
        help="Reset DB vectors and load tests.conftest.SAMPLE_PRODUCTS before eval",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from app.database import SessionLocal, init_db
    from app.evals.config import (
        AGENT_ALIGNED_EVAL_PARAMS,
        DEFAULT_EVAL_PARAMS,
        EvalParams,
        STRICT_EVAL_PARAMS,
    )
    from app.evals.runner import compare_retrieval_modes, run_retrieval_eval
    from app.evals.report import format_metrics_report, metrics_to_json

    init_db()
    preset_map = {
        "default": DEFAULT_EVAL_PARAMS,
        "agent": AGENT_ALIGNED_EVAL_PARAMS,
        "strict": STRICT_EVAL_PARAMS,
    }
    base = preset_map[args.preset] if args.preset else EvalParams()
    overrides: dict = {
        "k": args.k,
        "split": args.split,
        "retrieval_mode": args.mode,
        "shuffle_cases": args.shuffle,
        "seed": args.seed,
        "relevance_threshold": args.threshold,
        "use_catalog_accuracy": not args.no_catalog_accuracy,
        "include_ndcg": not args.no_ndcg,
        "include_map": not args.no_map,
        "leave_one_out": args.leave_one_out,
        "n_bootstrap": args.bootstrap,
        "include_per_case": not args.no_per_case,
        "extra": {"tag": args.tag} if args.tag else {},
    }
    if args.ks is not None:
        overrides["ks"] = tuple(args.ks)
    if args.case_ids:
        overrides["case_ids"] = tuple(args.case_ids)
    if args.exclude_case_ids:
        overrides["exclude_case_ids"] = tuple(args.exclude_case_ids)
    if args.tags:
        overrides["tags"] = tuple(args.tags)
    if args.tag_any:
        overrides["tag_any"] = tuple(args.tag_any)
    if args.limit_cases is not None:
        overrides["limit_cases"] = args.limit_cases
    if args.thresholds is not None:
        overrides["thresholds"] = tuple(args.thresholds)
    for flag, key in (
        (args.min_hit_rate, "min_hit_rate"),
        (args.min_precision, "min_precision"),
        (args.min_recall, "min_recall"),
        (args.min_f1, "min_f1"),
        (args.min_accuracy, "min_accuracy"),
        (args.min_mrr, "min_mrr"),
        (args.min_ndcg, "min_ndcg"),
        (args.min_map, "min_map"),
        (args.min_success_at_k, "min_success_at_k"),
        (args.min_relevant, "min_relevant"),
        (args.judge_weight, "judge_weight"),
        (args.retrieval_weight, "retrieval_weight"),
    ):
        if flag is not None:
            overrides[key] = flag
    if args.f_betas:
        overrides["f_betas"] = tuple(args.f_betas)
    if args.compare_modes:
        overrides["compare_modes"] = tuple(args.compare_modes)
    if args.skill_levels:
        overrides["skill_levels"] = tuple(args.skill_levels)
    if args.categories:
        overrides["categories"] = tuple(args.categories)
    if args.max_price is not None:
        overrides["max_price"] = args.max_price

    params = base.with_updates(**overrides)

    with SessionLocal() as db:
        if args.seed_sample:
            _seed_sample_catalog(db)

        if args.compare_modes:
            metrics = compare_retrieval_modes(
                db,
                modes=tuple(args.compare_modes),
                params=params,
                baseline=args.baseline_mode,
            )
            # Gate against the hybrid (or first non-baseline) mode when present.
            gate_source = (
                metrics["modes"].get("hybrid")
                or next(iter(metrics["modes"].values()))
            )
            passed = gate_source.get("passed_gates", True)
        else:
            metrics = run_retrieval_eval(db, params=params)
            passed = metrics.get("passed_gates", True)

        if args.json:
            print(metrics_to_json(metrics))
        elif args.csv:
            from app.evals.report import metrics_to_csv

            print(metrics_to_csv(metrics))
        else:
            title = (
                "Retrieval mode compare"
                if args.compare_modes
                else "Retrieval eval"
            )
            print(format_metrics_report(metrics, title=title))
        return 0 if passed else 1


def _seed_sample_catalog(db) -> None:
    """Load the pytest sample catalog so CLI runs work without a full seed."""
    from sqlalchemy import delete

    from app.models.product import Product
    from app.vector_store.qdrant_client import get_vector_store
    from app.vector_store.sync import sync_products

    # Import sample payloads from the test package when available.
    try:
        from tests.conftest import SAMPLE_PRODUCTS
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Cannot import SAMPLE_PRODUCTS: {exc}") from exc

    db.execute(delete(Product))
    db.commit()
    store = get_vector_store()
    store.reset_collection()
    rows = [Product(**payload) for payload in SAMPLE_PRODUCTS]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    result = sync_products(rows)
    if not result.ok:
        raise SystemExit(f"vector sync failed: {result.error}")


if __name__ == "__main__":
    raise SystemExit(main())
