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
    from app.evals.config import EvalParams
    from app.evals.runner import run_retrieval_eval
    from app.evals.report import metrics_to_json

    init_db()
    params = EvalParams(
        k=args.k,
        ks=tuple(args.ks) if args.ks else None,
        split=args.split,
        retrieval_mode=args.mode,
        case_ids=tuple(args.case_ids) if args.case_ids else None,
        exclude_case_ids=tuple(args.exclude_case_ids) if args.exclude_case_ids else None,
        tags=tuple(args.tags) if args.tags else None,
        tag_any=tuple(args.tag_any) if args.tag_any else None,
        limit_cases=args.limit_cases,
        shuffle_cases=args.shuffle,
        seed=args.seed,
        relevance_threshold=args.threshold,
        thresholds=tuple(args.thresholds) if args.thresholds else None,
        min_hit_rate=args.min_hit_rate,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        min_accuracy=args.min_accuracy,
        min_mrr=args.min_mrr,
        use_catalog_accuracy=not args.no_catalog_accuracy,
        include_ndcg=not args.no_ndcg,
        include_per_case=not args.no_per_case,
        extra={"tag": args.tag} if args.tag else {},
    )

    with SessionLocal() as db:
        if args.seed_sample:
            _seed_sample_catalog(db)

        metrics = run_retrieval_eval(db, params=params)
        if args.json:
            print(metrics_to_json(metrics))
        else:
            from app.evals.report import format_metrics_report

            print(format_metrics_report(metrics, title="Retrieval eval"))
        return 0 if metrics.get("passed_gates", True) else 1


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
