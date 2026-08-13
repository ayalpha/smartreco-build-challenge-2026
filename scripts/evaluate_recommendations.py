"""Evaluate recommendation fixtures. Run with ``python scripts/evaluate_recommendations.py``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute classification metrics for recommendation fixtures")
    parser.add_argument("--fixtures", type=Path, required=True, help="JSON file containing an examples array")
    parser.add_argument("--k", type=int, default=None, help="Evaluate only the top K predictions")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum score for scored predictions")
    parser.add_argument("--labels", nargs="+", help="Complete label set (otherwise inferred)")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.fixtures.read_text(encoding="utf-8"))
    examples = payload["examples"] if isinstance(payload, dict) else payload
    metrics = evaluate(
        [row["expected"] for row in examples],
        [row["predicted"] for row in examples],
        k=args.k, threshold=args.threshold, labels=args.labels,
    ).to_dict()
    report = json.dumps(metrics, indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
