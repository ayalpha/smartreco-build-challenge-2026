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


def _load_fixture(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        examples, config = payload, {}
    elif isinstance(payload, dict):
        examples, config = payload.get("examples"), payload.get("parameters", {})
    else:
        raise ValueError("fixture must be an examples array or an object containing one")
    if not isinstance(examples, list):
        raise ValueError("fixture 'examples' must be an array")
    if not isinstance(config, dict):
        raise ValueError("fixture 'parameters' must be an object")
    for index, row in enumerate(examples):
        if not isinstance(row, dict) or "expected" not in row or "predicted" not in row:
            raise ValueError(f"fixture example {index} must contain expected and predicted")
    return examples, config


def main() -> int:
    args = parse_args()
    examples, fixture_parameters = _load_fixture(args.fixtures)
    parameters = {
        "k": args.k if args.k is not None else fixture_parameters.get("k"),
        "threshold": args.threshold if args.threshold is not None else fixture_parameters.get("threshold"),
        "labels": args.labels if args.labels is not None else fixture_parameters.get("labels"),
    }
    metrics = evaluate(
        [row["expected"] for row in examples],
        [row["predicted"] for row in examples],
        **parameters,
    ).to_dict()
    metrics["parameters"] = parameters
    report = json.dumps(metrics, indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
