"""Turn metrics dicts into JSON / plain-text tables for stdout or tests."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence


def metrics_to_json(metrics: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a metrics mapping to stable JSON (sorted keys)."""
    return json.dumps(metrics, indent=indent, sort_keys=True, default=str)


def metrics_to_table(
    metrics: Mapping[str, Any],
    *,
    keys: Optional[Sequence[str]] = None,
    float_digits: int = 4,
) -> str:
    """Render a simple two-column table of scalar metrics.

    Nested mappings (``per_class``, ``by_k``, ``per_case``) are summarised as
    counts rather than fully expanded, so the table stays scannable.
    """
    preferred = keys or (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "accuracy_at_k",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "hit_at_k",
        "hit_rate",
        "mrr",
        "n_queries",
        "k",
        "passed_gates",
    )
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for key in preferred:
        if key not in metrics:
            continue
        rows.append((key, _format_value(metrics[key], float_digits)))
        seen.add(key)

    for key, value in metrics.items():
        if key in seen or key in {"per_case", "params", "by_k", "per_class", "cases"}:
            continue
        if isinstance(value, (int, float, bool, str)) or value is None:
            rows.append((key, _format_value(value, float_digits)))

    # Compact nest summaries
    for nest_key in ("by_k", "per_class", "per_case", "cases"):
        if nest_key in metrics and isinstance(metrics[nest_key], (dict, list)):
            size = len(metrics[nest_key])
            rows.append((nest_key, f"<{size} entries>"))

    if not rows:
        return "(no scalar metrics)"

    width = max(len(name) for name, _ in rows)
    lines = [f"{name:<{width}}  {value}" for name, value in rows]
    return "\n".join(lines)


def format_metrics_report(
    metrics: Mapping[str, Any],
    *,
    title: str = "Eval report",
    as_json: bool = False,
) -> str:
    """Human-readable multi-section report (stdout-friendly)."""
    if as_json:
        return metrics_to_json(metrics)

    sections: list[str] = [title, "=" * len(title)]

    params = metrics.get("params")
    if isinstance(params, dict):
        sections.append("Parameters")
        sections.append(metrics_to_table(params, keys=sorted(params.keys())))
        sections.append("")

    sections.append("Primary metrics")
    sections.append(metrics_to_table(metrics))

    by_k = metrics.get("by_k")
    if isinstance(by_k, dict) and by_k:
        sections.append("")
        sections.append("Metrics by k")
        for k_label, block in sorted(by_k.items(), key=lambda kv: str(kv[0])):
            sections.append(f"  — k={k_label}")
            if isinstance(block, dict):
                for line in metrics_to_table(block).splitlines():
                    sections.append(f"    {line}")

    gates = metrics.get("gate_failures")
    if gates:
        sections.append("")
        sections.append("Gate failures")
        for failure in gates:
            sections.append(f"  - {failure}")
    elif "passed_gates" in metrics:
        sections.append("")
        sections.append(
            "Gates: PASSED" if metrics["passed_gates"] else "Gates: FAILED"
        )

    return "\n".join(sections).rstrip() + "\n"


def _format_value(value: Any, float_digits: int) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{float_digits}f}"
    if value is None:
        return "null"
    return str(value)
