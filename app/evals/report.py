"""Turn metrics dicts into JSON / plain-text tables for stdout or tests."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence


def metrics_to_json(metrics: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a metrics mapping to stable JSON (sorted keys)."""
    return json.dumps(metrics, indent=indent, sort_keys=True, default=str)


def metrics_to_csv_rows(
    metrics: Mapping[str, Any],
    *,
    keys: Optional[Sequence[str]] = None,
) -> list[tuple[str, str]]:
    """Flatten scalar metrics to ``(name, value)`` rows for CSV export."""
    preferred = keys or (
        "task",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "hit_at_k",
        "mrr",
        "ndcg_at_k",
        "map_at_k",
        "pr_auc",
        "k",
        "n_cases",
        "n_samples",
        "passed_gates",
    )
    rows: list[tuple[str, str]] = []
    for key in preferred:
        if key in metrics and not isinstance(metrics[key], (dict, list)):
            rows.append((key, _format_value(metrics[key], 6)))
    return rows


def metrics_to_csv(metrics: Mapping[str, Any]) -> str:
    """Two-column CSV (metric,value) for scalar fields."""
    lines = ["metric,value"]
    for name, value in metrics_to_csv_rows(metrics):
        # Escape commas in values (unlikely for scalars).
        safe = value.replace(",", ";")
        lines.append(f"{name},{safe}")
    return "\n".join(lines) + "\n"


def write_metrics(
    metrics: Mapping[str, Any],
    path: str,
    *,
    fmt: str = "json",
) -> str:
    """Write metrics to ``path`` as json/csv/table; returns the path."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        target.write_text(metrics_to_json(metrics), encoding="utf-8")
    elif fmt == "csv":
        target.write_text(metrics_to_csv(metrics), encoding="utf-8")
    elif fmt == "table":
        target.write_text(format_metrics_report(metrics), encoding="utf-8")
    else:
        raise ValueError(f"fmt must be json|csv|table, got {fmt!r}")
    return str(target)


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
        "f0_5",
        "f2",
        "specificity",
        "balanced_accuracy",
        "pr_auc",
        "accuracy_at_k",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "hit_at_k",
        "hit_rate",
        "mrr",
        "ndcg_at_k",
        "map_at_k",
        "success_at_k",
        "mcc",
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
        if key in seen or key in {
            "per_case",
            "params",
            "by_k",
            "per_class",
            "cases",
            "by_threshold",
            "by_average",
            "classification",
            "modes",
            "delta_vs_baseline",
            "bundle",
            "per_case_summary",
        }:
            continue
        if isinstance(value, (int, float, bool, str)) or value is None:
            rows.append((key, _format_value(value, float_digits)))

    # Compact nest summaries
    for nest_key in (
        "by_k",
        "per_class",
        "per_case",
        "cases",
        "by_threshold",
        "modes",
        "classification",
    ):
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

    by_threshold = metrics.get("by_threshold")
    if isinstance(by_threshold, list) and by_threshold:
        sections.append("")
        sections.append("Metrics by threshold")
        for row in by_threshold:
            if not isinstance(row, dict):
                continue
            thr = row.get("threshold")
            sections.append(
                f"  — t={thr}: "
                f"A={_format_value(row.get('accuracy'), 4)} "
                f"P={_format_value(row.get('precision'), 4)} "
                f"R={_format_value(row.get('recall'), 4)} "
                f"F1={_format_value(row.get('f1'), 4)}"
            )

    vs_random = metrics.get("vs_random")
    if isinstance(vs_random, dict) and vs_random.get("by_key"):
        sections.append("")
        sections.append("Vs random baseline")
        for key, block in vs_random["by_key"].items():
            if not isinstance(block, dict):
                continue
            beat = "WIN" if block.get("beats") else "lose"
            sections.append(
                f"  — {key}: metric={_format_value(block.get('metric'), 4)} "
                f"base={_format_value(block.get('baseline'), 4)} "
                f"Δ={_format_value(block.get('delta'), 4)} [{beat}]"
            )

    k_sweep = metrics.get("k_sweep")
    if isinstance(k_sweep, dict) and k_sweep:
        sections.append("")
        sections.append("k-sweep (accuracy / precision / recall / f1)")
        for k_label, block in sorted(k_sweep.items(), key=lambda kv: int(kv[0])):
            if not isinstance(block, dict):
                continue
            sections.append(
                f"  — k={k_label}: "
                f"A={_format_value(block.get('accuracy'), 4)} "
                f"P={_format_value(block.get('precision'), 4)} "
                f"R={_format_value(block.get('recall'), 4)} "
                f"F1={_format_value(block.get('f1'), 4)} "
                f"hit={_format_value(block.get('hit_at_k'), 4)} "
                f"succ={_format_value(block.get('success_at_k'), 4)}"
            )

    modes = metrics.get("modes")
    if isinstance(modes, dict) and modes:
        sections.append("")
        sections.append("Modes")
        for mode_name, block in modes.items():
            sections.append(f"  — mode={mode_name}")
            if isinstance(block, dict):
                for line in metrics_to_table(block).splitlines()[:12]:
                    sections.append(f"    {line}")
        deltas = metrics.get("delta_vs_baseline")
        if isinstance(deltas, dict) and deltas:
            sections.append("")
            sections.append(f"Delta vs baseline ({metrics.get('baseline')})")
            for mode_name, delta in deltas.items():
                if isinstance(delta, dict):
                    sections.append(f"  — {mode_name}")
                    for line in metrics_to_table(delta).splitlines():
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
