"""Offline evaluation helpers for retrieval quality and classification metrics.

Public surface::

    from app.evals import (
        EvalParams,
        classification_metrics,
        ranking_metrics_at_k,
        run_retrieval_eval,
        format_metrics_report,
    )

The package is deliberately free of Mesh / network side effects: unit metrics
operate on labels and ranked id lists; the retrieval runner uses the same
in-process hybrid path as the agent (SQLite + embedded Qdrant in tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from app.evals.config import EvalParams
    from app.evals.metrics import (
        ClassificationReport,
        RankingReport,
        classification_metrics,
        confusion_counts,
        ranking_metrics_at_k,
    )
    from app.evals.report import format_metrics_report, metrics_to_json, metrics_to_table
    from app.evals.runner import run_retrieval_eval

_EXPORTS: dict[str, str] = {
    "ClassificationReport": "app.evals.metrics",
    "EvalParams": "app.evals.config",
    "RankingReport": "app.evals.metrics",
    "classification_metrics": "app.evals.metrics",
    "classification_metrics_bundle": "app.evals.metrics",
    "confusion_counts": "app.evals.metrics",
    "fbeta_score": "app.evals.metrics",
    "format_metrics_report": "app.evals.report",
    "mean_average_precision_at_k": "app.evals.metrics",
    "metrics_delta": "app.evals.metrics",
    "metrics_to_json": "app.evals.report",
    "metrics_to_table": "app.evals.report",
    "ranking_metrics_at_k": "app.evals.metrics",
    "run_classification_eval": "app.evals.runner",
    "run_retrieval_eval": "app.evals.runner",
    "threshold_sweep_metrics": "app.evals.metrics",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its submodule on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include the lazy exports in ``dir()`` output."""
    return sorted(set(globals()) | set(_EXPORTS))
