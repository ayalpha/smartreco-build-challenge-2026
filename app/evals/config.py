"""Configurable parameters for offline evals.

Defaults mirror production retrieval knobs where that makes sense
(``vector_search_top_k``, agent final count) so thresholds in tests stay
aligned with what the app actually serves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

AverageMode = Literal["binary", "micro", "macro", "weighted"]
SplitName = Literal["all", "train", "test"]
RetrievalMode = Literal["hybrid", "keyword", "dense"]


@dataclass(frozen=True)
class EvalParams:
    """Inputs that steer an evaluation run.

    Attributes:
        k: Cutoff for ranking metrics (precision@k, recall@k, hit@k, accuracy@k).
        ks: Optional multi-cutoff sweep; when set, reports metrics for each k.
        relevance_threshold: Minimum score to treat a scored item as positive
            when converting continuous scores → binary labels.
        average: Aggregation for multi-class precision/recall/F1.
            - ``binary``: positive class only (requires ``positive_label``).
            - ``micro``: global TP/FP/FN (good for imbalanced multi-class).
            - ``macro``: unweighted mean of per-class scores.
            - ``weighted``: support-weighted mean of per-class scores.
        positive_label: Label treated as the positive class for binary mode.
        min_accuracy / min_precision / min_recall / min_f1: Optional gates.
            When set, :meth:`passes_gates` checks the metrics dict against them.
        split: Filter golden cases by split tag (``all`` keeps everything).
        case_ids: If non-empty, only evaluate these case identifiers.
        limit_cases: Cap on how many cases to run (after split/id filters).
        retrieval_mode: Which retriever the harness should call.
        include_per_case: Whether the report embeds per-query breakdowns.
        zero_division: Value used when a metric denominator is zero
            (sklearn-compatible convention; default 0.0).
    """

    k: int = 3
    ks: Optional[tuple[int, ...]] = None
    relevance_threshold: float = 0.5
    average: AverageMode = "binary"
    positive_label: Any = 1
    min_accuracy: Optional[float] = None
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    min_f1: Optional[float] = None
    min_hit_rate: Optional[float] = None
    split: SplitName = "all"
    case_ids: Optional[tuple[str, ...]] = None
    limit_cases: Optional[int] = None
    retrieval_mode: RetrievalMode = "hybrid"
    include_per_case: bool = True
    zero_division: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.ks is not None:
            if not self.ks or any(value < 1 for value in self.ks):
                raise ValueError(f"ks must be non-empty positive ints, got {self.ks}")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be in [0, 1], got {self.relevance_threshold}"
            )
        if self.limit_cases is not None and self.limit_cases < 1:
            raise ValueError(f"limit_cases must be >= 1, got {self.limit_cases}")
        if not 0.0 <= self.zero_division <= 1.0:
            raise ValueError(
                f"zero_division must be in [0, 1], got {self.zero_division}"
            )

    def effective_ks(self) -> tuple[int, ...]:
        """Return the cutoffs to evaluate (``ks`` if set, else ``(k,)``)."""
        if self.ks:
            return self.ks
        return (self.k,)

    def with_updates(self, **changes: Any) -> "EvalParams":
        """Return a copy with selected fields replaced (immutable helper)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the parameters."""
        data = asdict(self)
        # tuples survive asdict; keep them as lists for JSON dumps.
        if data.get("ks") is not None:
            data["ks"] = list(data["ks"])
        if data.get("case_ids") is not None:
            data["case_ids"] = list(data["case_ids"])
        return data

    def passes_gates(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check optional quality gates against a flat metrics mapping.

        Looks up both top-level keys (``accuracy``) and ranking aliases
        (``precision_at_k``, ``hit_rate``). Returns ``(ok, failures)``.
        """
        failures: list[str] = []
        checks: Sequence[tuple[str, Optional[float], tuple[str, ...]]] = (
            ("accuracy", self.min_accuracy, ("accuracy", "accuracy_at_k")),
            ("precision", self.min_precision, ("precision", "precision_at_k")),
            ("recall", self.min_recall, ("recall", "recall_at_k")),
            ("f1", self.min_f1, ("f1", "f1_at_k")),
            ("hit_rate", self.min_hit_rate, ("hit_rate", "hit_at_k")),
        )
        for label, minimum, keys in checks:
            if minimum is None:
                continue
            value = _first_present(metrics, keys)
            if value is None:
                failures.append(f"{label}: missing (gate required >= {minimum})")
            elif float(value) < minimum:
                failures.append(f"{label}: {float(value):.4f} < {minimum}")
        return (not failures, failures)


def _first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """Return the first numeric value found under ``keys`` (nested ok)."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
        # Nested under a primary k-block: metrics["at_k"]["precision"]
        nested = mapping.get("at_k")
        if isinstance(nested, dict) and key in nested:
            try:
                return float(nested[key])
            except (TypeError, ValueError):
                continue
    return None


#: Sensible default for CI: k=3 matches a typical top-shelf recs strip.
DEFAULT_EVAL_PARAMS = EvalParams(
    k=3,
    ks=(1, 3, 5),
    average="binary",
    min_hit_rate=0.5,
    include_per_case=True,
)
