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
        thresholds: Optional multi-threshold sweep for score-based classification
            (precision/recall/F1/accuracy at each threshold).
        average: Aggregation for multi-class precision/recall/F1.
            - ``binary``: positive class only (requires ``positive_label``).
            - ``micro``: global TP/FP/FN (good for imbalanced multi-class).
            - ``macro``: unweighted mean of per-class scores.
            - ``weighted``: support-weighted mean of per-class scores.
        positive_label: Label treated as the positive class for binary mode.
        min_accuracy / min_precision / min_recall / min_f1: Optional gates.
            When set, :meth:`passes_gates` checks the metrics dict against them.
        min_mrr: Optional gate on mean reciprocal rank.
        split: Filter golden cases by split tag (``all`` keeps everything).
        case_ids: If non-empty, only evaluate these case identifiers.
        exclude_case_ids: Drop these case ids after other filters.
        tags: Require cases to include *all* of these tags (AND).
        tag_any: Require cases to include *any* of these tags (OR).
        limit_cases: Cap on how many cases to run (after split/id filters).
        shuffle_cases: Shuffle filtered cases before ``limit_cases`` (uses seed).
        seed: RNG seed for shuffle (and any future sampling).
        retrieval_mode: Which retriever the harness should call.
        use_catalog_accuracy: When True, accuracy@k uses full-catalog binary
            view; when False, accuracy@k aliases hit@k.
        include_per_case: Whether the report embeds per-query breakdowns.
        include_ndcg: Whether to attach nDCG@k alongside ranking metrics.
        include_map: Whether to attach MAP@k alongside ranking metrics.
        include_per_case_summary: Mean/min/max over per-case ranking metrics.
        f_betas: Extra F-beta scores to report in classification bundles
            (always includes F1 via beta=1 implicitly in primary metrics).
        min_ndcg / min_map: Optional ranking quality gates.
        min_pr_auc: Optional gate on precision–recall AUC (score-based evals).
        compare_modes: When non-empty, CLI/harness can run multi-mode compare.
        leave_one_out: When True, also report leave-one-case-out metric means.
        n_bootstrap: Bootstrap resamples for classification CI (0 disables).
        bootstrap_confidence: CI level for bootstrap (e.g. 0.95).
        skill_levels / categories / max_price: Optional retrieval metadata filters
            applied during hybrid search (mirrors agent SearchFilters).
        zero_division: Value used when a metric denominator is zero
            (sklearn-compatible convention; default 0.0).
    """

    k: int = 3
    ks: Optional[tuple[int, ...]] = None
    relevance_threshold: float = 0.5
    thresholds: Optional[tuple[float, ...]] = None
    average: AverageMode = "binary"
    positive_label: Any = 1
    min_accuracy: Optional[float] = None
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    min_f1: Optional[float] = None
    min_hit_rate: Optional[float] = None
    min_mrr: Optional[float] = None
    min_ndcg: Optional[float] = None
    min_map: Optional[float] = None
    min_pr_auc: Optional[float] = None
    split: SplitName = "all"
    case_ids: Optional[tuple[str, ...]] = None
    exclude_case_ids: Optional[tuple[str, ...]] = None
    tags: Optional[tuple[str, ...]] = None
    tag_any: Optional[tuple[str, ...]] = None
    limit_cases: Optional[int] = None
    shuffle_cases: bool = False
    seed: int = 0
    retrieval_mode: RetrievalMode = "hybrid"
    use_catalog_accuracy: bool = True
    include_per_case: bool = True
    include_per_case_summary: bool = True
    include_ndcg: bool = True
    include_map: bool = True
    leave_one_out: bool = False
    n_bootstrap: int = 0
    bootstrap_confidence: float = 0.95
    skill_levels: Optional[tuple[str, ...]] = None
    categories: Optional[tuple[str, ...]] = None
    max_price: Optional[float] = None
    f_betas: tuple[float, ...] = (0.5, 2.0)
    compare_modes: Optional[tuple[str, ...]] = None
    #: Agent re-rank blend (matches app.agent.nodes._RERANK_*).
    judge_weight: float = 0.65
    retrieval_weight: float = 0.35
    #: Agent gate: need this many relevants in top-k (agent_min_relevant_products).
    min_relevant: int = 3
    min_success_at_k: Optional[float] = None
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
        if self.thresholds is not None:
            if not self.thresholds:
                raise ValueError("thresholds must be non-empty when provided")
            for value in self.thresholds:
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"each threshold must be in [0, 1], got {value}"
                    )
        if self.limit_cases is not None and self.limit_cases < 1:
            raise ValueError(f"limit_cases must be >= 1, got {self.limit_cases}")
        if not 0.0 <= self.zero_division <= 1.0:
            raise ValueError(
                f"zero_division must be in [0, 1], got {self.zero_division}"
            )
        if any(beta < 0 for beta in self.f_betas):
            raise ValueError(f"f_betas must be >= 0, got {self.f_betas}")
        if self.n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be >= 0, got {self.n_bootstrap}")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError(
                f"bootstrap_confidence must be in (0,1), got {self.bootstrap_confidence}"
            )
        if self.max_price is not None and self.max_price < 0:
            raise ValueError(f"max_price must be >= 0, got {self.max_price}")
        if self.judge_weight < 0 or self.retrieval_weight < 0:
            raise ValueError("judge_weight and retrieval_weight must be >= 0")
        if self.min_relevant < 1:
            raise ValueError(f"min_relevant must be >= 1, got {self.min_relevant}")

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
        for key in (
            "ks",
            "thresholds",
            "case_ids",
            "exclude_case_ids",
            "tags",
            "tag_any",
            "f_betas",
            "compare_modes",
            "skill_levels",
            "categories",
        ):
            if data.get(key) is not None:
                data[key] = list(data[key])
        return data

    def search_filters(self) -> Optional[Any]:
        """Build a :class:`~app.vector_store.qdrant_client.SearchFilters` if set."""
        if not self.skill_levels and not self.categories and self.max_price is None:
            return None
        from app.vector_store.qdrant_client import SearchFilters

        return SearchFilters(
            skill_levels=list(self.skill_levels) if self.skill_levels else None,
            categories=list(self.categories) if self.categories else None,
            max_price=self.max_price,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EvalParams":
        """Build params from a plain dict (CLI / JSON fixtures / tests).

        Unknown keys are ignored so fixture files can carry documentation fields.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known or value is None:
                continue
            if key in {
                "ks",
                "thresholds",
                "case_ids",
                "exclude_case_ids",
                "tags",
                "tag_any",
                "f_betas",
                "compare_modes",
                "skill_levels",
                "categories",
            } and isinstance(value, list):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

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
            ("mrr", self.min_mrr, ("mrr",)),
            ("ndcg", self.min_ndcg, ("ndcg_at_k", "ndcg")),
            ("map", self.min_map, ("map_at_k", "map")),
            ("pr_auc", self.min_pr_auc, ("pr_auc",)),
            ("success_at_k", self.min_success_at_k, ("success_at_k",)),
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

#: Mirrors production agent knobs (heuristic threshold, re-rank blend, min relevant).
AGENT_ALIGNED_EVAL_PARAMS = EvalParams(
    k=3,
    ks=(1, 3, 6),
    relevance_threshold=0.35,
    thresholds=(0.25, 0.35, 0.5, 0.65, 0.8),
    judge_weight=0.65,
    retrieval_weight=0.35,
    min_relevant=3,
    min_hit_rate=0.4,
    include_per_case=True,
    include_ndcg=True,
    include_map=True,
    leave_one_out=False,
)

#: Stricter offline gate set for regression dashboards.
STRICT_EVAL_PARAMS = EvalParams(
    k=3,
    ks=(1, 3, 5),
    min_accuracy=0.5,
    min_precision=0.3,
    min_recall=0.3,
    min_f1=0.3,
    min_hit_rate=0.5,
    min_mrr=0.3,
    min_success_at_k=0.2,
    include_per_case=True,
)
