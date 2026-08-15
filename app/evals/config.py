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
        min_specificity / min_balanced_accuracy / min_jaccard / min_cohen_kappa /
            min_roc_auc: Optional extended classification gates.
        include_extended_classification: When True, classification evals attach
            NPV, Jaccard, Hamming loss, Cohen's κ, and ROC-AUC (if scores).
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
    min_specificity: Optional[float] = None
    min_balanced_accuracy: Optional[float] = None
    min_jaccard: Optional[float] = None
    min_cohen_kappa: Optional[float] = None
    min_roc_auc: Optional[float] = None
    include_extended_classification: bool = True
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
    min_price: Optional[float] = None
    exclude_product_ids: Optional[tuple[int, ...]] = None
    calibration_bins: int = 10
    include_calibration: bool = True
    min_ece: Optional[float] = None  # gate: require ECE <= this (lower better)
    f_betas: tuple[float, ...] = (0.5, 2.0)
    compare_modes: Optional[tuple[str, ...]] = None
    #: Agent re-rank blend (matches app.agent.nodes._RERANK_*).
    judge_weight: float = 0.65
    retrieval_weight: float = 0.35
    #: Agent gate: need this many relevants in top-k (agent_min_relevant_products).
    min_relevant: int = 3
    min_success_at_k: Optional[float] = None
    #: Monte-Carlo random baseline trials (0 disables).
    random_baseline_trials: int = 0
    require_beat_random: bool = False
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
        if self.min_price is not None and self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")
        if self.calibration_bins < 1:
            raise ValueError(
                f"calibration_bins must be >= 1, got {self.calibration_bins}"
            )
        if self.judge_weight < 0 or self.retrieval_weight < 0:
            raise ValueError("judge_weight and retrieval_weight must be >= 0")
        if self.min_relevant < 1:
            raise ValueError(f"min_relevant must be >= 1, got {self.min_relevant}")
        if self.random_baseline_trials < 0:
            raise ValueError(
                f"random_baseline_trials must be >= 0, got {self.random_baseline_trials}"
            )

    def effective_ks(self) -> tuple[int, ...]:
        """Return the cutoffs to evaluate (``ks`` if set, else ``(k,)``)."""
        if self.ks:
            return self.ks
        return (self.k,)

    def with_updates(self, **changes: Any) -> "EvalParams":
        """Return a copy with selected fields replaced (immutable helper)."""
        return replace(self, **changes)

    @classmethod
    def from_settings(cls, settings: Any = None, **overrides: Any) -> "EvalParams":
        """Build params aligned with live app settings knobs.

        Maps:
        * ``vector_search_top_k`` → default ``k`` when not overridden
        * ``agent_min_relevant_products`` → ``min_relevant``
        * ``agent_final_product_count`` → included in ``ks`` sweep
        """
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        top_k = int(getattr(settings, "vector_search_top_k", 12) or 12)
        min_rel = int(getattr(settings, "agent_min_relevant_products", 3) or 3)
        final_n = int(getattr(settings, "agent_final_product_count", 6) or 6)
        primary_k = min(3, top_k)
        ks = tuple(sorted({1, primary_k, min(final_n, top_k), top_k}))
        base = cls(
            k=primary_k,
            ks=ks,
            min_relevant=min_rel,
            relevance_threshold=0.35,
            judge_weight=0.65,
            retrieval_weight=0.35,
            thresholds=(0.25, 0.35, 0.5, 0.65, 0.8),
            include_ndcg=True,
            include_map=True,
            include_calibration=True,
        )
        if overrides:
            return base.with_updates(**overrides)
        return base

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
            "exclude_product_ids",
        ):
            if data.get(key) is not None:
                data[key] = list(data[key])
        return data

    def search_filters(self) -> Optional[Any]:
        """Build a :class:`~app.vector_store.qdrant_client.SearchFilters` if set."""
        if (
            not self.skill_levels
            and not self.categories
            and self.max_price is None
            and self.min_price is None
            and not self.exclude_product_ids
        ):
            return None
        from app.vector_store.qdrant_client import SearchFilters

        return SearchFilters(
            skill_levels=list(self.skill_levels) if self.skill_levels else None,
            categories=list(self.categories) if self.categories else None,
            max_price=self.max_price,
            min_price=self.min_price,
            exclude_product_ids=list(self.exclude_product_ids)
            if self.exclude_product_ids
            else None,
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
                "exclude_product_ids",
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
            ("specificity", self.min_specificity, ("specificity",)),
            (
                "balanced_accuracy",
                self.min_balanced_accuracy,
                ("balanced_accuracy",),
            ),
            ("jaccard", self.min_jaccard, ("jaccard",)),
            ("cohen_kappa", self.min_cohen_kappa, ("cohen_kappa", "kappa")),
            ("roc_auc", self.min_roc_auc, ("roc_auc", "auc")),
        )
        for label, minimum, keys in checks:
            if minimum is None:
                continue
            value = _first_present(metrics, keys)
            if value is None:
                failures.append(f"{label}: missing (gate required >= {minimum})")
            elif float(value) < minimum:
                failures.append(f"{label}: {float(value):.4f} < {minimum}")
        # ECE is lower-is-better.
        if self.min_ece is not None:
            ece = _first_present(metrics, ("ece",))
            if ece is None and isinstance(metrics.get("calibration"), dict):
                try:
                    ece = float(metrics["calibration"].get("ece"))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    ece = None
            if ece is None:
                failures.append(f"ece: missing (gate required <= {self.min_ece})")
            elif float(ece) > self.min_ece:
                failures.append(f"ece: {float(ece):.4f} > {self.min_ece}")
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

#: Wide parameter surface for sensitivity / APRF matrix tests.
SWEEP_EVAL_PARAMS = EvalParams(
    k=3,
    ks=(1, 2, 3, 5, 6),
    thresholds=(0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8),
    relevance_threshold=0.5,
    average="binary",
    f_betas=(0.5, 1.5, 2.0),
    include_ndcg=True,
    include_map=True,
    include_calibration=True,
    include_extended_classification=True,
    calibration_bins=10,
    min_relevant=1,
    n_bootstrap=0,
)

#: Multi-average classification regression (micro/macro/weighted).
MULTI_AVERAGE_EVAL_PARAMS = EvalParams(
    average="macro",
    thresholds=(0.25, 0.5, 0.75),
    include_extended_classification=True,
    f_betas=(0.5, 2.0),
    min_accuracy=0.0,
    min_precision=0.0,
    min_recall=0.0,
    min_f1=0.0,
)
