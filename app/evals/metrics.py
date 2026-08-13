"""Classification and ranking metrics for offline evals.

Formulas (standard IR / sklearn definitions)
--------------------------------------------
Let TP, FP, TN, FN be confusion counts for a chosen positive class.

* **Accuracy**  = (TP + TN) / (TP + TN + FP + FN)
* **Precision** = TP / (TP + FP)          (0 when denominator is 0)
* **Recall**    = TP / (TP + FN)          (0 when denominator is 0)
* **F1**        = 2 · P · R / (P + R)    (0 when P = R = 0)

Multi-class averages
--------------------
* **micro** — pool TP/FP/FN across classes, then compute P/R/F1 once.
* **macro** — unweighted mean of per-class P/R/F1.
* **weighted** — support-weighted mean of per-class P/R/F1.
* **binary** — scores for ``positive_label`` only.

Ranking at cutoff k (set-based relevance)
-----------------------------------------
Given retrieved ranked ids ``R`` and relevant id set ``G``:

* **Precision@k** = |R[:k] ∩ G| / k
* **Recall@k**    = |R[:k] ∩ G| / |G|     (0 if G empty)
* **Hit@k**       = 1 if |R[:k] ∩ G| > 0 else 0
* **Accuracy@k**  = Hit@k for single-label queries; for multi-label we report
  the fraction of catalog items whose relevant/non-relevant status is correct
  in the top-k *prediction set* vs full label set when ``catalog_size`` is
  provided: (TP + TN) / N over the binary-relevance view at k.
* **MRR**         = 1 / rank_of_first_relevant (0 if none)
* **F1@k**        = harmonic mean of Precision@k and Recall@k
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Union

Label = Hashable
BinaryLike = Union[int, bool, str]

_DEFAULT_ZERO = 0.0


@dataclass(frozen=True)
class ConfusionCounts:
    """Binary confusion matrix cells."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        """Number of true positives + false negatives (positive support)."""
        return self.tp + self.fn

    @property
    def total(self) -> int:
        """All labelled examples."""
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationReport:
    """Bundled classification metrics for one evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    average: str
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: Optional[dict[str, int]] = None
    zero_division: float = _DEFAULT_ZERO

    def to_dict(self) -> dict[str, Any]:
        """Flat + nested metrics dict used by reports and gates."""
        payload: dict[str, Any] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "average": self.average,
            "zero_division": self.zero_division,
        }
        if self.per_class:
            payload["per_class"] = self.per_class
        if self.confusion is not None:
            payload["confusion"] = self.confusion
        return payload


@dataclass(frozen=True)
class RankingReport:
    """Ranking metrics at a single cutoff k."""

    k: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_at_k: float
    accuracy_at_k: float
    mrr: float
    n_queries: int
    mean_relevant: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "hit_at_k": self.hit_at_k,
            "hit_rate": self.hit_at_k,  # alias used by gates
            "accuracy_at_k": self.accuracy_at_k,
            "accuracy": self.accuracy_at_k,  # alias for gate helpers
            "precision": self.precision_at_k,
            "recall": self.recall_at_k,
            "f1": self.f1_at_k,
            "mrr": self.mrr,
            "n_queries": self.n_queries,
            "mean_relevant": self.mean_relevant,
        }


def _safe_div(numerator: float, denominator: float, zero_division: float) -> float:
    if denominator == 0:
        return float(zero_division)
    return numerator / denominator


def _f1(precision: float, recall: float, zero_division: float) -> float:
    return fbeta_score(precision, recall, beta=1.0, zero_division=zero_division)


def fbeta_score(
    precision: float,
    recall: float,
    *,
    beta: float = 1.0,
    zero_division: float = _DEFAULT_ZERO,
) -> float:
    """F-beta = (1+β²)·P·R / (β²·P + R).

    β=1 → F1 (balanced). β>1 weights recall higher (e.g. F2); β<1 weights
    precision higher (e.g. F0.5).
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if precision == 0.0 and recall == 0.0:
        return float(zero_division)
    beta2 = beta * beta
    return _safe_div(
        (1.0 + beta2) * precision * recall,
        beta2 * precision + recall,
        zero_division,
    )


def confusion_counts(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    *,
    positive_label: Label = 1,
) -> ConfusionCounts:
    """Binary confusion counts treating ``positive_label`` as the positive class.

    All other labels are negatives. Lengths must match.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        is_pos = truth == positive_label
        pred_pos = pred == positive_label
        if is_pos and pred_pos:
            tp += 1
        elif not is_pos and pred_pos:
            fp += 1
        elif not is_pos and not pred_pos:
            tn += 1
        else:
            fn += 1
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _binary_scores(
    counts: ConfusionCounts, *, zero_division: float
) -> tuple[float, float, float, float]:
    accuracy = _safe_div(counts.tp + counts.tn, counts.total, zero_division)
    precision = _safe_div(counts.tp, counts.tp + counts.fp, zero_division)
    recall = _safe_div(counts.tp, counts.tp + counts.fn, zero_division)
    f1 = _f1(precision, recall, zero_division)
    return accuracy, precision, recall, f1


def _labels_to_str(label: Label) -> str:
    return str(label)


def classification_metrics(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    *,
    average: str = "binary",
    positive_label: Label = 1,
    labels: Optional[Sequence[Label]] = None,
    zero_division: float = _DEFAULT_ZERO,
) -> ClassificationReport:
    """Compute accuracy, precision, recall and F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (same length).
        average: ``binary`` | ``micro`` | ``macro`` | ``weighted``.
        positive_label: Positive class for ``average="binary"``.
        labels: Optional explicit label set (defaults to sorted unique labels).
        zero_division: Value when a denominator is zero.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true/y_pred length mismatch: {len(y_true)} vs {len(y_pred)}"
        )
    if not y_true:
        empty = ClassificationReport(
            accuracy=float(zero_division),
            precision=float(zero_division),
            recall=float(zero_division),
            f1=float(zero_division),
            support=0,
            average=average,
            zero_division=zero_division,
        )
        return empty

    if average == "binary":
        counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
        accuracy, precision, recall, f1 = _binary_scores(
            counts, zero_division=zero_division
        )
        return ClassificationReport(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            support=counts.support,
            average=average,
            confusion=counts.to_dict(),
            zero_division=zero_division,
        )

    # Multi-class path -------------------------------------------------------
    if labels is None:
        label_set = sorted(set(y_true) | set(y_pred), key=_labels_to_str)
    else:
        label_set = list(labels)

    per_class: dict[str, dict[str, float]] = {}
    # One-vs-rest counts per class
    class_counts: dict[Label, ConfusionCounts] = {}
    for label in label_set:
        c = confusion_counts(y_true, y_pred, positive_label=label)
        class_counts[label] = c
        acc, prec, rec, f1 = _binary_scores(c, zero_division=zero_division)
        per_class[_labels_to_str(label)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(c.support),
            "accuracy": acc,
        }

    # Overall accuracy is always micro-style correct/total (label equality).
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = _safe_div(correct, len(y_true), zero_division)

    if average == "micro":
        # Micro: sum TP/FP/FN across classes. For single-label multi-class,
        # sum(TP)=correct and sum(FP)=sum(FN)=n-correct, so P=R=F1=accuracy.
        tp = sum(c.tp for c in class_counts.values())
        fp = sum(c.fp for c in class_counts.values())
        fn = sum(c.fn for c in class_counts.values())
        precision = _safe_div(tp, tp + fp, zero_division)
        recall = _safe_div(tp, tp + fn, zero_division)
        f1 = _f1(precision, recall, zero_division)
        support = len(y_true)
    elif average in ("macro", "weighted"):
        precisions = [per_class[_labels_to_str(lbl)]["precision"] for lbl in label_set]
        recalls = [per_class[_labels_to_str(lbl)]["recall"] for lbl in label_set]
        f1s = [per_class[_labels_to_str(lbl)]["f1"] for lbl in label_set]
        supports = [class_counts[lbl].support for lbl in label_set]
        if average == "macro":
            n = len(label_set) or 1
            precision = sum(precisions) / n
            recall = sum(recalls) / n
            f1 = sum(f1s) / n
            support = sum(supports)
        else:
            total_support = sum(supports)
            if total_support == 0:
                precision = recall = f1 = float(zero_division)
            else:
                precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
                recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
                f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
            support = total_support
    else:
        raise ValueError(
            f"average must be binary|micro|macro|weighted, got {average!r}"
        )

    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        average=average,
        per_class=per_class,
        zero_division=zero_division,
    )


def scores_to_labels(
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    positive_label: Label = 1,
    negative_label: Label = 0,
) -> list[Label]:
    """Binarise continuous scores with ``score >= threshold → positive``."""
    return [
        positive_label if float(score) >= threshold else negative_label
        for score in scores
    ]


def ranking_metrics_at_k(
    retrieved: Sequence[Sequence[Label]],
    relevant: Sequence[Iterable[Label]],
    *,
    k: int = 3,
    catalog_size: Optional[int] = None,
    zero_division: float = _DEFAULT_ZERO,
) -> RankingReport:
    """Aggregate ranking metrics over many queries at cutoff ``k``.

    Args:
        retrieved: Per-query ranked id lists (best first).
        relevant: Per-query iterable of ground-truth relevant ids.
        k: Cutoff.
        catalog_size: If provided, accuracy@k uses a full-catalog binary view:
            predicted-positive = top-k ids; true-positive = relevant set.
            TN = catalog items neither retrieved nor relevant.
            When omitted, accuracy@k equals hit@k (single-label convention).
        zero_division: Denominator-zero fallback.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(retrieved) != len(relevant):
        raise ValueError(
            f"retrieved/relevant length mismatch: {len(retrieved)} vs {len(relevant)}"
        )

    n = len(retrieved)
    if n == 0:
        z = float(zero_division)
        return RankingReport(
            k=k,
            precision_at_k=z,
            recall_at_k=z,
            f1_at_k=z,
            hit_at_k=z,
            accuracy_at_k=z,
            mrr=z,
            n_queries=0,
            mean_relevant=0.0,
        )

    precisions: list[float] = []
    recalls: list[float] = []
    hits: list[float] = []
    accuracies: list[float] = []
    mrrs: list[float] = []
    relevant_sizes: list[int] = []

    for ranked, gold in zip(retrieved, relevant):
        gold_set = set(gold)
        relevant_sizes.append(len(gold_set))
        top = list(ranked)[:k]
        top_set = set(top)
        tp = len(top_set & gold_set)

        precision = _safe_div(tp, k, zero_division)
        recall = _safe_div(tp, len(gold_set), zero_division) if gold_set else float(zero_division)
        hit = 1.0 if tp > 0 else 0.0

        if catalog_size is not None and catalog_size > 0:
            # Binary relevance over the whole catalog at prediction set = top-k.
            fp = len(top_set - gold_set)
            fn = len(gold_set - top_set)
            tn = max(catalog_size - tp - fp - fn, 0)
            accuracy = _safe_div(tp + tn, catalog_size, zero_division)
        else:
            accuracy = hit

        # MRR: reciprocal rank of first relevant item in the full ranking.
        rr = 0.0
        for idx, doc_id in enumerate(ranked, start=1):
            if doc_id in gold_set:
                rr = 1.0 / idx
                break

        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)
        accuracies.append(accuracy)
        mrrs.append(rr)

    mean_p = sum(precisions) / n
    mean_r = sum(recalls) / n
    mean_hit = sum(hits) / n
    mean_acc = sum(accuracies) / n
    mean_mrr = sum(mrrs) / n
    mean_f1 = _f1(mean_p, mean_r, zero_division)

    return RankingReport(
        k=k,
        precision_at_k=mean_p,
        recall_at_k=mean_r,
        f1_at_k=mean_f1,
        hit_at_k=mean_hit,
        accuracy_at_k=mean_acc,
        mrr=mean_mrr,
        n_queries=n,
        mean_relevant=sum(relevant_sizes) / n,
    )


def per_query_ranking(
    ranked_ids: Sequence[Label],
    relevant_ids: Iterable[Label],
    *,
    k: int,
    catalog_size: Optional[int] = None,
    zero_division: float = _DEFAULT_ZERO,
) -> dict[str, float]:
    """Single-query ranking metrics (used for per-case reports)."""
    report = ranking_metrics_at_k(
        [ranked_ids],
        [relevant_ids],
        k=k,
        catalog_size=catalog_size,
        zero_division=zero_division,
    )
    return {
        "precision_at_k": report.precision_at_k,
        "recall_at_k": report.recall_at_k,
        "f1_at_k": report.f1_at_k,
        "hit_at_k": report.hit_at_k,
        "accuracy_at_k": report.accuracy_at_k,
        "mrr": report.mrr,
    }


def multilabel_sets_to_binary_vectors(
    predicted_sets: Sequence[Iterable[Label]],
    gold_sets: Sequence[Iterable[Label]],
    label_universe: Sequence[Label],
) -> tuple[list[int], list[int]]:
    """Flatten multi-label set predictions into aligned binary vectors.

    For each example and each label in ``label_universe``, appends 1/0 for
    gold and prediction. Useful for micro/macro multi-label metrics.
    """
    if len(predicted_sets) != len(gold_sets):
        raise ValueError("predicted_sets/gold_sets length mismatch")
    y_true: list[int] = []
    y_pred: list[int] = []
    for pred, gold in zip(predicted_sets, gold_sets):
        pred_set = set(pred)
        gold_set = set(gold)
        for label in label_universe:
            y_true.append(1 if label in gold_set else 0)
            y_pred.append(1 if label in pred_set else 0)
    return y_true, y_pred


def specificity_from_confusion(
    counts: ConfusionCounts, *, zero_division: float = _DEFAULT_ZERO
) -> float:
    """Specificity = TN / (TN + FP)."""
    return _safe_div(counts.tn, counts.tn + counts.fp, zero_division)


def balanced_accuracy_from_confusion(
    counts: ConfusionCounts, *, zero_division: float = _DEFAULT_ZERO
) -> float:
    """Balanced accuracy = (sensitivity + specificity) / 2.

    Sensitivity is recall = TP/(TP+FN). Useful under class imbalance where
    plain accuracy is dominated by the majority class.
    """
    sensitivity = _safe_div(counts.tp, counts.tp + counts.fn, zero_division)
    specificity = specificity_from_confusion(counts, zero_division=zero_division)
    return 0.5 * (sensitivity + specificity)


def threshold_sweep_metrics(
    y_true: Sequence[Label],
    scores: Sequence[float],
    *,
    thresholds: Sequence[float],
    positive_label: Label = 1,
    negative_label: Label = 0,
    zero_division: float = _DEFAULT_ZERO,
) -> list[dict[str, Any]]:
    """Accuracy / precision / recall / F1 at each score threshold.

    Returns one dict per threshold (sorted ascending), suitable for plotting a
    discrete precision–recall / F1 curve without extra dependencies.
    """
    if len(y_true) != len(scores):
        raise ValueError(
            f"y_true/scores length mismatch: {len(y_true)} vs {len(scores)}"
        )
    rows: list[dict[str, Any]] = []
    for threshold in sorted(float(t) for t in thresholds):
        y_pred = scores_to_labels(
            scores,
            threshold=threshold,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        report = classification_metrics(
            y_true,
            y_pred,
            average="binary",
            positive_label=positive_label,
            zero_division=zero_division,
        )
        counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": report.accuracy,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "f0_5": fbeta_score(
                    report.precision, report.recall, beta=0.5, zero_division=zero_division
                ),
                "f2": fbeta_score(
                    report.precision, report.recall, beta=2.0, zero_division=zero_division
                ),
                "specificity": specificity_from_confusion(
                    counts, zero_division=zero_division
                ),
                "balanced_accuracy": balanced_accuracy_from_confusion(
                    counts, zero_division=zero_division
                ),
                "support": report.support,
                "confusion": counts.to_dict(),
            }
        )
    return rows


def best_threshold_by_metric(
    sweep_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "f1",
) -> dict[str, Any]:
    """Pick the sweep row with the highest ``metric`` (ties → lower threshold)."""
    if not sweep_rows:
        raise ValueError("sweep_rows is empty")
    return max(
        sweep_rows,
        key=lambda row: (float(row.get(metric, float("-inf"))), -float(row["threshold"])),
    )


def ndcg_at_k(
    ranked_ids: Sequence[Label],
    relevant_ids: Iterable[Label],
    *,
    k: int,
    zero_division: float = _DEFAULT_ZERO,
) -> float:
    """Binary-relevance nDCG@k.

    DCG@k = Σ_{i=1..k} rel_i / log2(i+1) with rel_i ∈ {0,1}.
    IDCG@k uses the ideal ranking that puts all |G| relevants first.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gold = set(relevant_ids)
    if not gold:
        return float(zero_division)

    def _dcg(rels: Sequence[int]) -> float:
        total = 0.0
        for idx, rel in enumerate(rels, start=1):
            if rel:
                total += 1.0 / _log2(idx + 1)
        return total

    gains = [1 if doc_id in gold else 0 for doc_id in list(ranked_ids)[:k]]
    dcg = _dcg(gains)
    ideal_count = min(len(gold), k)
    idcg = _dcg([1] * ideal_count)
    return _safe_div(dcg, idcg, zero_division)


def mean_ndcg_at_k(
    retrieved: Sequence[Sequence[Label]],
    relevant: Sequence[Iterable[Label]],
    *,
    k: int,
    zero_division: float = _DEFAULT_ZERO,
) -> float:
    """Mean nDCG@k over a query batch."""
    if len(retrieved) != len(relevant):
        raise ValueError("retrieved/relevant length mismatch")
    if not retrieved:
        return float(zero_division)
    scores = [
        ndcg_at_k(ranked, gold, k=k, zero_division=zero_division)
        for ranked, gold in zip(retrieved, relevant)
    ]
    return sum(scores) / len(scores)


def average_precision_at_k(
    ranked_ids: Sequence[Label],
    relevant_ids: Iterable[Label],
    *,
    k: int,
    zero_division: float = _DEFAULT_ZERO,
) -> float:
    """Average Precision@k for one query (binary relevance).

    AP@k = (1/min(|G|, k)) · Σ_{i=1..k} P@i · rel_i
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    gold = set(relevant_ids)
    if not gold:
        return float(zero_division)

    hits = 0
    sum_precisions = 0.0
    top = list(ranked_ids)[:k]
    for idx, doc_id in enumerate(top, start=1):
        if doc_id in gold:
            hits += 1
            sum_precisions += hits / idx
    denom = min(len(gold), k)
    return _safe_div(sum_precisions, denom, zero_division)


def mean_average_precision_at_k(
    retrieved: Sequence[Sequence[Label]],
    relevant: Sequence[Iterable[Label]],
    *,
    k: int,
    zero_division: float = _DEFAULT_ZERO,
) -> float:
    """MAP@k = mean of AP@k over queries."""
    if len(retrieved) != len(relevant):
        raise ValueError("retrieved/relevant length mismatch")
    if not retrieved:
        return float(zero_division)
    scores = [
        average_precision_at_k(ranked, gold, k=k, zero_division=zero_division)
        for ranked, gold in zip(retrieved, relevant)
    ]
    return sum(scores) / len(scores)


def classification_metrics_bundle(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    *,
    positive_label: Label = 1,
    zero_division: float = _DEFAULT_ZERO,
    betas: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Single-call binary accuracy/precision/recall/F1 plus F-beta variants.

    Useful for tests and score-based graders that want a flat dict without
    picking an ``average`` mode up front.
    """
    report = classification_metrics(
        y_true,
        y_pred,
        average="binary",
        positive_label=positive_label,
        zero_division=zero_division,
    )
    counts = confusion_counts(y_true, y_pred, positive_label=positive_label)
    payload: dict[str, Any] = {
        "accuracy": report.accuracy,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "specificity": specificity_from_confusion(counts, zero_division=zero_division),
        "balanced_accuracy": balanced_accuracy_from_confusion(
            counts, zero_division=zero_division
        ),
        "support": report.support,
        "confusion": counts.to_dict(),
    }
    for beta in betas:
        payload[_beta_key(float(beta))] = fbeta_score(
            report.precision, report.recall, beta=float(beta), zero_division=zero_division
        )
    return payload


def _beta_key(beta: float) -> str:
    """Stable metric name for a beta value (``f1``, ``f2``, ``f0_5``, …)."""
    if beta == 1.0:
        return "f1"
    if float(beta) == int(beta):
        return f"f{int(beta)}"
    return f"f{str(beta).replace('.', '_')}"


def metrics_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    keys: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """candidate − baseline for shared numeric scalar keys."""
    if keys is None:
        keys = (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "hit_at_k",
            "mrr",
            "ndcg_at_k",
            "map_at_k",
        )
    delta: dict[str, float] = {}
    for key in keys:
        if key in baseline and key in candidate:
            try:
                delta[key] = float(candidate[key]) - float(baseline[key])
            except (TypeError, ValueError):
                continue
    return delta


def _log2(value: float) -> float:
    """log2 without importing math (keeps the module stdlib-light)."""
    from math import log2

    return log2(value)
