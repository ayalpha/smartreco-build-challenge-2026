"""Dependency-free metrics for evaluating ranked, multi-label recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationMetrics:
    examples: int
    labels: list[str]
    accuracy: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_predictions(
    predictions: Sequence[str] | Mapping[str, float],
    *,
    k: int | None = None,
    threshold: float | None = None,
) -> set[str]:
    """Select labels from an ordered list or a label-to-score mapping."""
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    if isinstance(predictions, Mapping):
        ranked = sorted(predictions.items(), key=lambda item: (-item[1], item[0]))
        if threshold is not None:
            ranked = [item for item in ranked if item[1] >= threshold]
        labels = [label for label, _ in ranked]
    else:
        labels = list(predictions)
    return set(labels[:k] if k is not None else labels)


def evaluate(
    expected: Sequence[Iterable[str]],
    predicted: Sequence[Sequence[str] | Mapping[str, float]],
    *,
    k: int | None = None,
    threshold: float | None = None,
    labels: Iterable[str] | None = None,
) -> EvaluationMetrics:
    """Compute subset accuracy and micro/macro precision, recall and F1."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must contain the same number of examples")
    truth = [set(values) for values in expected]
    guesses = [select_predictions(values, k=k, threshold=threshold) for values in predicted]
    label_set = set(labels or ())
    if labels is None:
        label_set.update(label for row in truth + guesses for label in row)
    else:
        unknown = set().union(*truth, *guesses) - label_set if truth else set()
        if unknown:
            raise ValueError(f"labels missing from configured label set: {sorted(unknown)}")
    ordered_labels = sorted(label_set)

    def safe_div(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    per_label: list[tuple[float, float, float]] = []
    total_tp = total_fp = total_fn = 0
    for label in ordered_labels:
        tp = sum(label in actual and label in guess for actual, guess in zip(truth, guesses))
        fp = sum(label not in actual and label in guess for actual, guess in zip(truth, guesses))
        fn = sum(label in actual and label not in guess for actual, guess in zip(truth, guesses))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        per_label.append((precision, recall, f1))
        total_tp += tp
        total_fp += fp
        total_fn += fn

    count = len(truth)
    macro = lambda index: (sum(row[index] for row in per_label) / len(per_label)) if per_label else 0.0
    return EvaluationMetrics(
        examples=count,
        labels=ordered_labels,
        accuracy=safe_div(sum(a == b for a, b in zip(truth, guesses)), count),
        precision_micro=safe_div(total_tp, total_tp + total_fp),
        recall_micro=safe_div(total_tp, total_tp + total_fn),
        f1_micro=safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        precision_macro=macro(0), recall_macro=macro(1), f1_macro=macro(2),
    )
