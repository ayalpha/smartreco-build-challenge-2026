"""Tests for offline eval metrics, parameters, reporting and retrieval harness.

Covers accuracy / precision / recall / F1 (binary + micro/macro/weighted),
ranking@k metrics, configurable :class:`~app.evals.config.EvalParams`, and an
end-to-end retrieval eval against the sample catalog.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.evals.config import DEFAULT_EVAL_PARAMS, EvalParams
from app.evals.datasets import (
    GOLDEN_RETRIEVAL_CASES,
    RetrievalCase,
    filter_cases,
    resolve_cases,
)
from app.evals.metrics import (
    classification_metrics,
    confusion_counts,
    multilabel_sets_to_binary_vectors,
    ranking_metrics_at_k,
    scores_to_labels,
)
from app.evals.report import format_metrics_report, metrics_to_json, metrics_to_table
from app.evals.runner import run_classification_eval, run_retrieval_eval
from app.models.product import Product


# --------------------------------------------------------------------------- #
# Classification metrics — exact hand-checked fixtures                         #
# --------------------------------------------------------------------------- #


class TestConfusionAndBinaryMetrics:
    """Binary formulas: Acc=(TP+TN)/N, P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)."""

    def test_perfect_predictions_score_one(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        report = classification_metrics(y_true, y_pred, average="binary")

        assert report.accuracy == pytest.approx(1.0)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.f1 == pytest.approx(1.0)
        assert report.confusion == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_hand_checked_mixed_case(self) -> None:
        # TP=1, FP=1, TN=1, FN=1  → Acc=0.5, P=0.5, R=0.5, F1=0.5
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        counts = confusion_counts(y_true, y_pred, positive_label=1)
        assert counts == counts.__class__(tp=1, fp=1, tn=1, fn=1)

        report = classification_metrics(y_true, y_pred, average="binary")
        assert report.accuracy == pytest.approx(0.5)
        assert report.precision == pytest.approx(0.5)
        assert report.recall == pytest.approx(0.5)
        assert report.f1 == pytest.approx(0.5)

    def test_all_negative_predictions_zero_precision_and_recall(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        report = classification_metrics(
            y_true, y_pred, average="binary", zero_division=0.0
        )
        assert report.precision == pytest.approx(0.0)  # TP+FP = 0
        assert report.recall == pytest.approx(0.0)  # no TP
        assert report.accuracy == pytest.approx(0.5)  # TN=2 / 4
        assert report.f1 == pytest.approx(0.0)

    def test_zero_division_override(self) -> None:
        report = classification_metrics(
            [0, 0], [0, 0], average="binary", zero_division=1.0
        )
        # No positive predictions and no positive labels → P/R use zero_division.
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            classification_metrics([1, 0], [1], average="binary")

    def test_scores_to_labels_respects_threshold_parameter(self) -> None:
        labels = scores_to_labels([0.2, 0.5, 0.9], threshold=0.5)
        assert labels == [0, 1, 1]
        labels_strict = scores_to_labels([0.2, 0.5, 0.9], threshold=0.8)
        assert labels_strict == [0, 0, 1]


class TestMultiClassAverages:
    """micro / macro / weighted over a 3-class fixture."""

    #: y_true:  A A B B C
    #: y_pred:  A B A B C
    #: correct: ✓ ✗ ✗ ✓ ✓  → accuracy = 3/5 = 0.6
    Y_TRUE = ["A", "A", "B", "B", "C"]
    Y_PRED = ["A", "B", "A", "B", "C"]

    def test_overall_accuracy_is_label_equality_rate(self) -> None:
        report = classification_metrics(
            self.Y_TRUE, self.Y_PRED, average="macro"
        )
        assert report.accuracy == pytest.approx(0.6)

    def test_micro_precision_equals_accuracy_for_single_label(self) -> None:
        # For single-label multi-class, micro-P = micro-R = accuracy.
        report = classification_metrics(
            self.Y_TRUE, self.Y_PRED, average="micro"
        )
        assert report.precision == pytest.approx(0.6)
        assert report.recall == pytest.approx(0.6)
        assert report.f1 == pytest.approx(0.6)

    def test_macro_is_unweighted_mean_of_per_class_f1(self) -> None:
        report = classification_metrics(
            self.Y_TRUE, self.Y_PRED, average="macro"
        )
        # Per-class (one-vs-rest):
        # A: TP1 FP1 FN1 → P=R=F1=0.5
        # B: TP1 FP1 FN1 → P=R=F1=0.5
        # C: TP1 FP0 FN0 → P=R=F1=1.0
        # macro F1 = (0.5+0.5+1.0)/3 = 2/3
        assert report.f1 == pytest.approx(2.0 / 3.0)
        assert report.precision == pytest.approx(2.0 / 3.0)
        assert report.recall == pytest.approx(2.0 / 3.0)
        assert "A" in report.per_class
        assert report.per_class["C"]["f1"] == pytest.approx(1.0)

    def test_weighted_prefers_majority_classes(self) -> None:
        report = classification_metrics(
            self.Y_TRUE, self.Y_PRED, average="weighted"
        )
        # supports A=2,B=2,C=1; f1s 0.5,0.5,1.0 → (1+1+1)/5 = 0.6
        assert report.f1 == pytest.approx(0.6)

    def test_unknown_average_raises(self) -> None:
        with pytest.raises(ValueError, match="average must be"):
            classification_metrics([1], [1], average="geometric")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Ranking metrics @k                                                           #
# --------------------------------------------------------------------------- #


class TestRankingMetricsAtK:
    """Precision@k, Recall@k, Hit@k, Accuracy@k, MRR, F1@k."""

    def test_perfect_single_query(self) -> None:
        report = ranking_metrics_at_k(
            retrieved=[[10, 20, 30]],
            relevant=[{10, 20}],
            k=2,
            catalog_size=5,
        )
        assert report.precision_at_k == pytest.approx(1.0)
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.f1_at_k == pytest.approx(1.0)
        assert report.hit_at_k == pytest.approx(1.0)
        assert report.mrr == pytest.approx(1.0)
        # TP=2, FP=0, FN=0, TN=3 → acc = 5/5
        assert report.accuracy_at_k == pytest.approx(1.0)

    def test_partial_hit_and_mrr(self) -> None:
        # Relevant doc is at rank 2 → MRR = 0.5; P@1 = 0; R@1 = 0; hit@1 = 0
        report = ranking_metrics_at_k(
            retrieved=[[99, 7, 3]],
            relevant=[{7}],
            k=1,
        )
        assert report.precision_at_k == pytest.approx(0.0)
        assert report.hit_at_k == pytest.approx(0.0)
        assert report.mrr == pytest.approx(0.5)

        report_k2 = ranking_metrics_at_k(
            retrieved=[[99, 7, 3]],
            relevant=[{7}],
            k=2,
        )
        assert report_k2.precision_at_k == pytest.approx(0.5)
        assert report_k2.recall_at_k == pytest.approx(1.0)
        assert report_k2.hit_at_k == pytest.approx(1.0)

    def test_mean_over_multiple_queries(self) -> None:
        report = ranking_metrics_at_k(
            retrieved=[[1, 2], [9, 8]],
            relevant=[{1}, {3}],
            k=1,
        )
        # Q1 hit, Q2 miss → hit_rate 0.5; P@1 = (1 + 0)/2 = 0.5
        assert report.hit_at_k == pytest.approx(0.5)
        assert report.precision_at_k == pytest.approx(0.5)
        assert report.n_queries == 2

    def test_without_catalog_size_accuracy_equals_hit(self) -> None:
        report = ranking_metrics_at_k(
            retrieved=[[1, 2]],
            relevant=[{1}],
            k=1,
        )
        assert report.accuracy_at_k == report.hit_at_k == pytest.approx(1.0)

    def test_empty_inputs_use_zero_division(self) -> None:
        report = ranking_metrics_at_k([], [], k=3, zero_division=0.0)
        assert report.n_queries == 0
        assert report.precision_at_k == 0.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            ranking_metrics_at_k([[1]], [{1}], k=0)


class TestMultilabelFlatten:
    def test_flattens_to_aligned_binary_vectors(self) -> None:
        y_true, y_pred = multilabel_sets_to_binary_vectors(
            predicted_sets=[{1, 2}, {3}],
            gold_sets=[{1}, {3, 4}],
            label_universe=[1, 2, 3, 4],
        )
        # query0: gold [1,0,0,0] pred [1,1,0,0]
        # query1: gold [0,0,1,1] pred [0,0,1,0]
        assert y_true == [1, 0, 0, 0, 0, 0, 1, 1]
        assert y_pred == [1, 1, 0, 0, 0, 0, 1, 0]


# --------------------------------------------------------------------------- #
# EvalParams                                                                   #
# --------------------------------------------------------------------------- #


class TestEvalParams:
    def test_defaults_are_frozen_and_serialisable(self) -> None:
        params = DEFAULT_EVAL_PARAMS
        assert params.k == 3
        assert params.effective_ks() == (1, 3, 5)
        payload = params.to_dict()
        assert payload["k"] == 3
        assert payload["ks"] == [1, 3, 5]
        assert json.dumps(payload)  # must be JSON-safe

    def test_with_updates_returns_new_instance(self) -> None:
        base = EvalParams(k=3)
        updated = base.with_updates(k=5, min_precision=0.4, split="test")
        assert base.k == 3
        assert updated.k == 5
        assert updated.min_precision == 0.4
        assert updated.split == "test"

    def test_validation_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            EvalParams(k=0)

    def test_validation_rejects_bad_threshold(self) -> None:
        with pytest.raises(ValueError, match="relevance_threshold"):
            EvalParams(relevance_threshold=1.5)

    def test_passes_gates_reports_failures(self) -> None:
        params = EvalParams(min_precision=0.9, min_recall=0.9, min_accuracy=0.9)
        ok, failures = params.passes_gates(
            {"precision": 0.5, "recall": 0.95, "accuracy": 0.2}
        )
        assert ok is False
        assert any("precision" in f for f in failures)
        assert any("accuracy" in f for f in failures)
        assert not any("recall" in f for f in failures)

    def test_passes_gates_accepts_ranking_aliases(self) -> None:
        params = EvalParams(min_hit_rate=0.5, min_precision=0.3)
        ok, failures = params.passes_gates(
            {"hit_rate": 0.75, "precision_at_k": 0.4}
        )
        assert ok is True
        assert failures == []


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #


class TestReportFormatting:
    def test_metrics_to_json_is_sorted_and_round_trippable(self) -> None:
        payload = {"recall": 0.5, "accuracy": 0.9, "precision": 0.7}
        text = metrics_to_json(payload)
        assert json.loads(text) == payload
        # sort_keys → accuracy before precision before recall
        assert text.index("accuracy") < text.index("precision")

    def test_table_and_report_include_core_metrics(self) -> None:
        metrics = {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "passed_gates": True,
            "params": {"k": 3},
            "by_k": {"1": {"precision_at_k": 0.5}, "3": {"precision_at_k": 0.7}},
        }
        table = metrics_to_table(metrics)
        assert "accuracy" in table
        assert "0.8000" in table

        report = format_metrics_report(metrics, title="Unit report")
        assert "Unit report" in report
        assert "Parameters" in report
        assert "Metrics by k" in report
        assert "Gates: PASSED" in report

    def test_json_report_mode(self) -> None:
        text = format_metrics_report({"accuracy": 1.0}, as_json=True)
        assert json.loads(text)["accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# Classification eval runner (threshold parameter path)                        #
# --------------------------------------------------------------------------- #


class TestClassificationEvalRunner:
    def test_thresholded_scores_flow_into_metrics(self) -> None:
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.2, 0.8, 0.1]
        # threshold 0.5 → pred [1, 0, 1, 0] → Acc/P/R/F1 = 0.5
        metrics = run_classification_eval(
            y_true,
            y_pred=[0, 0, 0, 0],  # ignored when scores provided
            scores=scores,
            params=EvalParams(
                average="binary",
                relevance_threshold=0.5,
                min_accuracy=0.4,
            ),
        )
        assert metrics["task"] == "classification"
        assert metrics["accuracy"] == pytest.approx(0.5)
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.5)
        assert "by_average" in metrics
        assert set(metrics["by_average"]) >= {"binary", "micro", "macro", "weighted"}
        assert metrics["passed_gates"] is True

    def test_stricter_threshold_changes_predictions(self) -> None:
        y_true = [1, 0]
        scores = [0.6, 0.55]
        loose = run_classification_eval(
            y_true,
            y_pred=[],
            scores=scores,
            params=EvalParams(relevance_threshold=0.5),
        )
        strict = run_classification_eval(
            y_true,
            y_pred=[],
            scores=scores,
            params=EvalParams(relevance_threshold=0.7),
        )
        # loose predicts [1, 1]; strict predicts [0, 0]
        assert loose["precision"] == pytest.approx(0.5)
        assert strict["precision"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Dataset helpers                                                              #
# --------------------------------------------------------------------------- #


class TestDatasets:
    def test_filter_cases_by_split_and_tag_and_limit(self) -> None:
        train = filter_cases(GOLDEN_RETRIEVAL_CASES, split="train")
        test = filter_cases(GOLDEN_RETRIEVAL_CASES, split="test")
        assert train and test
        assert all(c.split == "train" for c in train)
        assert all(c.split == "test" for c in test)

        agentic = filter_cases(GOLDEN_RETRIEVAL_CASES, tag="agentic")
        assert agentic
        assert all("agentic" in c.tags for c in agentic)

        limited = filter_cases(GOLDEN_RETRIEVAL_CASES, limit=2)
        assert len(limited) == 2

    def test_filter_by_case_ids(self) -> None:
        selected = filter_cases(
            GOLDEN_RETRIEVAL_CASES, case_ids=("langgraph-agents", "kubernetes")
        )
        assert {c.id for c in selected} == {"langgraph-agents", "kubernetes"}

    def test_resolve_cases_against_sample_catalog(
        self, db: Session, products: list[Product]
    ) -> None:
        resolved = resolve_cases(db, GOLDEN_RETRIEVAL_CASES)
        assert len(resolved) >= 6
        for case in resolved:
            assert case["relevant_ids"], case["id"]
            assert all(isinstance(i, int) for i in case["relevant_ids"])


# --------------------------------------------------------------------------- #
# End-to-end retrieval eval                                                    #
# --------------------------------------------------------------------------- #


class TestRetrievalEvalHarness:
    """Hybrid retrieval scored with full metric suite + configurable params."""

    def test_run_retrieval_eval_returns_core_metrics(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=3,
            ks=(1, 3),
            split="train",
            retrieval_mode="hybrid",
            include_per_case=True,
            min_hit_rate=0.3,
        )
        metrics = run_retrieval_eval(db, params=params)

        assert metrics["task"] == "retrieval"
        assert metrics["n_cases"] >= 1
        assert metrics["catalog_size"] == len(products)
        assert metrics["k"] == 3

        for key in (
            "accuracy_at_k",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "hit_at_k",
            "mrr",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            assert key in metrics, key
            assert 0.0 <= float(metrics[key]) <= 1.0 + 1e-9

        assert "1" in metrics["by_k"] and "3" in metrics["by_k"]
        assert "classification" in metrics
        for avg in ("binary", "micro", "macro", "weighted"):
            block = metrics["classification"][avg]
            for metric_name in ("accuracy", "precision", "recall", "f1"):
                assert metric_name in block

        assert metrics["precision_micro"] == metrics["classification"]["micro"]["precision"]
        assert isinstance(metrics["per_case"], list)
        assert metrics["per_case"][0]["id"]
        assert "passed_gates" in metrics

    def test_keyword_mode_and_case_id_filter(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=1,
            ks=(1,),
            case_ids=("langgraph-agents", "kubernetes"),
            retrieval_mode="keyword",
            include_per_case=True,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["n_cases"] == 2
        assert metrics["params"]["retrieval_mode"] == "keyword"
        # Keyword search should hit the obvious matches at rank 1.
        assert metrics["hit_at_k"] == pytest.approx(1.0)
        assert metrics["precision_at_k"] == pytest.approx(1.0)

    def test_split_test_runs_paraphrase_cases(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(k=3, ks=(3,), split="test", include_per_case=False)
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["n_cases"] >= 1
        assert "per_case" not in metrics

    def test_custom_retriever_injection(
        self, db: Session, products: list[Product]
    ) -> None:
        """Inject a perfect retriever to prove metrics wiring is independent of IR."""
        resolved = resolve_cases(db, filter_cases(GOLDEN_RETRIEVAL_CASES, split="train"))

        def perfect(
            _db: Session, query: str, limit: int, _filters: Any
        ) -> list[int]:
            for case in resolved:
                if case["query"] == query:
                    return list(case["relevant_ids"])[:limit]
            return []

        params = EvalParams(k=1, ks=(1,), split="train", include_per_case=False)
        metrics = run_retrieval_eval(db, params=params, retriever=perfect)
        assert metrics["hit_at_k"] == pytest.approx(1.0)
        assert metrics["precision_at_k"] == pytest.approx(1.0)

    def test_report_string_is_non_empty(
        self, db: Session, products: list[Product]
    ) -> None:
        from app.evals.runner import report_retrieval_eval

        text = report_retrieval_eval(
            db,
            params=EvalParams(
                k=3,
                ks=(3,),
                case_ids=("langgraph-agents",),
                include_per_case=False,
            ),
        )
        assert "Retrieval eval" in text
        assert "precision" in text.lower() or "precision_at_k" in text

    def test_gate_failure_when_thresholds_impossible(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=3,
            case_ids=("langgraph-agents",),
            min_precision=1.01,  # impossible
            include_per_case=False,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["passed_gates"] is False
        assert metrics["gate_failures"]


class TestLazyPackageExports:
    def test_package_exports_resolve(self) -> None:
        import app.evals as evals

        assert evals.EvalParams is EvalParams
        assert callable(evals.classification_metrics)
        assert callable(evals.ranking_metrics_at_k)
        assert callable(evals.run_retrieval_eval)
        assert callable(evals.format_metrics_report)
