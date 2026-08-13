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
    GOLDEN_CLASSIFICATION_FIXTURES,
    GOLDEN_MULTICLASS_FIXTURES,
    GOLDEN_RETRIEVAL_CASES,
    RetrievalCase,
    filter_cases,
    resolve_cases,
)
from app.evals.metrics import (
    average_precision_at_k,
    balanced_accuracy_from_confusion,
    best_threshold_by_metric,
    classification_metrics,
    classification_metrics_bundle,
    confusion_counts,
    fbeta_score,
    mean_average_precision_at_k,
    mean_ndcg_at_k,
    metrics_delta,
    multilabel_sets_to_binary_vectors,
    ndcg_at_k,
    precision_recall_auc,
    ranking_metrics_at_k,
    scores_to_labels,
    specificity_from_confusion,
    summarize_numeric_fields,
    threshold_sweep_metrics,
)
from app.evals.report import format_metrics_report, metrics_to_json, metrics_to_table
from app.evals.runner import (
    compare_retrieval_modes,
    run_classification_eval,
    run_retrieval_eval,
)
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


# --------------------------------------------------------------------------- #
# Expanded parameter matrix + extra metrics                                    #
# --------------------------------------------------------------------------- #


class TestParametrizedBinaryFixtures:
    """Table-driven accuracy / precision / recall / F1 checks."""

    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            # all correct
            ([1, 0, 1, 0], [1, 0, 1, 0], (1.0, 1.0, 1.0, 1.0)),
            # all wrong
            ([1, 0], [0, 1], (0.0, 0.0, 0.0, 0.0)),
            # high precision, low recall: TP=1 FP=0 FN=1 TN=1 → P=1 R=0.5 Acc=2/3 F1=2/3
            ([1, 1, 0], [1, 0, 0], (2.0 / 3.0, 1.0, 0.5, 2.0 / 3.0)),
            # low precision, high recall: TP=2 FP=1 FN=0 TN=0 → P=2/3 R=1 Acc=2/3 F1=0.8
            ([1, 1, 0], [1, 1, 1], (2.0 / 3.0, 2.0 / 3.0, 1.0, 0.8)),
        ],
        ids=["perfect", "inverted", "prec_over_rec", "rec_over_prec"],
    )
    def test_binary_metric_table(
        self,
        y_true: list[int],
        y_pred: list[int],
        expected: tuple[float, float, float, float],
    ) -> None:
        report = classification_metrics(y_true, y_pred, average="binary")
        acc, prec, rec, f1 = expected
        assert report.accuracy == pytest.approx(acc)
        assert report.precision == pytest.approx(prec)
        assert report.recall == pytest.approx(rec)
        assert report.f1 == pytest.approx(f1)

    @pytest.mark.parametrize("average", ["binary", "micro", "macro", "weighted"])
    def test_all_average_modes_emit_core_keys(self, average: str) -> None:
        report = classification_metrics(
            [0, 1, 2, 1], [0, 1, 1, 2], average=average  # type: ignore[arg-type]
        )
        payload = report.to_dict()
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= payload[key] <= 1.0 + 1e-9


class TestSpecificityBalancedAccuracyAndNdcg:
    def test_specificity_and_balanced_accuracy(self) -> None:
        # TP1 FP1 TN2 FN0 → sens=1, spec=2/3, bal=(1+2/3)/2 = 5/6
        counts = confusion_counts([1, 0, 0, 0], [1, 1, 0, 0], positive_label=1)
        assert specificity_from_confusion(counts) == pytest.approx(2.0 / 3.0)
        assert balanced_accuracy_from_confusion(counts) == pytest.approx(5.0 / 6.0)

    def test_ndcg_perfect_and_partial(self) -> None:
        gold = {10, 20}
        perfect = ndcg_at_k([10, 20, 30], gold, k=2)
        assert perfect == pytest.approx(1.0)
        # relevant only at rank 2
        partial = ndcg_at_k([99, 10], {10}, k=2)
        assert 0.0 < partial < 1.0
        mean = mean_ndcg_at_k([[10, 20], [99, 10]], [{10}, {10}], k=2)
        assert 0.0 < mean <= 1.0


class TestThresholdSweep:
    def test_sweep_returns_sorted_rows_with_core_metrics(self) -> None:
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.6, 0.1]
        rows = threshold_sweep_metrics(
            y_true, scores, thresholds=(0.3, 0.5, 0.8)
        )
        assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.8]
        for row in rows:
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "balanced_accuracy",
            ):
                assert key in row
                assert 0.0 <= row[key] <= 1.0 + 1e-9

    def test_classification_eval_includes_by_threshold(self) -> None:
        metrics = run_classification_eval(
            [1, 1, 0, 0],
            y_pred=[0, 0, 0, 0],
            scores=[0.9, 0.4, 0.6, 0.1],
            params=EvalParams(
                relevance_threshold=0.5,
                thresholds=(0.3, 0.5, 0.8),
                average="binary",
            ),
        )
        assert "by_threshold" in metrics
        assert len(metrics["by_threshold"]) == 3
        # At 0.5: pred [1,0,1,0] → Acc/P/R/F1 = 0.5
        mid = next(r for r in metrics["by_threshold"] if r["threshold"] == 0.5)
        assert mid["accuracy"] == pytest.approx(0.5)
        assert mid["precision"] == pytest.approx(0.5)
        assert mid["recall"] == pytest.approx(0.5)
        assert mid["f1"] == pytest.approx(0.5)


class TestExpandedEvalParams:
    def test_from_mapping_round_trip(self) -> None:
        raw = {
            "k": 5,
            "ks": [1, 5],
            "thresholds": [0.25, 0.75],
            "tags": ["agentic"],
            "min_mrr": 0.2,
            "shuffle_cases": True,
            "seed": 7,
            "unknown_doc_field": "ignored",
        }
        params = EvalParams.from_mapping(raw)
        assert params.k == 5
        assert params.ks == (1, 5)
        assert params.thresholds == (0.25, 0.75)
        assert params.tags == ("agentic",)
        assert params.min_mrr == 0.2
        assert params.seed == 7
        assert "unknown_doc_field" not in params.to_dict()

    def test_mrr_gate(self) -> None:
        params = EvalParams(min_mrr=0.9)
        ok, failures = params.passes_gates({"mrr": 0.5})
        assert ok is False
        assert any("mrr" in f for f in failures)

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            EvalParams(thresholds=(1.5,))


class TestExpandedDatasetFilters:
    def test_exclude_and_tag_any_and_tags_and(self) -> None:
        without_k8s = filter_cases(
            GOLDEN_RETRIEVAL_CASES, exclude_case_ids=("kubernetes",)
        )
        assert all(c.id != "kubernetes" for c in without_k8s)

        any_tags = filter_cases(
            GOLDEN_RETRIEVAL_CASES, tag_any=("devops", "career")
        )
        assert any_tags
        assert all(
            {"devops", "career"}.intersection(set(c.tags)) for c in any_tags
        )

        and_tags = filter_cases(
            GOLDEN_RETRIEVAL_CASES, tags=("agentic", "paraphrase")
        )
        assert and_tags
        assert all(
            "agentic" in c.tags and "paraphrase" in c.tags for c in and_tags
        )

    def test_shuffle_is_seeded(self) -> None:
        a = [c.id for c in filter_cases(GOLDEN_RETRIEVAL_CASES, shuffle=True, seed=1)]
        b = [c.id for c in filter_cases(GOLDEN_RETRIEVAL_CASES, shuffle=True, seed=1)]
        c = [c.id for c in filter_cases(GOLDEN_RETRIEVAL_CASES, shuffle=True, seed=2)]
        assert a == b
        # With enough cases, different seeds almost always reorder.
        assert a != c or len(a) < 2

    def test_golden_set_grew(self) -> None:
        assert len(GOLDEN_RETRIEVAL_CASES) >= 14
        assert len(GOLDEN_CLASSIFICATION_FIXTURES) >= 5


class TestRetrievalParamMatrix:
    """More retrieval harness configurations (parameters under test)."""

    @pytest.mark.parametrize("mode", ["hybrid", "keyword"])
    @pytest.mark.parametrize("k", [1, 3])
    def test_modes_and_k_emit_metrics(
        self, db: Session, products: list[Product], mode: str, k: int
    ) -> None:
        params = EvalParams(
            k=k,
            ks=(k,),
            retrieval_mode=mode,  # type: ignore[arg-type]
            split="train",
            include_per_case=False,
            include_ndcg=True,
            limit_cases=4,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["n_cases"] >= 1
        assert metrics["k"] == k
        assert "ndcg_at_k" in metrics["by_k"][str(k)]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= float(metrics[key]) <= 1.0 + 1e-9

    def test_tag_filter_and_exclude_via_params(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=2,
            ks=(2,),
            tags=("agentic",),
            exclude_case_ids=("agentic-category",),
            include_per_case=True,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["n_cases"] >= 1
        ids = {row["id"] for row in metrics["per_case"]}
        assert "agentic-category" not in ids

    def test_use_catalog_accuracy_flag(
        self, db: Session, products: list[Product]
    ) -> None:
        base = EvalParams(
            k=1,
            ks=(1,),
            case_ids=("langgraph-agents",),
            use_catalog_accuracy=True,
            include_per_case=False,
            include_ndcg=False,
        )
        with_catalog = run_retrieval_eval(db, params=base)
        without = run_retrieval_eval(
            db, params=base.with_updates(use_catalog_accuracy=False)
        )
        # Hit rate is identical; accuracy definition may differ.
        assert with_catalog["hit_at_k"] == without["hit_at_k"]
        assert without["accuracy_at_k"] == without["hit_at_k"]

    def test_shuffle_limit_seed(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=1,
            ks=(1,),
            shuffle_cases=True,
            seed=42,
            limit_cases=3,
            include_per_case=True,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert metrics["n_cases"] == 3
        assert len(metrics["per_case"]) == 3

    def test_map_and_ndcg_and_fbeta_bundle(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=3,
            ks=(1, 3),
            split="train",
            include_map=True,
            include_ndcg=True,
            include_per_case=False,
            f_betas=(0.5, 2.0),
            limit_cases=5,
        )
        metrics = run_retrieval_eval(db, params=params)
        assert "map_at_k" in metrics["by_k"]["3"]
        assert "ndcg_at_k" in metrics["by_k"]["3"]
        assert 0.0 <= metrics["by_k"]["3"]["map_at_k"] <= 1.0 + 1e-9
        bundle = metrics["classification"]["binary_bundle"]
        for key in ("accuracy", "precision", "recall", "f1", "f0_5", "f2"):
            assert key in bundle


class TestFbetaMapAndDelta:
    def test_fbeta_f1_matches_harmonic_mean(self) -> None:
        p, r = 0.5, 1.0
        assert fbeta_score(p, r, beta=1.0) == pytest.approx(2 * p * r / (p + r))
        # F2 weights recall → higher than F1 when R > P
        assert fbeta_score(p, r, beta=2.0) > fbeta_score(p, r, beta=1.0)
        # F0.5 weights precision → lower than F1 when R > P
        assert fbeta_score(p, r, beta=0.5) < fbeta_score(p, r, beta=1.0)

    def test_map_perfect_and_partial(self) -> None:
        assert average_precision_at_k([1, 2, 3], {1, 2}, k=2) == pytest.approx(1.0)
        # hit only at rank 2 → AP = (1/1)*(1/2) / 1 = 0.5 for single relevant
        assert average_precision_at_k([9, 1], {1}, k=2) == pytest.approx(0.5)
        mean = mean_average_precision_at_k(
            [[1, 2], [9, 1]], [{1}, {1}], k=2
        )
        assert 0.0 < mean <= 1.0

    def test_metrics_delta_subtracts_shared_keys(self) -> None:
        delta = metrics_delta(
            {"accuracy": 0.5, "precision": 0.4, "extra": "x"},
            {"accuracy": 0.8, "precision": 0.5, "recall": 0.9},
        )
        assert delta["accuracy"] == pytest.approx(0.3)
        assert delta["precision"] == pytest.approx(0.1)
        assert "recall" not in delta

    def test_best_threshold_by_f1(self) -> None:
        rows = threshold_sweep_metrics(
            [1, 1, 0, 0],
            [0.9, 0.4, 0.6, 0.1],
            thresholds=(0.3, 0.5, 0.8),
        )
        best = best_threshold_by_metric(rows, metric="f1")
        assert "threshold" in best
        assert best["f1"] == max(r["f1"] for r in rows)


class TestGoldenClassificationFixtures:
    """Regression fixtures: expected accuracy/precision/recall/F1 must hold."""

    @pytest.mark.parametrize(
        "fixture",
        list(GOLDEN_CLASSIFICATION_FIXTURES),
        ids=[f.id for f in GOLDEN_CLASSIFICATION_FIXTURES],
    )
    def test_fixture_matches_expected(self, fixture: Any) -> None:
        if fixture.scores:
            metrics = run_classification_eval(
                list(fixture.y_true),
                y_pred=[],
                scores=list(fixture.scores),
                params=EvalParams(
                    average="binary",
                    relevance_threshold=0.5,
                    thresholds=(0.3, 0.5, 0.8),
                    f_betas=(0.5, 2.0),
                ),
            )
            assert "best_threshold_by_f1" in metrics
        else:
            metrics = run_classification_eval(
                list(fixture.y_true),
                list(fixture.y_pred),
                params=EvalParams(average="binary", f_betas=(0.5, 2.0)),
            )
        for key, expected in fixture.expected.items():
            assert metrics[key] == pytest.approx(expected), (
                f"{fixture.id}.{key}: {metrics[key]} != {expected}"
            )
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics["bundle"]

    def test_bundle_matches_classification_metrics(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        report = classification_metrics(y_true, y_pred, average="binary")
        bundle = classification_metrics_bundle(y_true, y_pred)
        assert bundle["accuracy"] == pytest.approx(report.accuracy)
        assert bundle["precision"] == pytest.approx(report.precision)
        assert bundle["recall"] == pytest.approx(report.recall)
        assert bundle["f1"] == pytest.approx(report.f1)


class TestMulticlassGoldenFixtures:
    @pytest.mark.parametrize(
        "fixture",
        list(GOLDEN_MULTICLASS_FIXTURES),
        ids=[f["id"] for f in GOLDEN_MULTICLASS_FIXTURES],
    )
    def test_micro_macro_weighted_f1(self, fixture: dict[str, Any]) -> None:
        y_true = fixture["y_true"]
        y_pred = fixture["y_pred"]
        expected = fixture["expected"]
        micro = classification_metrics(y_true, y_pred, average="micro")
        macro = classification_metrics(y_true, y_pred, average="macro")
        weighted = classification_metrics(y_true, y_pred, average="weighted")
        assert micro.accuracy == pytest.approx(expected["accuracy"])
        assert micro.f1 == pytest.approx(expected["micro_f1"])
        assert macro.f1 == pytest.approx(expected["macro_f1"])
        assert weighted.f1 == pytest.approx(expected["weighted_f1"])
        # micro precision/recall track accuracy for single-label multi-class
        assert micro.precision == pytest.approx(expected["accuracy"])
        assert micro.recall == pytest.approx(expected["accuracy"])


class TestPrecisionRecallAucAndSummary:
    def test_pr_auc_non_negative(self) -> None:
        rows = threshold_sweep_metrics(
            [1, 1, 0, 0, 1, 0],
            [0.95, 0.7, 0.6, 0.2, 0.4, 0.1],
            thresholds=(0.2, 0.4, 0.6, 0.8),
        )
        auc = precision_recall_auc(rows)
        assert auc >= 0.0
        metrics = run_classification_eval(
            [1, 1, 0, 0],
            y_pred=[],
            scores=[0.9, 0.4, 0.6, 0.1],
            params=EvalParams(
                thresholds=(0.3, 0.5, 0.8),
                min_pr_auc=0.0,
            ),
        )
        assert "pr_auc" in metrics
        assert metrics["passed_gates"] is True

    def test_summarize_numeric_fields(self) -> None:
        summary = summarize_numeric_fields(
            [
                {"precision_at_k": 1.0, "recall_at_k": 0.5, "hit_at_k": 1.0},
                {"precision_at_k": 0.0, "recall_at_k": 0.0, "hit_at_k": 0.0},
            ]
        )
        assert summary["precision_at_k"]["mean"] == pytest.approx(0.5)
        assert summary["hit_at_k"]["min"] == pytest.approx(0.0)
        assert summary["hit_at_k"]["max"] == pytest.approx(1.0)


class TestModeCompareAndPerCaseSummary:
    def test_compare_hybrid_vs_keyword(
        self, db: Session, products: list[Product]
    ) -> None:
        params = EvalParams(
            k=3,
            ks=(3,),
            split="train",
            limit_cases=4,
            include_ndcg=True,
            include_map=True,
        )
        result = compare_retrieval_modes(
            db,
            modes=("hybrid", "keyword"),
            params=params,
            baseline="keyword",
        )
        assert result["task"] == "retrieval_mode_compare"
        assert set(result["modes"]) == {"hybrid", "keyword"}
        assert "hybrid" in result["delta_vs_baseline"]
        for mode in ("hybrid", "keyword"):
            block = result["modes"][mode]
            for key in ("accuracy", "precision", "recall", "f1", "hit_at_k"):
                assert key in block
                assert 0.0 <= float(block[key]) <= 1.0 + 1e-9

        report = format_metrics_report(result, title="Mode compare")
        assert "Modes" in report
        assert "Delta vs baseline" in report

    def test_per_case_summary_attached(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=2,
                ks=(2,),
                split="train",
                limit_cases=3,
                include_per_case=True,
                include_per_case_summary=True,
            ),
        )
        assert "per_case_summary" in metrics
        assert "precision_at_k" in metrics["per_case_summary"]
        assert metrics["per_case_summary"]["precision_at_k"]["n"] == 3.0

    def test_report_includes_threshold_section(self) -> None:
        metrics = run_classification_eval(
            [1, 0, 1, 0],
            y_pred=[],
            scores=[0.9, 0.2, 0.7, 0.1],
            params=EvalParams(thresholds=(0.3, 0.6)),
        )
        text = format_metrics_report(metrics, title="Class report")
        assert "Metrics by threshold" in text
        assert "F1=" in text
