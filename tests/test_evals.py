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

from app.evals.config import (
    AGENT_ALIGNED_EVAL_PARAMS,
    DEFAULT_EVAL_PARAMS,
    EvalParams,
    STRICT_EVAL_PARAMS,
)
from app.evals.datasets import (
    GOLDEN_CLASSIFICATION_FIXTURES,
    GOLDEN_GRADER_SCORE_FIXTURES,
    GOLDEN_MULTICLASS_FIXTURES,
    GOLDEN_RERANK_FIXTURES,
    GOLDEN_RETRIEVAL_CASES,
    RetrievalCase,
    filter_cases,
    load_label_fixture,
    load_label_fixtures,
    resolve_cases,
)
from app.evals.metrics import (
    average_precision_at_k,
    balanced_accuracy_from_confusion,
    beats_baseline,
    best_threshold_by_metric,
    blend_rerank_scores,
    bootstrap_metric_ci,
    classification_metrics,
    classification_metrics_bundle,
    confusion_counts,
    confusion_matrix_labels,
    fbeta_score,
    k_sweep_table,
    matthews_corrcoef,
    mcnemar_test,
    mean_average_precision_at_k,
    mean_ndcg_at_k,
    metrics_delta,
    multilabel_sets_to_binary_vectors,
    ndcg_at_k,
    precision_recall_auc,
    random_ranking_baseline,
    rank_by_scores,
    ranking_metrics_at_k,
    scores_to_labels,
    specificity_from_confusion,
    success_at_k,
    summarize_numeric_fields,
    threshold_sweep_metrics,
)
from app.evals.report import (
    format_metrics_report,
    metrics_to_csv,
    metrics_to_json,
    metrics_to_table,
)
from app.evals.runner import (
    compare_retrieval_modes,
    compare_thresholds,
    run_classification_eval,
    run_eval_suite,
    run_grader_threshold_eval,
    run_param_grid,
    run_rerank_eval,
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


class TestMatthewsBootstrapAndCsv:
    def test_mcc_perfect_and_chance(self) -> None:
        assert matthews_corrcoef([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
        # balanced errors → MCC = 0
        assert matthews_corrcoef([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_bootstrap_ci_bounds(self) -> None:
        ci = bootstrap_metric_ci(
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1],
            metric="f1",
            n_bootstrap=40,
            seed=1,
            confidence=0.9,
        )
        assert ci["low"] <= ci["mean"] <= ci["high"]
        assert ci["n_bootstrap"] == 40.0

    def test_classification_eval_emits_mcc_and_bootstrap(self) -> None:
        metrics = run_classification_eval(
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            params=EvalParams(n_bootstrap=30, seed=2, bootstrap_confidence=0.9),
        )
        assert "mcc" in metrics
        assert -1.0 <= metrics["mcc"] <= 1.0
        assert "bootstrap" in metrics
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics["bootstrap"]
            assert "low" in metrics["bootstrap"][key]

    def test_metrics_to_csv_has_header_and_core_rows(self) -> None:
        csv_text = metrics_to_csv(
            {
                "task": "classification",
                "accuracy": 0.75,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 2.0 / 3.0,
                "mcc": 0.5,
            }
        )
        assert csv_text.startswith("metric,value")
        assert "accuracy,0.750000" in csv_text
        assert "precision," in csv_text


class TestGraderThresholdFixtures:
    """Accuracy/precision/recall/F1 for score→label grader path."""

    @pytest.mark.parametrize(
        "fixture",
        list(GOLDEN_GRADER_SCORE_FIXTURES),
        ids=[f["id"] for f in GOLDEN_GRADER_SCORE_FIXTURES],
    )
    def test_grader_fixture(self, fixture: dict[str, Any]) -> None:
        metrics = run_grader_threshold_eval(
            fixture["items"],
            params=EvalParams(
                relevance_threshold=fixture["threshold"],
                thresholds=(0.25, 0.35, 0.5, 0.65),
                n_bootstrap=0,
            ),
        )
        for key, expected in fixture["expected"].items():
            assert metrics[key] == pytest.approx(expected), (
                f"{fixture['id']}.{key}: {metrics[key]} != {expected}"
            )
        assert "by_threshold" in metrics
        assert "best_threshold_by_f1" in metrics


class TestLeaveOneOutAndMetadataFilters:
    def test_leave_one_out_summary(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=2,
                ks=(2,),
                split="train",
                leave_one_out=True,
                include_per_case=False,
                limit_cases=5,
            ),
        )
        assert "leave_one_out" in metrics
        loo = metrics["leave_one_out"]
        assert loo["n_folds"] == metrics["n_cases"]
        assert "precision_at_k" in loo["summary"]
        assert loo["summary"]["precision_at_k"]["n"] == float(metrics["n_cases"])

    def test_skill_level_filter_param(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=3,
                ks=(3,),
                skill_levels=("advanced",),
                case_ids=("multi-agent", "kubernetes"),
                include_per_case=True,
                retrieval_mode="hybrid",
            ),
        )
        assert metrics["filters"]["skill_levels"] == ["advanced"]
        # All retrieved products should be advanced when filters stick.
        advanced_ids = {
            p.id for p in products if (p.skill_level or "").lower() == "advanced"
        }
        for row in metrics["per_case"]:
            for pid in row["retrieved_ids"]:
                assert pid in advanced_ids

    @pytest.mark.parametrize(
        "k,threshold",
        [(1, 0.35), (3, 0.5), (5, 0.65)],
        ids=["k1-t35", "k3-t50", "k5-t65"],
    )
    def test_param_matrix_k_and_grader_threshold(
        self, k: int, threshold: float
    ) -> None:
        """Joint parameter surface: ranking k + grader threshold both accepted."""
        ranking = ranking_metrics_at_k(
            [[1, 2, 3, 4, 5]],
            [{1, 2}],
            k=k,
        )
        assert ranking.k == k
        assert 0.0 <= ranking.precision_at_k <= 1.0

        scores = [0.9, 0.4, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        preds = scores_to_labels(scores, threshold=threshold)
        report = classification_metrics(labels, preds, average="binary")
        for value in (report.accuracy, report.precision, report.recall, report.f1):
            assert 0.0 <= value <= 1.0 + 1e-9


class TestSuccessAtKAndRerankBlend:
    def test_success_at_k_agent_gate(self) -> None:
        # Need 2 relevants in top-3 for success; q1 has 2, q2 has 1
        rate = success_at_k(
            [[1, 2, 9], [3, 8, 7]],
            [{1, 2, 4}, {3, 5}],
            k=3,
            min_relevant=2,
        )
        assert rate == pytest.approx(0.5)

    def test_blend_matches_agent_weights(self) -> None:
        # peak retrieval = 1.0; blend = 0.65*j + 0.35*(r/peak)
        blended = blend_rerank_scores([1.0, 0.0], [0.5, 1.0])
        assert blended[0] == pytest.approx(0.65 * 1.0 + 0.35 * 0.5)
        assert blended[1] == pytest.approx(0.65 * 0.0 + 0.35 * 1.0)

    def test_rank_by_scores_descending(self) -> None:
        assert rank_by_scores([10, 20, 30], [0.1, 0.9, 0.5]) == [20, 30, 10]

    @pytest.mark.parametrize(
        "fixture",
        list(GOLDEN_RERANK_FIXTURES),
        ids=[f["id"] for f in GOLDEN_RERANK_FIXTURES],
    )
    def test_rerank_golden_fixtures(self, fixture: dict[str, Any]) -> None:
        metrics = run_rerank_eval(
            fixture["candidates"],
            params=EvalParams(
                k=fixture["k"],
                judge_weight=0.65,
                retrieval_weight=0.35,
                relevance_threshold=0.35,
                min_relevant=1,
            ),
        )
        assert metrics["task"] == "rerank"
        for key, expected in fixture["expected_blend"].items():
            assert metrics["orderings"]["blend"][key] == pytest.approx(expected), (
                f"{fixture['id']}.blend.{key}"
            )
        if "expected_retrieval" in fixture:
            for key, expected in fixture["expected_retrieval"].items():
                assert metrics["orderings"]["retrieval"][key] == pytest.approx(
                    expected
                ), f"{fixture['id']}.retrieval.{key}"
        # Top-level accuracy/precision/recall/f1 from thresholded blend scores
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= float(metrics[key]) <= 1.0 + 1e-9

    def test_retrieval_eval_exposes_success_at_k(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=3,
                ks=(3,),
                split="train",
                min_relevant=1,
                include_per_case=False,
                limit_cases=4,
            ),
        )
        assert "success_at_k" in metrics
        assert 0.0 <= metrics["success_at_k"] <= 1.0

    def test_presets_exist_and_are_agent_aligned(self) -> None:
        assert AGENT_ALIGNED_EVAL_PARAMS.relevance_threshold == 0.35
        assert AGENT_ALIGNED_EVAL_PARAMS.judge_weight == 0.65
        assert AGENT_ALIGNED_EVAL_PARAMS.retrieval_weight == 0.35
        assert AGENT_ALIGNED_EVAL_PARAMS.min_relevant == 3
        assert DEFAULT_EVAL_PARAMS.k == 3
        assert STRICT_EVAL_PARAMS.min_f1 == 0.3
        assert STRICT_EVAL_PARAMS.min_success_at_k == 0.2


class TestMcNemarKSweepFixturesAndSuite:
    def test_mcnemar_identical_predictors(self) -> None:
        result = mcnemar_test([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0])
        assert result["n_discordant"] == 0.0
        assert result["statistic"] == 0.0

    def test_mcnemar_detects_discordance(self) -> None:
        # A wrong on first, B wrong on second
        result = mcnemar_test(
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        )
        assert result["b"] == 1.0
        assert result["c"] == 1.0
        assert result["n_discordant"] == 2.0

    def test_compare_thresholds_035_vs_05(self) -> None:
        y_true = [1, 1, 0, 0]
        scores = [0.4, 0.3, 0.36, 0.1]
        # t=0.35 → pred [1,0,1,0]; t=0.5 → pred [0,0,0,0]
        result = compare_thresholds(
            y_true, scores, threshold_a=0.35, threshold_b=0.5
        )
        assert result["task"] == "threshold_compare"
        assert result["a"]["accuracy"] == pytest.approx(0.5)
        assert result["b"]["accuracy"] == pytest.approx(0.5)
        assert result["b"]["recall"] == pytest.approx(0.0)
        assert "mcnemar" in result
        assert "delta_b_minus_a" in result

    def test_k_sweep_table_keys(self) -> None:
        table = k_sweep_table(
            [[1, 2, 3, 4]],
            [{1, 3}],
            ks=(1, 2, 3),
            min_relevant=1,
        )
        assert set(table) == {"1", "2", "3"}
        for block in table.values():
            for key in ("accuracy", "precision", "recall", "f1", "success_at_k"):
                assert key in block
                assert 0.0 <= block[key] <= 1.0 + 1e-9
        # At k=1 if top is relevant → P=1
        assert table["1"]["precision"] in (0.0, 1.0)

    def test_load_label_fixture_and_eval(self) -> None:
        raw = {
            "id": "json-style",
            "y_true": [1, 0, 1, 0],
            "y_pred": [1, 0, 0, 0],
            "expected": {
                "accuracy": 0.75,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2.0 / 3.0,
            },
        }
        fixture = load_label_fixture(raw)
        metrics = run_classification_eval(
            fixture["y_true"], fixture["y_pred"], params=EvalParams(average="binary")
        )
        for key, expected in fixture["expected"].items():
            assert metrics[key] == pytest.approx(expected)
        batch = load_label_fixtures([raw, {
            "id": "scores",
            "y_true": [1, 0],
            "scores": [0.9, 0.1],
            "threshold": 0.5,
            "expected": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        }])
        assert len(batch) == 2

    def test_retrieval_includes_k_sweep(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=3,
                ks=(1, 3),
                split="train",
                limit_cases=3,
                include_per_case=False,
                min_relevant=1,
            ),
        )
        assert "k_sweep" in metrics
        assert "1" in metrics["k_sweep"] and "3" in metrics["k_sweep"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics["k_sweep"]["3"]
        text = format_metrics_report(metrics, title="Sweep")
        assert "k-sweep" in text

    def test_run_eval_suite_matches_fixtures(
        self, db: Session, products: list[Product]
    ) -> None:
        suite = run_eval_suite(
            db,
            params=EvalParams(
                k=3,
                ks=(3,),
                split="train",
                limit_cases=4,
                include_per_case=False,
                min_hit_rate=0.0,
                min_relevant=1,
            ),
            include_mode_compare=False,
        )
        assert suite["task"] == "eval_suite"
        assert suite["retrieval"]["n_cases"] >= 1
        assert suite["classification_fixtures"]["n_match"] == suite[
            "classification_fixtures"
        ]["n"]
        assert suite["grader_fixtures"]["n_match"] == suite["grader_fixtures"]["n"]
        assert suite["rerank_fixtures"]["n"] >= 1
        assert suite["passed_gates"] is True


class TestRandomBaselineParamGridAndConfusion:
    def test_random_baseline_in_unit_range(self) -> None:
        baseline = random_ranking_baseline(
            [{1, 2}, {3}],
            catalog_ids=[1, 2, 3, 4, 5, 6],
            k=2,
            n_trials=20,
            seed=0,
            min_relevant=1,
        )
        for key in ("accuracy", "precision", "recall", "f1", "hit_at_k", "mrr"):
            assert 0.0 <= baseline[key] <= 1.0 + 1e-9
        assert baseline["n_trials"] == 20.0

    def test_beats_baseline_helper(self) -> None:
        result = beats_baseline(
            {"precision": 0.8, "recall": 0.5, "f1": 0.6, "hit_at_k": 1.0, "mrr": 0.7},
            {"precision": 0.2, "recall": 0.2, "f1": 0.2, "hit_at_k": 0.3, "mrr": 0.2},
        )
        assert result["all_beat"] is True
        assert result["n_wins"] == 5

    def test_retrieval_vs_random(
        self, db: Session, products: list[Product]
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=3,
                ks=(3,),
                split="train",
                limit_cases=5,
                include_per_case=False,
                min_relevant=1,
                random_baseline_trials=15,
                seed=1,
                min_hit_rate=0.0,
            ),
        )
        assert "random_baseline" in metrics
        assert "vs_random" in metrics
        assert metrics["vs_random"]["n_checked"] >= 1
        text = format_metrics_report(metrics, title="Random")
        assert "Vs random baseline" in text

    def test_param_grid_k_values(
        self, db: Session, products: list[Product]
    ) -> None:
        grid = run_param_grid(
            db,
            grid=[{"k": 1, "ks": (1,)}, {"k": 3, "ks": (3,)}],
            base_params=EvalParams(
                split="train",
                limit_cases=3,
                include_per_case=False,
                min_relevant=1,
                min_hit_rate=0.0,
            ),
        )
        assert grid["task"] == "param_grid"
        assert grid["n"] == 2
        assert grid["best_by_f1"] is not None
        for row in grid["rows"]:
            for key in ("accuracy", "precision", "recall", "f1"):
                assert 0.0 <= float(row[key]) <= 1.0 + 1e-9

    def test_confusion_matrix_multiclass(self) -> None:
        matrix = confusion_matrix_labels(
            ["A", "A", "B", "C"],
            ["A", "B", "B", "C"],
            labels=["A", "B", "C"],
        )
        assert matrix["labels"] == ["A", "B", "C"]
        assert matrix["matrix"]["A"]["A"] == 1
        assert matrix["matrix"]["A"]["B"] == 1
        assert matrix["matrix"]["B"]["B"] == 1
        assert matrix["matrix"]["C"]["C"] == 1

    def test_classification_includes_confusion_matrix(self) -> None:
        metrics = run_classification_eval(
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            params=EvalParams(average="binary"),
        )
        assert "confusion_matrix" in metrics
        assert "matrix" in metrics["confusion_matrix"]

    @pytest.mark.parametrize(
        "mode,k",
        [("hybrid", 1), ("hybrid", 3), ("keyword", 2)],
    )
    def test_grid_like_param_surface(
        self, db: Session, products: list[Product], mode: str, k: int
    ) -> None:
        metrics = run_retrieval_eval(
            db,
            params=EvalParams(
                k=k,
                ks=(k,),
                retrieval_mode=mode,  # type: ignore[arg-type]
                split="train",
                limit_cases=3,
                include_per_case=False,
                min_relevant=1,
                min_hit_rate=0.0,
            ),
        )
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert 0.0 <= float(metrics[key]) <= 1.0 + 1e-9
