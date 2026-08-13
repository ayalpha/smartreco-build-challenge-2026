"""Unit tests for the recommendation evaluation harness."""

import json
import subprocess
import sys

import pytest

from app.evaluation import evaluate, select_predictions


def test_perfect_multilabel_metrics() -> None:
    result = evaluate([["a", "b"], ["c"]], [["a", "b"], ["c"]])
    assert result.accuracy == 1.0
    assert result.precision_micro == result.recall_micro == result.f1_micro == 1.0
    assert result.precision_macro == result.recall_macro == result.f1_macro == 1.0


def test_micro_and_macro_metrics_with_false_positive_and_negative() -> None:
    result = evaluate([["a"], ["a", "b"]], [["a", "b"], ["b"]])
    assert result.accuracy == 0.0
    assert result.precision_micro == pytest.approx(2 / 3)
    assert result.recall_micro == pytest.approx(2 / 3)
    assert result.f1_micro == pytest.approx(2 / 3)
    assert result.precision_macro == pytest.approx(0.75)
    assert result.recall_macro == pytest.approx(0.75)
    assert result.f1_macro == pytest.approx(2 / 3)


def test_top_k_and_threshold_are_applied_in_score_order() -> None:
    assert select_predictions({"low": .2, "best": .9, "mid": .7}, k=2, threshold=.5) == {"best", "mid"}


def test_empty_input_has_defined_zero_metrics() -> None:
    result = evaluate([], [], labels=["a"])
    assert result.examples == 0
    assert result.accuracy == result.f1_micro == result.f1_macro == 0.0


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"threshold": -0.1}, {"threshold": 1.1}])
def test_invalid_selection_parameters_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        select_predictions([], **kwargs)


def test_length_mismatch_and_unknown_configured_labels_raise() -> None:
    with pytest.raises(ValueError, match="same number"):
        evaluate([["a"]], [])
    with pytest.raises(ValueError, match="configured label set"):
        evaluate([["a"]], [["a"]], labels=["b"])


def test_cli_reads_fixture_and_writes_report(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "report.json"
    fixture.write_text(json.dumps({"examples": [{"expected": ["a"], "predicted": {"a": .8, "b": .2}}]}))
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_recommendations.py", "--fixtures", str(fixture),
         "--threshold", "0.5", "--labels", "a", "b", "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(completed.stdout)
    assert report["accuracy"] == 1.0
    assert json.loads(output.read_text())["examples"] == 1
