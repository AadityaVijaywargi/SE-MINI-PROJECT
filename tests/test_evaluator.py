"""Unified evaluator (SM-52, SM-53, SM-93)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluator import evaluate, failed_result, summarise
from core.preprocessing import Split

METRIC_KEYS = {
    "key",
    "model",
    "paradigm",
    "status",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "train_time",
    "confusion_matrix",
}


class FixedModel:
    """Returns preset predictions and probabilities."""

    def __init__(self, predictions, proba=None):
        self.predictions = np.asarray(predictions)
        self.proba = proba

    def predict(self, _features):
        """The preset class predictions."""
        return self.predictions

    def predict_proba(self, _features):
        """The preset class probabilities."""
        return np.asarray(self.proba)


FAKE_TRAINER = SimpleNamespace(key="fake", name="Fake", paradigm="ML")


def make_split(y_test, classes):
    """A Split whose test labels are ``y_test``; the features are never read."""
    features = np.zeros((len(y_test), 1))
    return Split(features, features, np.asarray(y_test), np.asarray(y_test), classes, ["f"])


def test_binary_metrics():
    split = make_split([0, 0, 1, 1], ["no", "yes"])
    proba = [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]
    result = evaluate(FixedModel([0, 1, 1, 1], proba), split, FAKE_TRAINER, 0.5)
    assert set(result) == METRIC_KEYS
    assert result["accuracy"] == 0.75
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_multiclass_uses_weighted_average_and_no_roc():
    split = make_split([0, 1, 2, 2], ["a", "b", "c"])
    result = evaluate(FixedModel([0, 1, 2, 1]), split, FAKE_TRAINER, 0.1)
    assert result["roc_auc"] is None
    assert result["accuracy"] == 0.75
    expected_f1 = (1 * 1 + 1 * (2 / 3) + 2 * (2 / 3)) / 4
    assert result["f1"] == pytest.approx(expected_f1)


def test_zero_division_counts_as_zero():
    split = make_split([0, 0, 1, 1], ["no", "yes"])
    result = evaluate(FixedModel([0, 0, 0, 0], [[1, 0]] * 4), split, FAKE_TRAINER, 0.1)
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_confusion_matrix_includes_classes_never_predicted():
    split = make_split([0, 1, 2], ["a", "b", "c"])
    result = evaluate(FixedModel([0, 0, 0]), split, FAKE_TRAINER, 0.1)
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_summary_best_overall_and_per_paradigm():
    results = [
        {"model": "A", "paradigm": "ML", "status": "done", "accuracy": 0.8, "f1": 0.7},
        {"model": "B", "paradigm": "ML", "status": "done", "accuracy": 0.8, "f1": 0.9},
        {"model": "C", "paradigm": "DL", "status": "done", "accuracy": 0.6, "f1": 0.6},
        failed_result(FAKE_TRAINER, "boom"),
    ]
    assert summarise(results) == {
        "best_overall": "B",
        "best_ml": "B",
        "best_dl": "C",
        "best_qml": None,
    }


def test_summary_with_nothing_done():
    assert summarise([failed_result(FAKE_TRAINER, "boom")])["best_overall"] is None
