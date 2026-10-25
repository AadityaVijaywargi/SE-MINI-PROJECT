"""Unified evaluator (SM-52 to SM-55, SM-93)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.evaluator import (
    ROC_POINTS,
    best_f1_threshold,
    downsampled_roc,
    evaluate,
    failed_result,
    summarise,
)
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
    "roc_curve",
    "pr_auc",
    "pr_curve",
    "threshold",
    "threshold_source",
    "train_time",
    "test_time",
    "test_rows",
    "confusion_matrix",
    "details",
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
    assert result["roc_curve"] is None
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


def test_records_test_time_and_test_rows():
    split = make_split([0, 1, 1], ["no", "yes"])
    result = evaluate(FixedModel([0, 1, 1], [[1, 0], [0, 1], [0, 1]]), split, FAKE_TRAINER, 0.1)
    assert result["test_rows"] == 3
    assert result["test_time"] >= 0


def test_summary_best_overall_and_per_paradigm():
    results = [
        {"model": "A", "paradigm": "ML", "status": "done", "accuracy": 0.8, "f1": 0.7},
        {"model": "B", "paradigm": "ML", "status": "done", "accuracy": 0.8, "f1": 0.9},
        {"model": "C", "paradigm": "DL", "status": "done", "accuracy": 0.6, "f1": 0.6},
        failed_result(FAKE_TRAINER, "boom"),
    ]
    assert summarise(results) == {
        "best_overall": "B",
        "ranked_by": ["f1", "accuracy"],
        "best_ml": "B",
        "best_dl": "C",
        "best_qml": None,
    }


def test_summary_with_nothing_done():
    assert summarise([failed_result(FAKE_TRAINER, "boom")])["best_overall"] is None


def test_binary_result_has_roc_curve_from_origin_to_corner():
    split = make_split([0, 0, 1, 1], ["no", "yes"])
    proba = [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]
    curve = evaluate(FixedModel([0, 1, 1, 1], proba), split, FAKE_TRAINER, 0.5)["roc_curve"]
    assert curve["fpr"][0] == 0 and curve["tpr"][0] == 0
    assert curve["fpr"][-1] == 1 and curve["tpr"][-1] == 1


def test_roc_curve_is_downsampled_to_about_100_points():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 5000)
    scores = labels + rng.normal(0, 1, 5000)
    curve = downsampled_roc(labels, scores)
    assert 2 < len(curve["fpr"]) <= ROC_POINTS
    assert len(curve["fpr"]) == len(curve["tpr"])
    assert (curve["fpr"][0], curve["tpr"][0]) == (0.0, 0.0)
    assert (curve["fpr"][-1], curve["tpr"][-1]) == (1.0, 1.0)
    assert curve["fpr"] == sorted(curve["fpr"])


def test_trainer_details_are_reported():
    trainer = SimpleNamespace(key="k", name="K", paradigm="DL", details={"Epochs run": 7})
    split = make_split([0, 1], ["no", "yes"])
    result = evaluate(FixedModel([0, 1], [[1, 0], [0, 1]]), split, trainer, 0.1)
    assert result["details"] == {"Epochs run": 7}


class ScoreModel:
    """Positive-class probability equal to the single feature, for threshold tests."""

    def predict(self, features):
        """Class 1 at the usual 0.5 threshold."""
        return (np.asarray(features)[:, 0] >= 0.5).astype(int)

    def predict_proba(self, features):
        """``[1 - x, x]`` per row."""
        positive = np.asarray(features)[:, 0]
        return np.column_stack([1 - positive, positive])


def rare_split():
    """Positives score about 0.2 and negatives 0.05: a 0.5 threshold finds none."""
    test_scores = np.array([0.05] * 18 + [0.2, 0.22])
    val_scores = np.array([0.04] * 18 + [0.21, 0.19])
    labels = np.array([0] * 18 + [1, 1])
    return Split(
        val_scores[:, None],
        test_scores[:, None],
        labels,
        labels,
        ["legit", "fraud"],
        ["s"],
        x_val=val_scores[:, None],
        y_val=labels,
    )


def test_threshold_is_tuned_on_validation_rows():
    result = evaluate(ScoreModel(), rare_split(), FAKE_TRAINER, 0.1)
    assert result["threshold_source"] == "tuned on validation rows"
    assert result["threshold"] == pytest.approx(0.19)
    assert result["recall"] == 1.0 and result["precision"] == 1.0
    assert result["confusion_matrix"] == [[18, 0], [0, 2]]


def test_fixed_threshold_reproduces_the_all_negative_failure():
    # The bug this guards against: 90% accuracy while every positive row is missed.
    result = evaluate(ScoreModel(), rare_split(), FAKE_TRAINER, 0.1, threshold=0.5)
    assert result["threshold_source"] == "fixed"
    assert result["accuracy"] == 0.9
    assert result["recall"] == 0.0
    assert result["pr_auc"] == 1.0  # the ranking was perfect, only the threshold was wrong


def test_threshold_defaults_to_half_without_validation_rows():
    split = make_split([0, 0, 1, 1], ["no", "yes"])
    proba = [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]
    result = evaluate(FixedModel([0, 1, 1, 1], proba), split, FAKE_TRAINER, 0.5)
    assert (result["threshold"], result["threshold_source"]) == (0.5, "default")


def test_zero_one_probabilities_never_get_threshold_zero():
    labels = np.array([0, 0, 0, 1])
    assert best_f1_threshold(labels, np.array([0.0, 0.0, 1.0, 0.0])) == 0.5


def test_binary_result_has_pr_curve_and_multiclass_has_none():
    curve = evaluate(ScoreModel(), rare_split(), FAKE_TRAINER, 0.1)["pr_curve"]
    assert len(curve["recall"]) == len(curve["precision"]) <= ROC_POINTS
    assert curve["recall"] == sorted(curve["recall"])
    split = make_split([0, 1, 2], ["a", "b", "c"])
    multiclass = evaluate(FixedModel([0, 1, 2]), split, FAKE_TRAINER, 0.1)
    assert multiclass["pr_auc"] is None and multiclass["pr_curve"] is None
    assert multiclass["threshold"] is None


def test_binary_summary_ranks_by_pr_auc_not_accuracy():
    results = [
        {
            "model": "Majority",
            "paradigm": "ML",
            "status": "done",
            "accuracy": 0.99,
            "f1": 0.0,
            "pr_auc": 0.02,
        },
        {
            "model": "Finder",
            "paradigm": "DL",
            "status": "done",
            "accuracy": 0.95,
            "f1": 0.4,
            "pr_auc": 0.35,
        },
    ]
    summary = summarise(results)
    assert summary["best_overall"] == "Finder"
    assert summary["ranked_by"] == ["pr_auc", "f1"]
