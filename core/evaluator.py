"""Unified evaluation (SM-52 to SM-55).

Implements REQ-EV-1 to REQ-EV-4: one function tests every model on the held-out test set,
whatever its paradigm.

Binary targets are scored at a decision threshold picked on the validation rows (the one
with the best F1), not at a fixed 0.5: on a dataset with 1% fraud, a model's fraud
probability rarely reaches 0.5, so a 0.5 threshold flags nothing and reports 99% accuracy
with 0% recall. Binary results also carry the precision-recall curve and PR-AUC, which,
unlike accuracy and ROC-AUC, show how well a model finds a rare class.
"""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config

PARADIGMS = ("ML", "DL", "QML")
ROC_POINTS = 100
THRESHOLD_AUTO = "auto"
DEFAULT_THRESHOLD = 0.5
# How the best model is chosen (REQ-EV-4). Binary targets: PR-AUC, which does not depend
# on the threshold and is not inflated by a large majority class, then F1. Multi-class
# targets: weighted F1, then accuracy.
BINARY_RANKING = ("pr_auc", "f1")
MULTICLASS_RANKING = ("f1", "accuracy")


def keep_points(length, points):
    """Indices of at most ``points`` evenly spaced entries, always with the first and last."""
    if length <= points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, points).round().astype(int))


def downsampled_roc(y_true, scores, points=ROC_POINTS):
    """ROC curve as ``{fpr, tpr}`` lists of at most ``points`` points (REQ-EV-3, PERF-5).

    The first and last points are always kept, so the curve still runs from (0, 0) to
    (1, 1).
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    keep = keep_points(len(fpr), points)
    return {
        "fpr": [round(float(value), 4) for value in fpr[keep]],
        "tpr": [round(float(value), 4) for value in tpr[keep]],
    }


def downsampled_pr(y_true, scores, points=ROC_POINTS):
    """Precision-recall curve as ``{recall, precision}`` lists of at most ``points`` points,
    ordered by rising recall."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    precision, recall = precision[::-1], recall[::-1]
    keep = keep_points(len(recall), points)
    return {
        "recall": [round(float(value), 4) for value in recall[keep]],
        "precision": [round(float(value), 4) for value in precision[keep]],
    }


def positive_scores(model, features):
    """Scores for the positive class of a binary model, or None if it has neither
    ``predict_proba`` nor ``decision_function``."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return None


def best_f1_threshold(y_true, scores, low=config.MIN_THRESHOLD, high=config.MAX_THRESHOLD):
    """The probability threshold from ``low`` to ``high`` with the highest F1 on ``y_true``.

    The range keeps out degenerate choices: a model whose probabilities are only 0 or 1
    (a fully grown decision tree) would otherwise get threshold 0 and flag every row.
    Returns DEFAULT_THRESHOLD when no score lies in the range.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    allowed = (thresholds >= low) & (thresholds <= high)
    if not allowed.any():
        return DEFAULT_THRESHOLD
    f1 = np.where(allowed, f1, -1.0)
    return float(thresholds[int(np.argmax(f1))])


def decision_threshold(model, split, threshold=THRESHOLD_AUTO):
    """``(threshold, source)`` for a binary model with ``predict_proba``.

    ``threshold`` is a fixed probability, or ``"auto"`` to pick the best-F1 threshold on
    the validation rows. Auto falls back to 0.5 when there are no validation rows or they
    hold only one class. The test rows are never used to choose it.
    """
    if threshold != THRESHOLD_AUTO:
        return float(threshold), "fixed"
    y_val = split.y_val
    if y_val is None or len(np.unique(y_val)) < 2:
        return DEFAULT_THRESHOLD, "default"
    scores = np.asarray(model.predict_proba(split.x_val))[:, 1]
    return best_f1_threshold(y_val, scores), "tuned on validation rows"


def binary_scores(model, features, threshold):
    """Predictions and positive-class scores for a binary target.

    With a ``threshold`` (from ``decision_threshold``) the class-1 probability is
    thresholded; without one, for a model lacking ``predict_proba``, the model's own
    ``predict`` is kept.
    """
    if threshold is None:
        return np.asarray(model.predict(features)), positive_scores(model, features)
    scores = np.asarray(model.predict_proba(features))[:, 1]
    return (scores >= threshold).astype(int), scores


def curve_metrics(y_true, scores):
    """ROC-AUC, ROC curve, PR-AUC and precision-recall curve of binary ``scores``; all
    None when there are no scores (multi-class targets)."""
    if scores is None:
        return {"roc_auc": None, "roc_curve": None, "pr_auc": None, "pr_curve": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "roc_curve": downsampled_roc(y_true, scores),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "pr_curve": downsampled_pr(y_true, scores),
    }


def evaluate(model, split, trainer, train_time, threshold=THRESHOLD_AUTO):
    """Test a fitted model on the shared held-out test set (REQ-EV-1, REQ-EV-2).

    The test rows are never seen during training. Binary targets use binary averaging
    for the positive class (label 1) at the decision threshold from ``decision_threshold``
    and get ROC and precision-recall curves (REQ-EV-3); multi-class targets use weighted
    averaging. Zero division counts as 0. ``test_time`` is how long predicting the test
    rows took.
    """
    binary = split.n_classes == 2
    curves, used, source = curve_metrics(split.y_test, None), None, None
    if binary and hasattr(model, "predict_proba"):
        used, source = decision_threshold(model, split, threshold)
    start = time.perf_counter()
    if binary:
        predictions, scores = binary_scores(model, split.x_test, used)
        curves = curve_metrics(split.y_test, scores)
    else:
        predictions = np.asarray(model.predict(split.x_test))
    test_time = time.perf_counter() - start
    average = "binary" if binary else "weighted"

    matrix = confusion_matrix(split.y_test, predictions, labels=list(range(split.n_classes)))
    return {
        "key": trainer.key,
        "model": trainer.name,
        "paradigm": trainer.paradigm,
        "status": "done",
        "accuracy": float(accuracy_score(split.y_test, predictions)),
        "precision": float(
            precision_score(split.y_test, predictions, average=average, zero_division=0)
        ),
        "recall": float(recall_score(split.y_test, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(split.y_test, predictions, average=average, zero_division=0)),
        **curves,
        "threshold": None if used is None else round(used, 4),
        "threshold_source": source,
        "train_time": float(train_time),
        "test_time": float(test_time),
        "test_rows": int(len(split.y_test)),
        "confusion_matrix": matrix.tolist(),
        "details": dict(getattr(trainer, "details", None) or {}),
    }


def failed_result(trainer, reason):
    """Result entry for a model that raised during training (REQ-TR-8)."""
    return {
        "key": trainer.key,
        "model": trainer.name,
        "paradigm": trainer.paradigm,
        "status": "failed",
        "error": reason,
    }


def ranking_keys(done):
    """The metrics the best model is chosen by: BINARY_RANKING when any result has a
    PR-AUC, MULTICLASS_RANKING otherwise."""
    if any(result.get("pr_auc") is not None for result in done):
        return BINARY_RANKING
    return MULTICLASS_RANKING


def summarise(results):
    """Best model overall and per paradigm, by the metrics in ``ranking_keys`` (REQ-EV-4).

    ``ranked_by`` names those metrics so the dashboard can say how the winner was chosen.
    A missing score ranks below every real one.
    """
    done = [result for result in results if result.get("status") == "done"]
    keys = ranking_keys(done)

    def rank(result):
        values = (result.get(key) for key in keys)
        return tuple(-1.0 if value is None else value for value in values)

    def best(candidates):
        if not candidates:
            return None
        return max(candidates, key=rank)["model"]

    summary = {"best_overall": best(done), "ranked_by": list(keys)}
    for paradigm in PARADIGMS:
        summary[f"best_{paradigm.lower()}"] = best(
            [result for result in done if result["paradigm"] == paradigm]
        )
    return summary
