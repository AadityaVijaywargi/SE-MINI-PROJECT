"""Unified evaluation (SM-52 to SM-55).

Implements REQ-EV-1 to REQ-EV-4: one function tests every model on the held-out test set,
whatever its paradigm.
"""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PARADIGMS = ("ML", "DL", "QML")
ROC_POINTS = 100


def downsampled_roc(y_true, scores, points=ROC_POINTS):
    """ROC curve as ``{fpr, tpr}`` lists of at most ``points`` points (REQ-EV-3, PERF-5).

    The first and last points are always kept, so the curve still runs from (0, 0) to
    (1, 1).
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    if len(fpr) > points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, points).round().astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {
        "fpr": [round(float(value), 4) for value in fpr],
        "tpr": [round(float(value), 4) for value in tpr],
    }


def positive_scores(model, features):
    """Scores for the positive class of a binary model, or None if it has neither
    ``predict_proba`` nor ``decision_function``."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return None


def evaluate(model, split, trainer, train_time):
    """Test a fitted model on the shared held-out test set (REQ-EV-1, REQ-EV-2).

    The test rows are never seen during training. Binary targets use binary averaging;
    multi-class targets use weighted averaging; zero division counts as 0. ROC-AUC is
    computed for binary targets only, with the ROC curve (REQ-EV-3). ``test_time`` is how
    long predicting took.
    """
    start = time.perf_counter()
    predictions = np.asarray(model.predict(split.x_test))
    binary = split.n_classes == 2
    average = "binary" if binary else "weighted"

    roc_auc, curve = None, None
    if binary:
        scores = positive_scores(model, split.x_test)
        if scores is not None:
            roc_auc = float(roc_auc_score(split.y_test, scores))
            curve = downsampled_roc(split.y_test, scores)
    test_time = time.perf_counter() - start

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
        "roc_auc": roc_auc,
        "roc_curve": curve,
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


def summarise(results):
    """Best model overall and per paradigm, ranked by accuracy then F1 (REQ-EV-4)."""
    done = [result for result in results if result.get("status") == "done"]

    def best(candidates):
        if not candidates:
            return None
        return max(candidates, key=lambda result: (result["accuracy"], result["f1"]))["model"]

    summary = {"best_overall": best(done)}
    for paradigm in PARADIGMS:
        summary[f"best_{paradigm.lower()}"] = best(
            [result for result in done if result["paradigm"] == paradigm]
        )
    return summary
