"""Unified evaluation (SM-52, SM-53).

Implements REQ-EV-1 and REQ-EV-2: one function tests every model on the held-out test set,
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
)

PARADIGMS = ("ML", "DL", "QML")


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
    computed for binary targets only. ``test_time`` is how long predicting took.
    """
    start = time.perf_counter()
    predictions = model.predict(split.x_test)
    binary = split.n_classes == 2
    average = "binary" if binary else "weighted"

    roc_auc = None
    if binary:
        scores = positive_scores(model, split.x_test)
        if scores is not None:
            roc_auc = float(roc_auc_score(split.y_test, scores))
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
        "train_time": float(train_time),
        "test_time": float(test_time),
        "test_rows": int(len(split.y_test)),
        "confusion_matrix": matrix.tolist(),
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
