"""Target suggestion and task analysis (SM-21, SM-22, SM-24, SM-26).

Implements REQ-TG-1, REQ-TG-2, REQ-TG-4 and REQ-TG-6.
"""

import re

from core.profiling import is_numeric

MIN_CLASSES = 2
MAX_CLASSES = 20
IMBALANCE_WARNING_RATIO = 3.0
TARGET_KEYWORDS = frozenset(
    {
        "target",
        "label",
        "class",
        "churn",
        "exited",
        "diagnosis",
        "outcome",
        "result",
        "y",
        "default",
        "fraud",
    }
)


def name_tokens(name):
    """Split a column name into lower-case words: 'is_Fraud' and 'isFraud' -> {'is', 'fraud'}."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token}


def is_candidate(series):
    """A column is a target candidate if it has 2 to 20 distinct non-null values (REQ-TG-1)."""
    return MIN_CLASSES <= series.nunique(dropna=True) <= MAX_CLASSES


def score_candidate(series, is_last):
    """Score a candidate target column (REQ-TG-2). Returns ``(score, reasons)``."""
    score, reasons = 0, []
    if series.nunique(dropna=True) == 2:
        score += 50
        reasons.append("two classes")
    if name_tokens(series.name) & TARGET_KEYWORDS:
        score += 40
        reasons.append("name looks like a target")
    if is_last:
        score += 20
        reasons.append("last column")
    if not is_numeric(series):
        score += 10
        reasons.append("text labels")
    return score, reasons


def suggest_targets(frame):
    """Rank every candidate column, best first (REQ-TG-1, REQ-TG-2)."""
    last = frame.columns[-1]
    suggestions = []
    for name in frame.columns:
        series = frame[name]
        if not is_candidate(series):
            continue
        score, reasons = score_candidate(series, name == last)
        suggestions.append(
            {
                "column": str(name),
                "score": score,
                "n_classes": int(series.nunique(dropna=True)),
                "reasons": reasons,
            }
        )
    suggestions.sort(key=lambda item: item["score"], reverse=True)
    return suggestions


def analyse_task(series):
    """Classify the task for a chosen target column (REQ-TG-4, REQ-TG-6).

    Returns task ('binary', 'multiclass' or 'unsupported'), class list, per-class
    counts, imbalance ratio, an optional warning, and whether training may start.
    """
    counts = series.dropna().astype(str).value_counts()
    n_classes = int(len(counts))
    analysis = {
        "task": "unsupported",
        "n_classes": n_classes,
        "classes": [str(label) for label in counts.index],
        "distribution": {str(label): int(count) for label, count in counts.items()},
        "imbalance_ratio": None,
        "warning": None,
        "trainable": False,
        "reason": None,
    }

    if n_classes < MIN_CLASSES:
        analysis["reason"] = (
            f"Column '{series.name}' has only {n_classes} distinct value, so there is nothing "
            "to classify. Choose another target."
        )
        return analysis
    if n_classes > MAX_CLASSES:
        analysis["reason"] = (
            f"Column '{series.name}' has {n_classes} distinct values. This looks like a "
            f"regression target; only classification with {MIN_CLASSES} to {MAX_CLASSES} "
            "classes is supported."
        )
        return analysis

    analysis["task"] = "binary" if n_classes == 2 else "multiclass"
    ratio = float(counts.max() / counts.min())
    analysis["imbalance_ratio"] = round(ratio, 2)
    if counts.min() < 2:
        analysis["reason"] = (
            f"Class '{counts.idxmin()}' has a single row, so a stratified train/test split is "
            "impossible. Remove that row or choose another target."
        )
        return analysis
    if ratio > IMBALANCE_WARNING_RATIO:
        analysis["warning"] = (
            f"The classes are imbalanced ({ratio:.1f}:1). Accuracy is misleading here: a model "
            "that always predicts the larger class scores high. Compare PR-AUC, recall and F1; "
            "models are trained with balanced class weights and ranked by PR-AUC."
        )
    analysis["trainable"] = True
    return analysis
