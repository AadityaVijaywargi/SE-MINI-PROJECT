"""Model trainers (SM-36).

Implements REQ-TR-1. Every trainer takes the shared Split and returns a fitted model; the
DNN (SM-38) and QML (SM-40) trainers plug into the same interface in Sprint 2.
"""

import time
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Trainer(ABC):
    """One model: a key used by the API, a display name and its paradigm (ML, DL or QML)."""

    key = ""
    name = ""
    paradigm = ""

    @abstractmethod
    def fit(self, split):
        """Train on ``split.x_train`` / ``split.y_train`` and return the fitted model."""

    def train(self, split):
        """Train and time the model. Returns ``(model, train_time_seconds)``."""
        start = time.perf_counter()
        model = self.fit(split)
        return model, time.perf_counter() - start


class MLTrainer(Trainer):
    """A scikit-learn style classifier built from a factory that takes the seed."""

    paradigm = "ML"

    def __init__(self, key, name, factory, seed):
        self.key = key
        self.name = name
        self.factory = factory
        self.seed = seed

    def fit(self, split):
        model = self.factory(self.seed)
        model.fit(split.x_train, split.y_train)
        return model


ML_MODELS = {
    "logistic_regression": (
        "Logistic Regression",
        lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    ),
    "decision_tree": (
        "Decision Tree",
        lambda seed: DecisionTreeClassifier(random_state=seed),
    ),
    "random_forest": (
        "Random Forest",
        lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    ),
    "svm": (
        "SVM",
        lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
    ),
    "naive_bayes": (
        "Naive Bayes",
        lambda seed: GaussianNB(),
    ),
    "xgboost": (
        "XGBoost",
        lambda seed: XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1),
    ),
}


def available_models():
    """Keys of every model that can be trained in this release."""
    return list(ML_MODELS)


def build_trainers(model_keys, seed):
    """Create trainers for the requested keys, in catalogue order.

    Raises ValueError naming any key that is not a known, available model.
    """
    unknown = [key for key in model_keys if key not in ML_MODELS]
    if unknown:
        raise ValueError(f"Unknown or unavailable model(s): {', '.join(map(str, unknown))}.")
    return [
        MLTrainer(key, name, factory, seed)
        for key, (name, factory) in ML_MODELS.items()
        if key in model_keys
    ]
