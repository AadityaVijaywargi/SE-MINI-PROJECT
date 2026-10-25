"""The Trainer interface shared by the ML, DL and QML trainers (SAD class diagram).

Kept apart from core/trainers.py so core/dnn.py and core/qml.py can subclass Trainer
without importing the model catalogue that, in turn, loads them.
"""

import time
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


def no_report(_fraction, _message=None):
    """Progress callback used when nobody is listening."""


class Trainer(ABC):
    """One model: a key used by the API, a display name and its paradigm (ML, DL or QML).

    ``details`` collects facts about the fitted model (rows used, epochs run, qubits) that
    are reported with its result. ``balance`` weights the classes equally while training,
    so a rare class is not ignored.
    """

    key = ""
    name = ""
    paradigm = ""

    def __init__(self, balance=True):
        self.details = {}
        self.balance = balance

    @abstractmethod
    def fit(self, split, report):
        """Train on ``split.x_train`` / ``split.y_train`` and return the fitted model.

        ``report(fraction, message)`` may be called with progress inside the model, from 0
        to 1. It raises JobCancelled when the user has cancelled the job.
        """

    def train(self, split, report=None):
        """Train and time the model. Returns ``(model, train_time_seconds)``."""
        self.details = {}
        start = time.perf_counter()
        model = self.fit(split, report or no_report)
        return model, time.perf_counter() - start


def balanced_weights(labels):
    """Per-row weights that give every class the same total weight.

    Without them a model trained on 1% fraud learns that predicting "not fraud" for every
    row is 99% accurate and never flags anything.
    """
    return compute_sample_weight("balanced", labels)


def stratified_holdout(features, labels, ratio, seed):
    """Split off ``ratio`` of the rows, stratified when every class has enough rows.

    Returns ``(x_keep, x_held, y_keep, y_held)``.
    """
    try:
        return train_test_split(
            features, labels, test_size=ratio, random_state=seed, stratify=labels
        )
    except ValueError:
        return train_test_split(features, labels, test_size=ratio, random_state=seed)


def stratified_sample(features, labels, max_rows, seed):
    """At most ``max_rows`` rows, keeping the class proportions (REQ-TR-2, REQ-TR-6).

    Returns the inputs unchanged when they are already small enough.
    """
    if len(labels) <= max_rows:
        return features, labels
    try:
        sample_x, _, sample_y, _ = train_test_split(
            features, labels, train_size=max_rows, random_state=seed, stratify=labels
        )
    except ValueError:
        # A class too small to stratify: fall back to a plain random sample.
        sample_x, _, sample_y, _ = train_test_split(
            features, labels, train_size=max_rows, random_state=seed
        )
    return sample_x, sample_y
