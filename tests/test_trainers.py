"""ML trainers (SM-36)."""

import numpy as np
import pytest

from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from core.trainers import available_models, build_trainers
from tests.conftest import load_sample
from tests.test_evaluator import METRIC_KEYS


@pytest.fixture(name="banking_split", scope="module")
def fixture_banking_split():
    """The banking sample, preprocessed once for every test in this module."""
    return Preprocessor().fit_split(load_sample("banking"), "Exited")


def test_six_ml_models_available():
    assert available_models() == [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "naive_bayes",
        "xgboost",
    ]


@pytest.mark.parametrize("key", available_models())
def test_each_model_trains_and_predicts(banking_split, key):
    (trainer,) = build_trainers([key], seed=42)
    model, seconds = trainer.train(banking_split)
    predictions = model.predict(banking_split.x_test)
    assert trainer.paradigm == "ML"
    assert seconds >= 0
    assert predictions.shape == banking_split.y_test.shape
    assert set(np.unique(predictions)) <= {0, 1}


@pytest.mark.parametrize("key", available_models())
def test_each_model_result_has_every_metric_key(banking_split, key):
    (trainer,) = build_trainers([key], seed=42)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert set(result) == METRIC_KEYS
    assert result["roc_auc"] is not None


def test_trainers_follow_catalogue_order():
    keys = [trainer.key for trainer in build_trainers(["xgboost", "svm"], seed=1)]
    assert keys == ["svm", "xgboost"]


def test_same_seed_same_predictions(banking_split):
    (first,) = build_trainers(["random_forest"], seed=42)
    (second,) = build_trainers(["random_forest"], seed=42)
    a = first.train(banking_split)[0].predict(banking_split.x_test)
    b = second.train(banking_split)[0].predict(banking_split.x_test)
    assert np.array_equal(a, b)


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="dnn"):
        build_trainers(["svm", "dnn"], seed=42)
