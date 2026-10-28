"""ML trainers and the model catalogue (SM-36, SM-37)."""

import numpy as np
import pandas as pd
import pytest

import config
from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from core.trainers import ML_MODELS, available_models, build_trainers, stratified_sample
from tests.conftest import load_sample
from tests.test_evaluator import METRIC_KEYS

ML_KEYS = list(ML_MODELS)


@pytest.fixture(name="banking_split", scope="module")
def fixture_banking_split():
    """The banking sample, preprocessed once for every test in this module."""
    return Preprocessor().fit_split(load_sample("banking"), "Exited")


def test_nine_models_available_in_paradigm_order():
    assert available_models() == [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "naive_bayes",
        "xgboost",
        "dnn",
        "vqc",
        "qsvm",
    ]


def test_paradigms_of_the_catalogue():
    trainers = build_trainers(available_models(), seed=42)
    assert [trainer.paradigm for trainer in trainers] == ["ML"] * 6 + ["DL", "QML", "QML"]


@pytest.mark.parametrize("key", ML_KEYS)
def test_each_model_trains_and_predicts(banking_split, key):
    (trainer,) = build_trainers([key], seed=42)
    model, seconds = trainer.train(banking_split)
    predictions = model.predict(banking_split.x_test)
    assert trainer.paradigm == "ML"
    assert seconds >= 0
    assert predictions.shape == banking_split.y_test.shape
    assert set(np.unique(predictions)) <= {0, 1}


@pytest.mark.parametrize("key", ML_KEYS)
def test_each_model_result_has_every_metric_key(banking_split, key):
    (trainer,) = build_trainers([key], seed=42)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert set(result) == METRIC_KEYS
    assert result["roc_auc"] is not None
    assert result["roc_curve"] is not None


def test_trainers_follow_catalogue_order():
    keys = [trainer.key for trainer in build_trainers(["vqc", "xgboost", "dnn", "svm"], seed=1)]
    assert keys == ["svm", "xgboost", "dnn", "vqc"]


def test_same_seed_same_predictions(banking_split):
    (first,) = build_trainers(["random_forest"], seed=42)
    (second,) = build_trainers(["random_forest"], seed=42)
    a = first.train(banking_split)[0].predict(banking_split.x_test)
    b = second.train(banking_split)[0].predict(banking_split.x_test)
    assert np.array_equal(a, b)


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="nope"):
        build_trainers(["svm", "nope"], seed=42)


def test_options_reach_dl_and_qml_trainers():
    dnn, vqc = build_trainers(["dnn", "vqc"], seed=3, options={"dnn_epochs": 7, "qubits": 3})
    assert dnn.epochs == 7
    assert vqc.qubits == 3
    assert vqc.epochs == config.QML_EPOCHS


def test_stratified_sample_keeps_class_shares():
    labels = np.array([0] * 900 + [1] * 100)
    features = np.arange(1000).reshape(-1, 1)
    sample_x, sample_y = stratified_sample(features, labels, 200, seed=0)
    assert len(sample_y) == 200
    assert sample_x.shape == (200, 1)
    assert int(sample_y.sum()) == 20


def test_stratified_sample_returns_small_inputs_unchanged():
    labels = np.array([0, 1, 0, 1])
    features = np.zeros((4, 2))
    sample_x, sample_y = stratified_sample(features, labels, 10, seed=0)
    assert sample_x is features and sample_y is labels


def test_svm_is_subsampled_above_the_row_cap(banking_split, monkeypatch):
    monkeypatch.setattr(config, "SVM_MAX_ROWS", 60)
    (trainer,) = build_trainers(["svm"], seed=42)
    model, _ = trainer.train(banking_split)
    assert trainer.details["Training rows"] == 60
    assert len(model.support_) <= 60
    assert model.predict(banking_split.x_test).shape == banking_split.y_test.shape


def test_svm_not_subsampled_below_the_cap(banking_split):
    (trainer,) = build_trainers(["svm"], seed=42)
    trainer.train(banking_split)
    assert "Training rows" not in trainer.details


def test_other_models_are_never_subsampled(banking_split, monkeypatch):
    monkeypatch.setattr(config, "SVM_MAX_ROWS", 60)
    (trainer,) = build_trainers(["random_forest"], seed=42)
    trainer.train(banking_split)
    assert trainer.details == {"Device": "CPU", "Class weights": "balanced"}


@pytest.mark.parametrize("key", ML_KEYS)
def test_balanced_models_find_a_rare_class(key):
    rng = np.random.default_rng(0)
    rows = 3000
    signal = rng.normal(size=rows)
    fraud = rng.random(rows) < 1 / (1 + np.exp(-(-5.5 + 2.5 * signal)))
    frame = pd.DataFrame(
        {"Signal": signal, "Noise": rng.normal(size=rows), "Fraud": np.where(fraud, "yes", "no")}
    )
    split = Preprocessor().fit_split(frame, "Fraud")
    trainer = build_trainers([key], seed=0)[0]
    model, seconds = trainer.train(split)
    result = evaluate(model, split, trainer, seconds)
    assert trainer.details["Class weights"] == "balanced"
    assert result["confusion_matrix"][1][1] > 0, "the model never flagged the rare class"
    assert result["pr_auc"] > split.metadata()["test_positive_rate"]
