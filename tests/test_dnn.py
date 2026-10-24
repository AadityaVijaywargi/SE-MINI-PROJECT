"""Deep neural network trainer (SM-38, SM-39)."""

import numpy as np
import pytest

from core.dnn import DNNTrainer, build_network, validation_split
from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from tests.conftest import load_sample
from tests.test_evaluator import METRIC_KEYS

FAST_EPOCHS = 15


@pytest.fixture(name="banking_split", scope="module")
def fixture_banking_split():
    """The banking sample (binary), preprocessed once."""
    return Preprocessor().fit_split(load_sample("banking"), "Exited")


@pytest.fixture(name="three_class_split", scope="module")
def fixture_three_class_split():
    """The three-class sample, preprocessed once."""
    return Preprocessor().fit_split(load_sample("three_class"), "Segment")


def layer_summary(network):
    """(class name, units or rate) for every layer, to compare with REQ-TR-3."""
    summary = []
    for layer in network.layers:
        config = layer.get_config()
        summary.append((layer.__class__.__name__, config.get("units", config.get("rate"))))
    return summary


def test_binary_architecture_matches_srs():
    network = build_network(10, 2)
    assert layer_summary(network) == [
        ("Dense", 128),
        ("BatchNormalization", None),
        ("Dropout", 0.3),
        ("Dense", 64),
        ("Dropout", 0.2),
        ("Dense", 32),
        ("Dense", 1),
    ]
    assert network.layers[-1].get_config()["activation"] == "sigmoid"
    assert network.loss == "binary_crossentropy"
    assert network.optimizer.learning_rate.numpy() == pytest.approx(1e-3)


def test_multiclass_output_is_softmax():
    network = build_network(6, 3)
    assert network.layers[-1].get_config()["units"] == 3
    assert network.layers[-1].get_config()["activation"] == "softmax"
    assert network.loss == "sparse_categorical_crossentropy"


def test_validation_split_comes_from_training_rows_only(banking_split):
    x_fit, x_val, y_fit, y_val = validation_split(
        banking_split.x_train, banking_split.y_train, seed=42
    )
    assert len(y_fit) + len(y_val) == len(banking_split.y_train)
    assert len(y_val) == round(0.2 * len(banking_split.y_train))
    training_rows = {row.tobytes() for row in banking_split.x_train}
    assert all(row.tobytes() in training_rows for row in x_val)
    assert all(row.tobytes() in training_rows for row in x_fit)


def test_binary_dnn_trains_and_scores(banking_split):
    trainer = DNNTrainer(seed=42, epochs=FAST_EPOCHS)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert set(result) == METRIC_KEYS
    assert result["paradigm"] == "DL"
    assert result["roc_auc"] is not None
    proba = model.predict_proba(banking_split.x_test)
    assert proba.shape == (len(banking_split.y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert 1 <= trainer.details["Epochs run"] <= FAST_EPOCHS
    assert trainer.details["Validation rows"] == 32


def test_multiclass_dnn_predicts_every_row(three_class_split):
    trainer = DNNTrainer(seed=42, epochs=FAST_EPOCHS)
    model, seconds = trainer.train(three_class_split)
    result = evaluate(model, three_class_split, trainer, seconds)
    assert result["roc_auc"] is None
    assert model.predict_proba(three_class_split.x_test).shape == (40, 3)
    assert set(np.unique(model.predict(three_class_split.x_test))) <= {0, 1, 2}


def test_early_stopping_restores_the_best_epoch(banking_split):
    trainer = DNNTrainer(seed=42, epochs=300)
    trainer.train(banking_split)
    assert trainer.details["Stopped early"] is True
    assert trainer.details["Epochs run"] == trainer.details["Best epoch"] + 10


def test_progress_is_reported_every_epoch(banking_split):
    calls = []
    DNNTrainer(seed=42, epochs=3).train(banking_split, lambda f, m=None: calls.append((f, m)))
    assert [round(fraction, 2) for fraction, _ in calls] == [0.33, 0.67, 1.0]
    assert "epoch 1 of up to 3" in calls[0][1]


def test_report_can_stop_training(banking_split):
    class Stop(Exception):
        """Raised by the callback to stop training."""

    def report(_fraction, _message=None):
        raise Stop()

    with pytest.raises(Stop):
        DNNTrainer(seed=42, epochs=50).train(banking_split, report)
