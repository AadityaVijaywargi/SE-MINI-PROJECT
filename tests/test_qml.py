"""Variational quantum classifier (SM-40, SM-41, SM-42)."""

import math

import numpy as np
import pytest

from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from core.qml import QMLNotApplicable, QMLTrainer, QuantumFeatureMap
from tests.conftest import load_sample
from tests.test_evaluator import METRIC_KEYS


@pytest.fixture(name="banking_split", scope="module")
def fixture_banking_split():
    """The banking sample (binary), preprocessed once."""
    return Preprocessor().fit_split(load_sample("banking"), "Exited")


def test_feature_map_compresses_to_qubits_and_scales_to_pi(banking_split):
    feature_map = QuantumFeatureMap(n_qubits=4, seed=0)
    angles = feature_map.fit_transform(banking_split.x_train)
    assert angles.shape == (len(banking_split.y_train), 4)
    assert angles.min() >= 0 and angles.max() <= math.pi + 1e-9
    assert 0 < feature_map.variance_retained <= 1


def test_feature_map_clips_unseen_rows_to_the_angle_range(banking_split):
    feature_map = QuantumFeatureMap(n_qubits=2, seed=0)
    feature_map.fit_transform(banking_split.x_train)
    far_away = banking_split.x_test * 100
    angles = feature_map.transform(far_away)
    assert angles.min() >= 0 and angles.max() <= math.pi


def test_fewer_features_than_qubits_uses_every_feature():
    feature_map = QuantumFeatureMap(n_qubits=4, seed=0)
    angles = feature_map.fit_transform(np.random.default_rng(0).normal(size=(30, 2)))
    assert angles.shape == (30, 2)
    assert feature_map.variance_retained == pytest.approx(1.0)


def test_vqc_trains_and_scores_on_the_full_test_set(banking_split):
    trainer = QMLTrainer(seed=42, epochs=1, qubits=2)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert set(result) == METRIC_KEYS
    assert result["paradigm"] == "QML"
    assert result["test_rows"] == len(banking_split.y_test)
    assert result["roc_auc"] is not None
    proba = model.predict_proba(banking_split.x_test)
    assert proba.shape == (len(banking_split.y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_rows_are_capped_and_recorded(banking_split):
    trainer = QMLTrainer(seed=42, epochs=1, qubits=2, max_rows=50)
    trainer.train(banking_split)
    assert trainer.details["Training rows"] == 50
    assert trainer.details["Subsampled"] is True
    metadata = trainer.run_metadata()
    assert metadata["qml_subsampled_to"] == 50
    assert 0 < metadata["pca_variance_retained"] <= 1


def test_multiclass_target_is_refused():
    split = Preprocessor().fit_split(load_sample("three_class"), "Segment")
    with pytest.raises(QMLNotApplicable, match="binary targets only"):
        QMLTrainer(seed=42, epochs=1).train(split)


def test_same_seed_same_predictions(banking_split):
    first = QMLTrainer(seed=7, epochs=1, qubits=2).train(banking_split)[0]
    second = QMLTrainer(seed=7, epochs=1, qubits=2).train(banking_split)[0]
    assert np.allclose(
        first.predict_proba(banking_split.x_test), second.predict_proba(banking_split.x_test)
    )


def test_progress_is_reported_every_epoch(banking_split):
    calls = []
    QMLTrainer(seed=1, epochs=2, qubits=2).train(
        banking_split, lambda f, m=None: calls.append((f, m))
    )
    assert [fraction for fraction, _ in calls] == [0.5, 1.0]
    assert calls[-1][1].startswith("epoch 2 of 2")


@pytest.mark.slow
def test_default_vqc_learns_better_than_chance(banking_split):
    trainer = QMLTrainer(seed=42)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert trainer.details["Qubits"] == 4 and trainer.details["Epochs"] == 20
    assert result["roc_auc"] > 0.6
