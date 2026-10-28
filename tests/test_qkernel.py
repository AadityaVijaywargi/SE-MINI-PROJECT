"""Quantum kernel SVM: kernel correctness, model behaviour and evaluator compatibility."""

import math

import numpy as np
import pennylane as qml
import pytest
from sklearn.svm import SVC

from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from core.qkernel import (
    QuantumKernelSVMTrainer,
    encode_states,
    fidelity_kernel,
    make_fidelity_circuit,
    make_state_circuit,
)
from core.qml import QMLNotApplicable
from tests.conftest import load_sample
from tests.test_evaluator import METRIC_KEYS

N_QUBITS = 3


@pytest.fixture(name="angles", scope="module")
def fixture_angles():
    """Random rows already scaled to [0, pi]."""
    return np.random.default_rng(0).uniform(0, math.pi, (12, N_QUBITS))


@pytest.fixture(name="states", scope="module")
def fixture_states(angles):
    return encode_states(make_state_circuit(N_QUBITS), angles)


@pytest.fixture(name="banking_split", scope="module")
def fixture_banking_split():
    return Preprocessor().fit_split(load_sample("banking"), "Exited")


# ---- the kernel is mathematically correct -------------------------------------------


def test_states_are_normalised(states):
    assert states.shape == (12, 2**N_QUBITS)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0)


def test_batched_encoding_matches_one_row_at_a_time(angles, states):
    circuit = make_state_circuit(N_QUBITS)
    single = np.stack([np.asarray(circuit(row)) for row in angles])
    assert np.allclose(states, single)


def test_chunking_does_not_change_the_states(angles, states, monkeypatch):
    monkeypatch.setattr("core.qkernel.CHUNK_ROWS", 5)
    assert np.allclose(encode_states(make_state_circuit(N_QUBITS), angles), states)


def test_kernel_is_symmetric_unit_diagonal_and_bounded(states):
    gram = fidelity_kernel(states, states)
    assert np.allclose(gram, gram.T)
    assert np.allclose(np.diag(gram), 1.0)
    assert gram.min() >= -1e-12 and gram.max() <= 1 + 1e-12


def test_kernel_is_positive_semi_definite(states):
    eigenvalues = np.linalg.eigvalsh(fidelity_kernel(states, states))
    assert eigenvalues.min() > -1e-9


def test_kernel_is_not_trivial(states):
    gram = fidelity_kernel(states, states)
    off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
    assert off_diagonal.std() > 0.01
    assert off_diagonal.max() < 1 - 1e-6


def test_matrix_product_matches_the_compute_uncompute_circuit(angles, states):
    """The statevector shortcut equals the hardware-style fidelity circuit, entry by entry."""
    circuit = make_fidelity_circuit(N_QUBITS)
    gram = fidelity_kernel(states, states)
    for i, j in [(0, 1), (2, 7), (5, 5), (11, 3)]:
        all_zeros = float(np.asarray(circuit(angles[i], angles[j]))[0])
        assert all_zeros == pytest.approx(gram[i, j], abs=1e-10)


def test_kernel_matches_an_independent_numpy_statevector():
    """With one qubit there are no ZZ terms, so the circuit can be built by hand in NumPy."""
    x_a, x_b = 0.7, 2.1
    circuit = make_state_circuit(1)
    states = encode_states(circuit, np.array([[x_a], [x_b]]))
    hadamard = np.array([[1, 1], [1, -1]]) / math.sqrt(2)

    def by_hand(x):
        phase = np.diag([np.exp(-1j * x / 2), np.exp(1j * x / 2)])
        state = np.array([1, 0], dtype=complex)
        for _ in range(2):  # n_repeats = 2: (H then RZ(x)) twice
            state = phase @ (hadamard @ state)
        return state

    manual = np.abs(np.vdot(by_hand(x_a), by_hand(x_b))) ** 2
    assert fidelity_kernel(states, states)[0, 1] == pytest.approx(manual, abs=1e-10)


# ---- the SVM on that kernel actually classifies --------------------------------------


def test_svm_on_the_kernel_separates_a_toy_problem():
    rng = np.random.default_rng(1)
    zero = rng.normal(0.8, 0.15, (40, 2))
    one = rng.normal(2.3, 0.15, (40, 2))
    angles = np.clip(np.vstack([zero, one]), 0, math.pi)
    labels = np.array([0] * 40 + [1] * 40)
    circuit = make_state_circuit(2)
    states = encode_states(circuit, angles)
    svc = SVC(kernel="precomputed").fit(fidelity_kernel(states, states), labels)
    assert svc.score(fidelity_kernel(states, states), labels) > 0.95


# ---- the trainer plugs into the existing pipeline ------------------------------------


def test_trainer_fits_and_scores_on_the_full_test_set(banking_split):
    trainer = QuantumKernelSVMTrainer(seed=42, qubits=3)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert set(result) == METRIC_KEYS
    assert result["paradigm"] == "QML"
    assert result["test_rows"] == len(banking_split.y_test)
    assert result["roc_auc"] is not None
    proba = model.predict_proba(banking_split.x_test)
    assert proba.shape == (len(banking_split.y_test), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(np.unique(model.predict(banking_split.x_test))) <= {0, 1}


def test_trainer_learns_better_than_chance(banking_split):
    trainer = QuantumKernelSVMTrainer(seed=42, qubits=4)
    model, seconds = trainer.train(banking_split)
    result = evaluate(model, banking_split, trainer, seconds)
    assert result["roc_auc"] > 0.6


def test_rows_are_capped_and_recorded(banking_split):
    trainer = QuantumKernelSVMTrainer(seed=42, qubits=2, max_rows=50)
    trainer.train(banking_split)
    assert trainer.details["Training rows"] == 50
    assert trainer.details["Subsampled"] is True
    assert trainer.details["Support vectors"] > 0
    metadata = trainer.run_metadata()
    assert metadata["qml_subsampled_to"] == 50
    assert 0 < metadata["pca_variance_retained"] <= 1


def test_multiclass_target_is_refused():
    split = Preprocessor().fit_split(load_sample("three_class"), "Segment")
    with pytest.raises(QMLNotApplicable, match="binary targets only"):
        QuantumKernelSVMTrainer(seed=42).train(split)


def test_same_seed_same_predictions(banking_split):
    first = QuantumKernelSVMTrainer(seed=7, qubits=2).train(banking_split)[0]
    second = QuantumKernelSVMTrainer(seed=7, qubits=2).train(banking_split)[0]
    assert np.allclose(
        first.predict_proba(banking_split.x_test), second.predict_proba(banking_split.x_test)
    )


def test_progress_is_reported_and_ends_at_one(banking_split):
    calls = []
    QuantumKernelSVMTrainer(seed=1, qubits=2).train(
        banking_split, lambda f, m=None: calls.append((f, m))
    )
    fractions = [fraction for fraction, _ in calls]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_cancellation_propagates(banking_split):
    class Cancelled(Exception):
        pass

    def cancel(_fraction, _message=None):
        raise Cancelled

    with pytest.raises(Cancelled):
        QuantumKernelSVMTrainer(seed=1, qubits=2).train(banking_split, cancel)


def test_works_with_fewer_features_than_qubits():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 2))
    labels = (features[:, 0] > 0).astype(int)

    class Split:  # minimal stand-in for core.preprocessing.Split
        n_classes = 2
        x_train, y_train = features, labels

    model = QuantumKernelSVMTrainer(seed=0, qubits=6).fit(Split, lambda *_: None)
    assert model.predict(features).shape == (60,)
    assert qml.math.shape(model.train_states)[1] == 4  # 2 usable qubits


def test_smaller_bandwidth_widens_the_kernel(angles):
    circuit = make_state_circuit(N_QUBITS)

    def mean_off_diagonal(scale):
        states = encode_states(circuit, angles * scale)
        gram = fidelity_kernel(states, states)
        return gram[~np.eye(len(gram), dtype=bool)].mean()

    assert mean_off_diagonal(0.25) > mean_off_diagonal(1.0)


def test_bandwidth_is_recorded_and_used_at_predict_time(banking_split):
    trainer = QuantumKernelSVMTrainer(seed=3, qubits=2, bandwidth=0.5)
    model, _ = trainer.train(banking_split)
    assert trainer.details["Bandwidth"] == 0.5 and model.bandwidth == 0.5
    gram = model.gram(banking_split.x_test)
    assert gram.shape == (len(banking_split.y_test), model.train_states.shape[0])
