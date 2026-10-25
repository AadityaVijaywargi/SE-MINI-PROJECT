"""Variational quantum classifier (SM-40, SM-41, SM-42).

Implements REQ-TR-5 to REQ-TR-7, CON-4 and CON-5. The pipeline is: the shared encoded
features, PCA down to the qubit count, scaling to [0, pi], angle encoding, a 2-layer
variational circuit, the expectation of Z on qubit 0, and the class-1 probability
(1 - <Z>) / 2, which the evaluator thresholds like every other binary model. It runs on a
PennyLane simulator (config.QML_DEVICE), binary targets only, on at most QML_MAX_ROWS
training rows. The held-out test rows are never
subsampled, so the quantum model is scored on exactly the same rows as every other model.
"""

import math

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import config
from core import gpu
from core.base import Trainer, balanced_weights, stratified_sample

EPSILON = 1e-7


class QMLNotApplicable(ValueError):
    """The quantum classifier cannot run on this target. The message is shown to the user."""


def make_circuit(n_qubits, device_name=config.QML_DEVICE):
    """QNode: angle encoding, then strongly entangling layers, then <Z> on qubit 0.

    ``x`` may be one row or a batch of rows (parameter broadcasting). Lightning uses the
    adjoint method; default.qubit backpropagates through the whole batch at once.
    """
    device = qml.device(device_name, wires=n_qubits)
    diff_method = "adjoint" if device_name.startswith("lightning") else "backprop"

    @qml.qnode(device, diff_method=diff_method)
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def class_one_probability(circuit, weights, x):
    """P(class 1) = (1 - <Z>) / 2, which lies in [0, 1]."""
    return (1 - circuit(weights, x)) / 2


def bce_loss(circuit, weights, x, y, row_weights=None):  # pylint: disable=too-many-arguments
    """Binary cross-entropy of the circuit's class-1 probability, averaged over the rows
    with ``row_weights`` (all 1 when omitted)."""
    # pennylane.numpy wraps NumPy's functions at import time, so pylint cannot see them.
    # pylint: disable=no-member
    prob = pnp.clip(class_one_probability(circuit, weights, x), EPSILON, 1 - EPSILON)
    losses = y * pnp.log(prob) + (1 - y) * pnp.log(1 - prob)
    if row_weights is None:
        return -pnp.mean(losses)
    return -pnp.sum(row_weights * losses) / pnp.sum(row_weights)


class QuantumFeatureMap:
    """PCA to the qubit count, then scaling to [0, pi] (REQ-TR-5), fitted on training rows."""

    def __init__(self, n_qubits, seed):
        self.n_qubits = n_qubits
        self.seed = seed
        self.pca = None
        self.scaler = None

    def fit_transform(self, features):
        """Fit PCA and the scaler on ``features`` and return the encoded angles."""
        components = min(self.n_qubits, features.shape[1], features.shape[0])
        self.pca = PCA(n_components=components, random_state=self.seed)
        self.scaler = MinMaxScaler(feature_range=(0, math.pi))
        return self.scaler.fit_transform(self.pca.fit_transform(features))

    def transform(self, features):
        """Encode unseen rows; values outside the training range are clipped to [0, pi]."""
        return np.clip(self.scaler.transform(self.pca.transform(features)), 0, math.pi)

    @property
    def variance_retained(self):
        """Share of the feature variance the PCA components keep (REQ-TR-7)."""
        return float(np.sum(self.pca.explained_variance_ratio_))


class VQCModel:
    """A trained circuit with the scikit-learn ``predict`` / ``predict_proba`` interface."""

    def __init__(self, circuit, weights, feature_map):
        self.circuit = circuit
        self.weights = weights
        self.feature_map = feature_map

    def predict_proba(self, features):
        """Probabilities ``[P(class 0), P(class 1)]`` per row."""
        angles = self.feature_map.transform(np.asarray(features, dtype=np.float64))
        positive = np.asarray(class_one_probability(self.circuit, self.weights, angles))
        positive = np.clip(positive.reshape(-1), 0.0, 1.0)
        return np.column_stack([1 - positive, positive])

    def predict(self, features):
        """Class 1 when its probability is at least 0.5 (thresholding, REQ-TR-5)."""
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


class QMLTrainer(Trainer):  # pylint: disable=too-many-instance-attributes
    """The single quantum model (REQ-MS-1). Binary targets only (REQ-TR-6, BR-3)."""

    key = "vqc"
    name = "Variational Quantum Classifier"
    paradigm = "QML"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        seed,
        *,
        epochs=config.QML_EPOCHS,
        qubits=config.QML_QUBITS,
        layers=config.QML_LAYERS,
        max_rows=config.QML_MAX_ROWS,
        batch_size=config.QML_BATCH_SIZE,
        learning_rate=config.QML_LEARNING_RATE,
        balance=True,
    ):
        super().__init__(balance)
        self.seed = seed
        self.epochs = epochs
        self.qubits = qubits
        self.layers = layers
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, split, report):
        if split.n_classes != 2:
            raise QMLNotApplicable(
                "The quantum classifier supports binary targets only; this target has "
                f"{split.n_classes} classes."
            )
        features, labels = stratified_sample(split.x_train, split.y_train, self.max_rows, self.seed)
        feature_map = QuantumFeatureMap(self.qubits, self.seed)
        angles = feature_map.fit_transform(np.asarray(features, dtype=np.float64))
        targets = np.asarray(labels, dtype=np.float64)

        circuit = make_circuit(self.qubits)
        weights = self.optimise(circuit, angles, targets, report)

        self.details.update(
            {
                "Qubits": self.qubits,
                "Layers": self.layers,
                "Epochs": self.epochs,
                "Training rows": int(len(targets)),
                "Subsampled": len(targets) < len(split.y_train),
                "PCA variance retained": round(feature_map.variance_retained, 4),
                "Trainable parameters": int(np.prod(self.weight_shape)),
                "Class weights": "balanced" if self.balance else "none",
                "Simulator": config.QML_DEVICE,
                "Device": gpu.device_label(config.QML_DEVICE == "lightning.gpu"),
            }
        )
        return VQCModel(circuit, weights, feature_map)

    @property
    def weight_shape(self):
        """Shape of the trainable rotation angles: (layers, qubits, 3)."""
        return qml.StronglyEntanglingLayers.shape(n_layers=self.layers, n_wires=self.qubits)

    def batches(self, rng, rows):
        """Row indices for one epoch, shuffled and cut into mini-batches."""
        order = rng.permutation(rows)
        for start in range(0, rows, self.batch_size):
            yield order[start : start + self.batch_size]

    def optimise(self, circuit, angles, targets, report):
        """Mini-batch Adam on the (class-weighted, with ``balance``) binary cross-entropy;
        reports the loss after each epoch."""
        rng = np.random.default_rng(self.seed)
        weights = pnp.array(rng.uniform(0, 2 * math.pi, self.weight_shape), requires_grad=True)
        row_weights = balanced_weights(targets) if self.balance else np.ones(len(targets))
        optimiser = qml.AdamOptimizer(stepsize=self.learning_rate)
        for epoch in range(self.epochs):
            for batch in self.batches(rng, len(targets)):
                x_batch = pnp.array(angles[batch], requires_grad=False)
                y_batch = pnp.array(targets[batch], requires_grad=False)
                w_batch = pnp.array(row_weights[batch], requires_grad=False)
                weights = optimiser.step(
                    lambda w, xb=x_batch, yb=y_batch, wb=w_batch: bce_loss(circuit, w, xb, yb, wb),
                    weights,
                )
            loss = float(
                bce_loss(
                    circuit,
                    weights,
                    pnp.array(angles),
                    pnp.array(targets),
                    pnp.array(row_weights, requires_grad=False),
                )
            )
            report(
                (epoch + 1) / self.epochs, f"epoch {epoch + 1} of {self.epochs}, loss {loss:.3f}"
            )
        return weights

    def run_metadata(self):
        """Fields the results metadata must carry for QML (REQ-TR-7, REQ-EV-5, BR-3)."""
        return {
            "qml_subsampled_to": self.details.get("Training rows"),
            "pca_variance_retained": self.details.get("PCA variance retained"),
        }
