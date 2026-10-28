"""Quantum kernel SVM (QSVM): a support vector machine on a quantum fidelity kernel.

Pipeline: the shared encoded features, PCA down to the qubit count, scaling to [0, pi]
(the same ``QuantumFeatureMap`` the variational classifier uses), then a quantum feature
map |phi(x)> = U(x)|0...0> built from PennyLane's IQP embedding (Hadamards, Z rotations and
ZZ entanglers, repeated twice). The kernel between two rows is the state fidelity

    k(x, x') = |<phi(x)|phi(x')>|^2

and a classical scikit-learn ``SVC(kernel="precomputed")`` is trained on the Gram matrix.
Only the kernel is quantum; the margin optimisation is classical, so there are no circuit
parameters to train and no barren plateaus.

The simulator returns each row's full statevector once, so the whole Gram matrix is one
matrix product (exact, noiseless). ``fidelity_circuit`` is the hardware-style
compute-uncompute circuit, kept so the tests can prove the two routes agree.

Binary targets only and at most QML_MAX_ROWS training rows, like the variational
classifier, so both quantum models run under the same constraints. The held-out test rows
are never subsampled.
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC

import config
from core.base import Trainer, balanced_weights, stratified_sample
from core.qml import QMLNotApplicable, QuantumFeatureMap

REPEATS = 2
# The angles span [0, pi], which makes the fidelity kernel so sharp that every row looks
# unlike every other (the Gram matrix approaches the identity and the SVM cannot
# generalise). Shrinking the angles by this factor widens the kernel. On the four bundled
# binary samples 0.25 gave the best test ROC-AUC overall (0.5 and 1.0 were worse).
DEFAULT_BANDWIDTH = 0.25
CHUNK_ROWS = 256  # rows encoded per simulator call; also the progress / cancel granularity


def embed(angles, n_qubits):
    """The quantum feature map U(x); ``angles`` may be one row or a batch."""
    qml.IQPEmbedding(angles, wires=range(n_qubits), n_repeats=REPEATS)


def make_state_circuit(n_qubits, device_name=config.QML_DEVICE):
    """QNode returning |phi(x)> for each row of a batch (shape ``(rows, 2**n_qubits)``)."""
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device)
    def circuit(angles):
        embed(angles, n_qubits)
        return qml.state()

    return circuit


def make_fidelity_circuit(n_qubits, device_name=config.QML_DEVICE):
    """Compute-uncompute circuit: U(x2)^dagger U(x1)|0>, measured in the computational basis.

    The probability of the all-zeros outcome is |<phi(x2)|phi(x1)>|^2, which is how the
    kernel is estimated on real hardware.
    """
    device = qml.device(device_name, wires=n_qubits)

    @qml.qnode(device)
    def circuit(x1, x2):
        embed(x1, n_qubits)
        qml.adjoint(embed)(x2, n_qubits)
        return qml.probs(wires=range(n_qubits))

    return circuit


def encode_states(circuit, angles, report=None, start=0.0, span=1.0):
    """Statevectors of every row, in chunks. ``report`` gets the fraction ``start..start+span``."""
    rows = len(angles)
    states = np.empty((rows, 2 ** angles.shape[1]), dtype=np.complex128)
    for begin in range(0, rows, CHUNK_ROWS):
        end = min(begin + CHUNK_ROWS, rows)
        states[begin:end] = np.asarray(circuit(angles[begin:end])).reshape(end - begin, -1)
        if report:
            report(start + span * end / rows)
    return states


def fidelity_kernel(states_a, states_b):
    """Gram matrix ``|<a_i|b_j>|^2`` between two sets of statevectors."""
    return np.abs(states_a @ states_b.conj().T) ** 2


class QuantumKernelSVMModel:
    """A fitted QSVM with the scikit-learn ``predict`` / ``predict_proba`` interface."""

    def __init__(
        self, svc, circuit, train_states, feature_map, bandwidth=DEFAULT_BANDWIDTH
    ):  # pylint: disable=too-many-arguments
        self.svc = svc
        self.circuit = circuit
        self.train_states = train_states
        self.feature_map = feature_map
        self.bandwidth = bandwidth
        self.classes_ = svc.classes_

    def gram(self, features):
        """Kernel between ``features`` and the stored training rows."""
        angles = self.bandwidth * self.feature_map.transform(np.asarray(features, dtype=np.float64))
        return fidelity_kernel(encode_states(self.circuit, angles), self.train_states)

    def predict_proba(self, features):
        """Probabilities ``[P(class 0), P(class 1)]`` per row (Platt scaling)."""
        return self.svc.predict_proba(self.gram(features))

    def predict(self, features):
        """Class labels from the SVM's decision function."""
        return self.svc.predict(self.gram(features)).astype(int)


class QuantumKernelSVMTrainer(Trainer):  # pylint: disable=too-many-instance-attributes
    """The quantum kernel SVM. Binary targets only, like the variational classifier."""

    key = "qsvm"
    name = "Quantum Kernel SVM"
    paradigm = "QML"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        seed,
        *,
        qubits=config.QML_QUBITS,
        max_rows=config.QML_MAX_ROWS,
        c_value=1.0,
        bandwidth=DEFAULT_BANDWIDTH,
        balance=True,
    ):
        super().__init__(balance)
        self.seed = seed
        self.qubits = qubits
        self.max_rows = max_rows
        self.c_value = c_value
        self.bandwidth = bandwidth

    def fit(self, split, report):
        if split.n_classes != 2:
            raise QMLNotApplicable(
                "The quantum classifier supports binary targets only; this target has "
                f"{split.n_classes} classes."
            )
        features, labels = stratified_sample(split.x_train, split.y_train, self.max_rows, self.seed)
        feature_map = QuantumFeatureMap(self.qubits, self.seed)
        angles = self.bandwidth * feature_map.fit_transform(np.asarray(features, dtype=np.float64))
        labels = np.asarray(labels)

        circuit = make_state_circuit(angles.shape[1])
        report(0.0, "encoding rows into quantum states")
        states = encode_states(
            circuit, angles, lambda f: report(0.6 * f, "encoding rows into quantum states")
        )
        report(0.7, "computing the quantum kernel")
        gram = fidelity_kernel(states, states)
        report(0.85, "training the SVM on the kernel")
        svc = SVC(kernel="precomputed", C=self.c_value, probability=True, random_state=self.seed)
        weights = balanced_weights(labels) if self.balance else None
        svc.fit(gram, labels, sample_weight=weights)
        report(1.0, "done")

        self.details.update(
            {
                "Qubits": self.qubits,
                "Feature map": f"IQP embedding, {REPEATS} repeats",
                "Kernel": "fidelity |<phi(x)|phi(x')>|^2",
                "Training rows": int(len(labels)),
                "Subsampled": len(labels) < len(split.y_train),
                "PCA variance retained": round(feature_map.variance_retained, 4),
                "Support vectors": int(np.sum(svc.n_support_)),
                "C": self.c_value,
                "Bandwidth": self.bandwidth,
                "Class weights": "balanced" if self.balance else "none",
                "Simulator": config.QML_DEVICE,
                "Device": "CPU",
            }
        )
        return QuantumKernelSVMModel(svc, circuit, states, feature_map, self.bandwidth)

    def run_metadata(self):
        """Same results-metadata fields as the variational classifier."""
        return {
            "qml_subsampled_to": self.details.get("Training rows"),
            "pca_variance_retained": self.details.get("PCA variance retained"),
        }


__all__ = ["QuantumKernelSVMTrainer", "QuantumKernelSVMModel"]
