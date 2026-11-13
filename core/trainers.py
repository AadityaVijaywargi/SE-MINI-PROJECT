"""Model trainers (SM-36, SM-37, SM-38, SM-40).

Implements REQ-TR-1 and REQ-TR-2. Every trainer takes the shared Split and returns a fitted
model with ``predict`` and ``predict_proba``, so the evaluator treats all paradigms alike.
The DNN lives in core/dnn.py and the quantum classifier in core/qml.py; each imports its
heavy library only when it trains. XGBoost trains on the GPU when CUDA is usable; the
scikit-learn models have no GPU support and stay on the CPU.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from core import gpu
from core.base import Trainer, balanced_weights, no_report, stratified_sample

__all__ = ["Trainer", "MLTrainer", "no_report", "stratified_sample", "build_trainers"]


class MLTrainer(Trainer):
    """A scikit-learn style classifier built from a factory that takes the seed.

    ``max_rows`` caps the training rows. It is set for SVM, whose training time grows
    roughly with the square of the row count. With ``balance`` every model is fitted with
    balanced sample weights.
    """

    paradigm = "ML"

    def __init__(
        self, key, name, factory, seed, *, max_rows=None, balance=True
    ):  # pylint: disable=too-many-arguments
        super().__init__(balance)
        self.key = key
        self.name = name
        self.factory = factory
        self.seed = seed
        self.max_rows = max_rows

    def fit(self, split, report):
        features, labels = split.x_train, split.y_train
        if self.max_rows and len(labels) > self.max_rows:
            features, labels = stratified_sample(features, labels, self.max_rows, self.seed)
            self.details["Training rows"] = int(len(labels))
            self.details["Subsampled"] = f"from {len(split.y_train)} rows"
        model = self.factory(self.seed)
        weights = balanced_weights(labels) if self.balance else None
        model.fit(features, labels, sample_weight=weights)
        self.details["Class weights"] = "balanced" if self.balance else "none"
        cuda = model.get_params().get("device") == "cuda"
        self.details["Device"] = gpu.device_label(cuda)
        if cuda:
            # The test rows are NumPy arrays in host memory; predicting on the CPU avoids
            # XGBoost's device-mismatch warning on every call and is fast at this size.
            model.set_params(device="cpu")
        return model


def xgboost_classifier(seed):
    """XGBoost on the GPU when CUDA is usable (core/gpu.py), otherwise on every CPU core."""
    return XGBClassifier(
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if gpu.use_cuda() else "cpu",
    )


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
    "xgboost": ("XGBoost", xgboost_classifier),
}
DL_MODELS = ("dnn",)
QML_MODELS = ("vqc",)


def available_models():
    """Keys of every model, in the order they train: 6 ML, then DL, then QML."""
    return [*ML_MODELS, *DL_MODELS, *QML_MODELS]


def default_options():
    """Per-run model options and their defaults (REQ-MS-5)."""
    return {
        "dnn_epochs": config.DNN_EPOCHS,
        "qml_epochs": config.QML_EPOCHS,
        "qubits": config.QML_QUBITS,
        "balance_classes": True,
    }


#: Display name and paradigm of the models whose trainers load TensorFlow or PennyLane.
#: They must match DNNTrainer and QMLTrainer (checked in tests/test_trainers.py).
DEFERRED_MODELS = {
    "dnn": ("Deep Neural Network", "DL"),
    "vqc": ("Variational Quantum Classifier", "QML"),
}


class DeferredTrainer:
    """Stands in for the DNN or VQC trainer until something first uses it (DEF-04, PERF-1).

    Importing TensorFlow or PennyLane takes seconds. Building the real trainer on first
    attribute access moves that import from the /api/train request into the job thread,
    so the request returns 202 at once. ``key``, ``name`` and ``paradigm`` are known
    without the import; every other attribute comes from the real trainer.
    """

    def __init__(self, key, build):
        self.key = key
        self.name, self.paradigm = DEFERRED_MODELS[key]
        self._build = build
        self._trainer = None

    @property
    def loaded(self):
        """True once the real trainer has been built."""
        return self._trainer is not None

    def resolve(self):
        """The real trainer, built on first use."""
        if self._trainer is None:
            self._trainer = self._build()
        return self._trainer

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self.resolve(), attr)


def build_dnn(seed, options):
    """The DNN trainer. Imported here so a run without DL never loads TensorFlow."""
    from core.dnn import DNNTrainer  # pylint: disable=import-outside-toplevel

    return DNNTrainer(seed, epochs=options["dnn_epochs"], balance=options["balance_classes"])


def build_vqc(seed, options):
    """The VQC trainer. Imported here so a run without QML never loads PennyLane."""
    from core.qml import QMLTrainer  # pylint: disable=import-outside-toplevel

    return QMLTrainer(
        seed,
        epochs=options["qml_epochs"],
        qubits=options["qubits"],
        balance=options["balance_classes"],
    )


def build_one(key, seed, options):
    """The trainer for one known model key; DL and QML trainers are deferred."""
    if key in ML_MODELS:
        name, factory = ML_MODELS[key]
        max_rows = config.SVM_MAX_ROWS if key == "svm" else None
        return MLTrainer(
            key, name, factory, seed, max_rows=max_rows, balance=options["balance_classes"]
        )
    build = build_dnn if key in DL_MODELS else build_vqc
    return DeferredTrainer(key, lambda: build(seed, options))


def build_trainers(model_keys, seed, options=None):
    """Create trainers for the requested keys, in catalogue order.

    ``options`` overrides entries of ``default_options()``. Raises ValueError naming any
    key that is not a known model.
    """
    known = available_models()
    unknown = [key for key in model_keys if key not in known]
    if unknown:
        raise ValueError(f"Unknown or unavailable model(s): {', '.join(map(str, unknown))}.")
    settings = {**default_options(), **(options or {})}
    return [build_one(key, seed, settings) for key in known if key in model_keys]
