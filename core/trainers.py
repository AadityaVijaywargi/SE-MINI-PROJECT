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
from core.base import Trainer, no_report, stratified_sample

__all__ = ["Trainer", "MLTrainer", "no_report", "stratified_sample", "build_trainers"]


class MLTrainer(Trainer):
    """A scikit-learn style classifier built from a factory that takes the seed.

    ``max_rows`` caps the training rows. It is set for SVM, whose training time grows
    roughly with the square of the row count.
    """

    paradigm = "ML"

    def __init__(
        self, key, name, factory, seed, *, max_rows=None
    ):  # pylint: disable=too-many-arguments
        super().__init__()
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
        model.fit(features, labels)
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
    }


def build_one(key, seed, options):
    """The trainer for one known model key."""
    if key in ML_MODELS:
        name, factory = ML_MODELS[key]
        max_rows = config.SVM_MAX_ROWS if key == "svm" else None
        return MLTrainer(key, name, factory, seed, max_rows=max_rows)
    # Imported here so a run without DL or QML never loads TensorFlow or PennyLane.
    if key in DL_MODELS:
        from core.dnn import DNNTrainer  # pylint: disable=import-outside-toplevel

        return DNNTrainer(seed, epochs=options["dnn_epochs"])
    from core.qml import QMLTrainer  # pylint: disable=import-outside-toplevel

    return QMLTrainer(seed, epochs=options["qml_epochs"], qubits=options["qubits"])


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
