"""Deep neural network trainer (SM-38, SM-39).

Implements REQ-TR-3 and REQ-TR-4: a Keras dense network on TensorFlow-CPU with batch
normalisation, dropout, Adam at 1e-3, early stopping (patience 10, best weights restored)
and a validation split taken from the training rows only.
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

import config
from core.base import Trainer

# TensorFlow's start-up banner and oneDNN notices would flood the server log.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# The environment variables above must be set before Keras loads TensorFlow.
import keras  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

VALIDATION_RATIO = 0.2
PATIENCE = 10
LEARNING_RATE = 1e-3


def build_network(n_features, n_classes):
    """The REQ-TR-3 architecture: Dense 128, BatchNorm, Dropout 0.3, Dense 64, Dropout 0.2,
    Dense 32, then a sigmoid (binary) or softmax (multi-class) output."""
    binary = n_classes == 2
    network = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(
                1 if binary else n_classes, activation="sigmoid" if binary else "softmax"
            ),
        ]
    )
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy" if binary else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return network


def validation_split(features, labels, seed):
    """Hold out 20% of the training rows for early stopping (REQ-TR-4).

    The held-out test rows are never touched. Stratified when every class has enough rows.
    """
    try:
        return train_test_split(
            features, labels, test_size=VALIDATION_RATIO, random_state=seed, stratify=labels
        )
    except ValueError:
        return train_test_split(features, labels, test_size=VALIDATION_RATIO, random_state=seed)


class DNNModel:
    """A trained network with the scikit-learn ``predict`` / ``predict_proba`` interface."""

    def __init__(self, network, n_classes):
        self.network = network
        self.n_classes = n_classes

    def predict_proba(self, features):
        """Class probabilities, shape ``(rows, n_classes)``."""
        output = self.network.predict(np.asarray(features, dtype=np.float32), verbose=0)
        if self.n_classes == 2:
            positive = output.ravel()
            return np.column_stack([1 - positive, positive])
        return output

    def predict(self, features):
        """The most probable class for each row."""
        return np.argmax(self.predict_proba(features), axis=1)


class ProgressCallback(keras.callbacks.Callback):
    """Reports each finished epoch to the job, which also lets a cancel stop training."""

    def __init__(self, report, epochs):
        super().__init__()
        self.report = report
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_loss")
        note = f", validation loss {loss:.3f}" if loss is not None else ""
        self.report((epoch + 1) / self.epochs, f"epoch {epoch + 1} of up to {self.epochs}{note}")


class DNNTrainer(Trainer):
    """The single deep learning model (REQ-MS-1)."""

    key = "dnn"
    name = "Deep Neural Network"
    paradigm = "DL"

    def __init__(self, seed, epochs=config.DNN_EPOCHS, batch_size=config.DNN_BATCH_SIZE):
        super().__init__()
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, split, report):
        keras.utils.set_random_seed(self.seed)
        features = np.asarray(split.x_train, dtype=np.float32)
        x_fit, x_val, y_fit, y_val = validation_split(features, split.y_train, self.seed)

        network = build_network(features.shape[1], split.n_classes)
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
        history = network.fit(
            x_fit,
            y_fit,
            validation_data=(x_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[stopper, ProgressCallback(report, self.epochs)],
            verbose=0,
        )
        epochs_run = len(history.history["loss"])
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        self.details.update(
            {
                "Epochs run": epochs_run,
                "Best epoch": best_epoch,
                "Stopped early": epochs_run < self.epochs,
                "Validation rows": int(len(y_val)),
                "Parameters": int(network.count_params()),
            }
        )
        return DNNModel(network, split.n_classes)
