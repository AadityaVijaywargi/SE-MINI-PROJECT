"""Shared preprocessing and the train/test split (SM-31 to SM-35).

Implements REQ-PP-1 to REQ-PP-10. One Preprocessor.fit_split call per job produces the
arrays every model trains and is tested on, so differences in results come from the
learning paradigm, not from data handling.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

import config
from core.profiling import column_role, is_numeric

ONE_HOT_MAX_UNIQUE = 20


class PreprocessingError(ValueError):
    """The data cannot be turned into a train/test split. The message is shown to the user."""


@dataclass
class Split:
    """The shared train/test arrays plus what was done to produce them (REQ-PP-10)."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    classes: list
    feature_names: list
    dropped: list = field(default_factory=list)
    numeric: list = field(default_factory=list)
    one_hot: list = field(default_factory=list)
    ordinal: list = field(default_factory=list)
    fingerprint: str = ""

    @property
    def n_classes(self):
        """Number of target classes."""
        return len(self.classes)

    def metadata(self):
        """JSON-safe description of the split for the results payload."""
        return {
            "train_rows": int(len(self.y_train)),
            "test_rows": int(len(self.y_test)),
            "features": int(self.X_train.shape[1]),
            "classes": [str(label) for label in self.classes],
            "dropped_columns": self.dropped,
            "numeric_columns": self.numeric,
            "one_hot_columns": self.one_hot,
            "ordinal_columns": self.ordinal,
            "split_fingerprint": self.fingerprint,
        }


def split_fingerprint(y_test):
    """Short hash of the test-label bytes, so two runs can be checked for the same split."""
    data = np.ascontiguousarray(y_test, dtype=np.int64).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


class Preprocessor:
    """Builds the column transformer and the stratified split for one job."""

    def __init__(self, seed=config.DEFAULT_SEED, test_ratio=config.DEFAULT_TEST_RATIO):
        self.seed = seed
        self.test_ratio = test_ratio

    def plan_columns(self, frame, target):
        """Decide what happens to each feature column (REQ-PP-2 to REQ-PP-5).

        Returns ``(dropped, numeric, one_hot, ordinal)`` lists of column names. The target
        is never dropped and never used as a feature.
        """
        dropped, numeric, one_hot, ordinal = [], [], [], []
        for name in frame.columns:
            if name == target:
                continue
            series = frame[name]
            if column_role(series) != "feature":
                dropped.append(name)
            elif is_numeric(series):
                numeric.append(name)
            elif series.nunique(dropna=True) <= ONE_HOT_MAX_UNIQUE:
                one_hot.append(name)
            else:
                ordinal.append(name)
        return dropped, numeric, one_hot, ordinal

    @staticmethod
    def build_transformer(numeric, one_hot, ordinal):
        """Column transformer: median+scale, mode+one-hot, mode+ordinal+scale."""
        numeric_steps = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        one_hot_steps = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        ordinal_steps = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                ("scale", StandardScaler()),
            ]
        )
        parts = [
            ("numeric", numeric_steps, numeric),
            ("one_hot", one_hot_steps, one_hot),
            ("ordinal", ordinal_steps, ordinal),
        ]
        return ColumnTransformer(
            [part for part in parts if part[2]], remainder="drop", sparse_threshold=0
        )

    def fit_split(self, frame, target):
        """Turn a raw table into a Split (REQ-PP-1 to REQ-PP-10).

        Rows with a missing target are removed, the target is label-encoded from its
        string form, the split is stratified with a fixed seed, and the transformer is
        fitted on the training rows only.
        """
        if target not in frame.columns:
            raise PreprocessingError(f"Target column '{target}' is not in the dataset.")

        data = frame[frame[target].notna()].copy()
        dropped, numeric, one_hot, ordinal = self.plan_columns(data, target)
        if not (numeric or one_hot or ordinal):
            raise PreprocessingError(
                "No usable feature columns remain after dropping identifier and constant "
                "columns."
            )
        for name in one_hot + ordinal:
            column = data[name]
            data[name] = column.where(column.isna(), column.astype(str))

        encoder = LabelEncoder()
        labels = encoder.fit_transform(data[target].astype(str))
        features = data[numeric + one_hot + ordinal]

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.test_ratio,
                random_state=self.seed,
                stratify=labels,
            )
        except ValueError as exc:
            raise PreprocessingError(
                "A stratified split is impossible: every class needs at least two rows and "
                "the test set needs room for each class."
            ) from exc

        transformer = self.build_transformer(numeric, one_hot, ordinal)
        train_matrix = np.asarray(transformer.fit_transform(x_train), dtype=np.float64)
        test_matrix = np.asarray(transformer.transform(x_test), dtype=np.float64)
        if not (np.isfinite(train_matrix).all() and np.isfinite(test_matrix).all()):
            raise PreprocessingError("Preprocessing produced missing or infinite values.")

        return Split(
            X_train=train_matrix,
            X_test=test_matrix,
            y_train=y_train,
            y_test=y_test,
            classes=list(encoder.classes_),
            feature_names=[str(name) for name in transformer.get_feature_names_out()],
            dropped=[str(name) for name in dropped],
            numeric=[str(name) for name in numeric],
            one_hot=[str(name) for name in one_hot],
            ordinal=[str(name) for name in ordinal],
            fingerprint=split_fingerprint(y_test),
        )
