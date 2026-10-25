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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import config
from core.base import stratified_holdout
from core.profiling import column_role, is_numeric

ONE_HOT_MAX_UNIQUE = 20
# Share of the training rows held out as validation rows. They pick each binary model's
# decision threshold and stop the DNN early; the test rows are never used for either.
VALIDATION_RATIO = 0.2
# Labels that mean "nothing happened". In a binary target the other label is the positive
# class, so precision and recall describe fraud, churn or disease rather than its absence.
NEGATIVE_LABELS = frozenset(
    {
        "0",
        "0.0",
        "n",
        "no",
        "f",
        "false",
        "neg",
        "negative",
        "none",
        "normal",
        "legit",
        "legitimate",
        "genuine",
        "benign",
    }
)


class PreprocessingError(ValueError):
    """The data cannot be turned into a train/test split. The message is shown to the user."""


@dataclass
class ColumnPlan:
    """What happens to each non-target column (REQ-PP-2 to REQ-PP-5, REQ-PP-10)."""

    dropped: list = field(default_factory=list)
    numeric: list = field(default_factory=list)
    one_hot: list = field(default_factory=list)
    ordinal: list = field(default_factory=list)

    @property
    def features(self):
        """Columns that are kept, in transformer order."""
        return self.numeric + self.one_hot + self.ordinal

    def metadata(self):
        """Column lists for the results payload."""
        return {
            "dropped_columns": [str(name) for name in self.dropped],
            "numeric_columns": [str(name) for name in self.numeric],
            "one_hot_columns": [str(name) for name in self.one_hot],
            "ordinal_columns": [str(name) for name in self.ordinal],
        }


def split_fingerprint(y_test):
    """Short hash of the test-label bytes, so two runs can be checked for the same split."""
    data = np.ascontiguousarray(y_test, dtype=np.int64).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class Split:  # pylint: disable=too-many-instance-attributes
    """The shared train, validation and test arrays plus what was done to produce them
    (REQ-PP-10). The validation rows come from the training rows; models never train on
    them."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    classes: list
    feature_names: list
    plan: ColumnPlan = field(default_factory=ColumnPlan)
    x_val: np.ndarray = None
    y_val: np.ndarray = None

    @property
    def n_classes(self):
        """Number of target classes."""
        return len(self.classes)

    @property
    def positive_class(self):
        """The class precision, recall and F1 describe for a binary target (label 1)."""
        return str(self.classes[1]) if self.n_classes == 2 else None

    @property
    def fingerprint(self):
        """Hash of the test labels (REQ-PP-10)."""
        return split_fingerprint(self.y_test)

    def metadata(self):
        """JSON-safe description of the split for the results payload."""
        return {
            "train_rows": int(len(self.y_train)),
            "validation_rows": 0 if self.y_val is None else int(len(self.y_val)),
            "test_rows": int(len(self.y_test)),
            "features": int(self.x_train.shape[1]),
            "classes": [str(label) for label in self.classes],
            "positive_class": self.positive_class,
            "test_positive_rate": (
                float(np.mean(self.y_test == 1)) if self.n_classes == 2 else None
            ),
            **self.plan.metadata(),
            "split_fingerprint": self.fingerprint,
        }


def as_text(data, columns):
    """Cast categorical columns to strings, keeping missing values missing, so encoders
    never see mixed types such as 1 and "1"."""
    for name in columns:
        column = data[name]
        data[name] = column.where(column.isna(), column.astype(str))


def class_order(labels):
    """Target classes in label order. For a binary target the positive class comes second.

    The positive class is the label that is not a "nothing happened" value such as 0, No
    or legit; failing that, the rarer class. Sorting alone would make "legit" the positive
    class of a fraud/legit target and report how well models find legitimate rows.
    """
    counts = labels.value_counts()
    names = sorted(counts.index)
    if len(names) != 2:
        return names
    negatives = [name for name in names if name.strip().lower() in NEGATIVE_LABELS]
    if len(negatives) == 1:
        negative = negatives[0]
    elif counts[names[0]] != counts[names[1]]:
        negative = counts.idxmax()
    else:
        negative = names[0]
    return [negative] + [name for name in names if name != negative]


def to_dense(matrix):
    """Return ``matrix`` as a float64 array, or raise if it has NaN or inf (REQ-PP-9)."""
    dense = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(dense).all():
        raise PreprocessingError("Preprocessing produced missing or infinite values.")
    return dense


class Preprocessor:
    """Builds the column transformer and the stratified split for one job."""

    def __init__(self, seed=config.DEFAULT_SEED, test_ratio=config.DEFAULT_TEST_RATIO):
        self.seed = seed
        self.test_ratio = test_ratio

    @staticmethod
    def plan_columns(frame, target):
        """Decide what happens to each feature column (REQ-PP-2 to REQ-PP-5).

        The target is never dropped and never used as a feature.
        """
        plan = ColumnPlan()
        for name in frame.columns:
            if name == target:
                continue
            series = frame[name]
            if column_role(series) != "feature":
                plan.dropped.append(name)
            elif is_numeric(series):
                plan.numeric.append(name)
            elif series.nunique(dropna=True) <= ONE_HOT_MAX_UNIQUE:
                plan.one_hot.append(name)
            else:
                plan.ordinal.append(name)
        return plan

    @staticmethod
    def build_transformer(plan):
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
            ("numeric", numeric_steps, plan.numeric),
            ("one_hot", one_hot_steps, plan.one_hot),
            ("ordinal", ordinal_steps, plan.ordinal),
        ]
        return ColumnTransformer(
            [part for part in parts if part[2]], remainder="drop", sparse_threshold=0
        )

    def stratified_split(self, features, labels):
        """Stratified train/test split with the configured seed and ratio (REQ-PP-8)."""
        try:
            return train_test_split(
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

    def fit_split(self, frame, target):
        """Turn a raw table into a Split (REQ-PP-1 to REQ-PP-10).

        Rows with a missing target are removed, the target is label-encoded from its
        string form (binary: positive class = 1, see ``class_order``), the split is
        stratified with a fixed seed, VALIDATION_RATIO of the training rows are held out
        as validation rows, and the transformer is fitted on the remaining training rows
        only.
        """
        if target not in frame.columns:
            raise PreprocessingError(f"Target column '{target}' is not in the dataset.")

        data = frame[frame[target].notna()].copy()
        plan = self.plan_columns(data, target)
        if not plan.features:
            raise PreprocessingError(
                "No usable feature columns remain after dropping identifier and constant "
                "columns."
            )
        as_text(data, plan.one_hot + plan.ordinal)

        text_labels = data[target].astype(str)
        classes = class_order(text_labels)
        labels = text_labels.map({name: index for index, name in enumerate(classes)})
        x_train, x_test, y_train, y_test = self.stratified_split(
            data[plan.features], labels.to_numpy(dtype=np.int64)
        )
        x_train, x_val, y_train, y_val = stratified_holdout(
            x_train, y_train, VALIDATION_RATIO, self.seed
        )

        transformer = self.build_transformer(plan)
        return Split(
            x_train=to_dense(transformer.fit_transform(x_train)),
            x_test=to_dense(transformer.transform(x_test)),
            y_train=y_train,
            y_test=y_test,
            classes=classes,
            feature_names=[str(name) for name in transformer.get_feature_names_out()],
            plan=plan,
            x_val=to_dense(transformer.transform(x_val)),
            y_val=y_val,
        )
