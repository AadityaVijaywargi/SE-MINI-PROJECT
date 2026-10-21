"""Shared preprocessing and split (SM-31 to SM-35, SM-88)."""

import numpy as np
import pandas as pd
import pytest

from core.preprocessing import Preprocessor, PreprocessingError


def test_dense_finite_matrix_for_every_domain(domain_sample):
    _, frame, target = domain_sample
    split = Preprocessor().fit_split(frame, target)
    for matrix in (split.X_train, split.X_test):
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float64
        assert np.isfinite(matrix).all()
    assert split.X_train.shape[1] == split.X_test.shape[1] == len(split.feature_names)


def test_identifier_and_constant_dropped_target_kept(banking):
    frame = banking.assign(Bank="Same bank")
    split = Preprocessor().fit_split(frame, "Exited")
    assert set(split.dropped) == {"CustomerId", "Bank"}
    assert "Exited" not in split.numeric + split.one_hot + split.ordinal


def test_column_plan_telecom():
    from tests.conftest import load_sample

    split = Preprocessor().fit_split(load_sample("telecom"), "Churn")
    assert "City" in split.ordinal  # 60 categories > 20
    assert "Contract" in split.one_hot
    assert "Tenure" in split.numeric


def test_string_labels_encoded():
    from tests.conftest import load_sample

    split = Preprocessor().fit_split(load_sample("healthcare"), "Diagnosis")
    assert split.classes == ["Negative", "Positive"]
    assert set(np.unique(split.y_train)) == {0, 1}


def test_split_is_stratified_and_sized(banking):
    split = Preprocessor(test_ratio=0.2).fit_split(banking, "Exited")
    assert len(split.y_test) == 40
    train_share = np.bincount(split.y_train) / len(split.y_train)
    test_share = np.bincount(split.y_test) / len(split.y_test)
    assert np.allclose(train_share, test_share, atol=0.05)


def test_same_seed_same_split_other_seed_differs(banking):
    first = Preprocessor(seed=42).fit_split(banking, "Exited")
    again = Preprocessor(seed=42).fit_split(banking, "Exited")
    other = Preprocessor(seed=7).fit_split(banking, "Exited")
    assert first.fingerprint == again.fingerprint
    assert np.array_equal(first.X_test, again.X_test)
    assert not np.array_equal(first.X_test, other.X_test)


def test_scaler_fitted_on_training_rows_only(banking):
    split = Preprocessor().fit_split(banking, "Exited")
    age = split.feature_names.index("numeric__Age")
    assert abs(split.X_train[:, age].mean()) < 1e-9
    assert abs(split.X_test[:, age].mean()) > 1e-6


def test_rows_with_missing_target_removed(banking):
    frame = banking.copy()
    frame.loc[:9, "Exited"] = np.nan
    split = Preprocessor().fit_split(frame, "Exited")
    assert len(split.y_train) + len(split.y_test) == 190


def test_unseen_category_in_test_does_not_break():
    frame = pd.DataFrame(
        {
            "colour": ["red", "blue"] * 30 + ["green"],
            "size": np.arange(61, dtype=float),
            "label": ["a", "b"] * 30 + ["a"],
        }
    )
    split = Preprocessor().fit_split(frame, "label")
    assert np.isfinite(split.X_test).all()


def test_metadata_records_columns_and_fingerprint(banking):
    meta = Preprocessor().fit_split(banking, "Exited").metadata()
    assert meta["dropped_columns"] == ["CustomerId"]
    assert len(meta["split_fingerprint"]) == 16
    assert meta["train_rows"] + meta["test_rows"] == 200


def test_unknown_target_raises(banking):
    with pytest.raises(PreprocessingError):
        Preprocessor().fit_split(banking, "Nope")


def test_no_features_left_raises():
    frame = pd.DataFrame({"id": [f"r{i}" for i in range(60)], "label": ["a", "b"] * 30})
    with pytest.raises(PreprocessingError, match="No usable feature"):
        Preprocessor().fit_split(frame, "label")


def test_preprocessing_runs_once_per_job(banking, monkeypatch):
    from core import jobs
    from core.trainers import build_trainers

    calls = []
    original = jobs.Preprocessor.fit_split

    def counting(self, frame, target):
        calls.append(target)
        return original(self, frame, target)

    monkeypatch.setattr(jobs.Preprocessor, "fit_split", counting)
    trainers = build_trainers(["logistic_regression", "decision_tree", "naive_bayes"], 42)
    job = jobs.Job(trainers)
    jobs.JobRunner(job, banking, "Exited", trainers, {"seed": 42, "test_ratio": 0.2}).run()
    assert job.status == "complete"
    assert calls == ["Exited"]
