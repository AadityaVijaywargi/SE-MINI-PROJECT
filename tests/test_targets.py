"""Target suggestion and task analysis (SM-21, SM-22, SM-24, SM-26)."""

import pandas as pd
import pytest

from core.targets import analyse_task, name_tokens, suggest_targets


def test_domain_samples_suggest_their_target_first(domain_sample):
    _, frame, target = domain_sample
    assert suggest_targets(frame)[0]["column"] == target


def test_three_class_suggests_segment(three_class):
    assert suggest_targets(three_class)[0]["column"] == "Segment"


def test_only_2_to_20_classes_are_candidates():
    frame = pd.DataFrame(
        {"one": [1] * 30, "many": range(30), "few": [0, 1, 2] * 10, "label": ["a", "b"] * 15}
    )
    columns = [item["column"] for item in suggest_targets(frame)]
    assert set(columns) == {"few", "label"}


def test_scoring_rules():
    frame = pd.DataFrame({"is_fraud": [0, 1] * 10, "colour": ["r", "g"] * 10, "x": [1, 2] * 10})
    scores = {item["column"]: item["score"] for item in suggest_targets(frame)}
    assert scores["is_fraud"] == 50 + 40
    assert scores["colour"] == 50 + 10
    assert scores["x"] == 50 + 20


@pytest.mark.parametrize(
    "name, expected",
    [("isFraud", {"is", "fraud"}), ("Churn Label", {"churn", "label"}), ("y", {"y"})],
)
def test_name_tokens(name, expected):
    assert name_tokens(name) == expected


def test_keyword_must_be_a_whole_word():
    assert not name_tokens("yearly") & {"y"}


def test_binary_task():
    analysis = analyse_task(pd.Series(["Yes", "No"] * 20, name="Churn"))
    assert analysis["task"] == "binary"
    assert analysis["distribution"] == {"Yes": 20, "No": 20}
    assert analysis["trainable"] and analysis["warning"] is None


def test_multiclass_task(three_class):
    analysis = analyse_task(three_class["Segment"])
    assert analysis["task"] == "multiclass"
    assert analysis["n_classes"] == 3


def test_imbalance_warning():
    analysis = analyse_task(pd.Series([0] * 40 + [1] * 10, name="t"))
    assert analysis["imbalance_ratio"] == 4.0
    assert "F1" in str(analysis["warning"])
    assert analysis["trainable"]


def test_regression_target_blocked():
    analysis = analyse_task(pd.Series(range(100), name="price"))
    assert analysis["task"] == "unsupported"
    assert not analysis["trainable"]
    assert "regression" in str(analysis["reason"])


def test_single_class_blocked():
    assert not analyse_task(pd.Series(["a"] * 50, name="t"))["trainable"]


def test_class_with_one_row_blocked():
    analysis = analyse_task(pd.Series(["a"] * 49 + ["b"], name="t"))
    assert not analysis["trainable"]
    assert "stratified" in str(analysis["reason"])
