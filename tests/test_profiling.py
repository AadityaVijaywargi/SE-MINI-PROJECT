"""Column profiling and preview (SM-17, SM-18)."""

import json

import numpy as np
import pandas as pd

from core.profiling import column_role, preview, profile_dataframe


def test_roles():
    assert column_role(pd.Series(["x"] * 10)) == "constant"
    assert column_role(pd.Series([np.nan] * 10)) == "constant"
    assert column_role(pd.Series([f"id{i}" for i in range(100)])) == "identifier"
    assert column_role(pd.Series(range(100))) == "feature"  # numeric is never an identifier
    assert column_role(pd.Series(["a", "b"] * 50)) == "feature"


def test_banking_profile(banking):
    profile = profile_dataframe(banking)
    columns = {column["name"]: column for column in profile["column_profiles"]}
    assert profile["rows"] == 200
    assert profile["columns"] == banking.shape[1]
    assert columns["CustomerId"]["role"] == "identifier"
    assert columns["Balance"]["kind"] == "numeric"
    assert columns["Geography"]["kind"] == "categorical"
    assert columns["Balance"]["missing"] == 30
    assert columns["Balance"]["missing_pct"] == 15.0
    assert profile["missing_cells"] == 45
    assert len(columns["Geography"]["samples"]) <= 3


def test_profile_is_json_serialisable(domain_sample):
    _, frame, _ = domain_sample
    json.dumps(profile_dataframe(frame))


def test_preview_blanks_missing_values(banking):
    result = preview(banking)
    assert len(result["rows"]) == 10
    assert result["columns"] == list(banking.columns)
    assert all(cell is not None for row in result["rows"] for cell in row)
