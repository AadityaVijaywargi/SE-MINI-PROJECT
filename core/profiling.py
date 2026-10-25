"""Column profiling and the upload preview (SM-17, SM-18).

Implements REQ-UP-5 to REQ-UP-7.
"""

import json
import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

IDENTIFIER_UNIQUE_RATIO = 0.95
ID_NAME = re.compile(r"(^|[^a-z])id([^a-z]|$)|[a-z]Id($|[^a-z])|^ID|ID$")
PREVIEW_ROWS = 10
SAMPLE_VALUES = 3


def is_numeric(series):
    """True for numeric columns. Booleans count as categorical."""
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def column_role(series):
    """Classify a column as 'constant', 'identifier' or 'feature' (REQ-UP-5).

    Constant: at most one distinct non-null value. Identifier: a unique ratio above 0.95
    and either non-numeric, or whole numbers in a column named like an ID (TransactionID,
    customer_id). A numeric ID is just a row counter and would otherwise be a feature.
    Everything else is a feature.
    """
    unique = series.nunique(dropna=True)
    if unique <= 1:
        return "constant"
    present = int(series.notna().sum())
    if unique / max(present, 1) > IDENTIFIER_UNIQUE_RATIO:
        if not is_numeric(series):
            return "identifier"
        if is_integer_dtype(series) and ID_NAME.search(str(series.name)):
            return "identifier"
    return "feature"


def to_json_value(value):
    """Convert a pandas/numpy scalar to a JSON-safe Python value; missing becomes None."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def profile_column(series):
    """Return the per-column profile dict required by REQ-UP-6."""
    missing = int(series.isna().sum())
    samples = series.dropna().unique()[:SAMPLE_VALUES]
    return {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "kind": "numeric" if is_numeric(series) else "categorical",
        "unique": int(series.nunique(dropna=True)),
        "missing": missing,
        "missing_pct": round(100 * missing / max(len(series), 1), 1),
        "role": column_role(series),
        "samples": [to_json_value(value) for value in samples],
    }


def profile_dataframe(frame):
    """Profile a whole table: totals plus one entry per column (REQ-UP-6, REQ-UP-8)."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "column_profiles": [profile_column(frame[name]) for name in frame.columns],
    }


def preview(frame, rows=PREVIEW_ROWS):
    """First ``rows`` rows as ``{columns, rows}`` with missing values blanked (REQ-UP-7)."""
    head = frame.head(rows)
    values = json.loads(head.to_json(orient="values", date_format="iso"))
    blanked = [["" if cell is None else cell for cell in row] for row in values]
    return {"columns": [str(name) for name in head.columns], "rows": blanked}
