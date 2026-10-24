"""CSV export with formula-injection escaping (SM-65, TC-12)."""

import csv
import io

import pytest

from core.export import escape_cell, results_csv


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('=HYPERLINK("http://x")', '\'=HYPERLINK("http://x")'),
        ("+1+1", "'+1+1"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1:A9)", "'@SUM(A1:A9)"),
        ("\tTAB", "'\tTAB"),
        ("\rCR", "'\rCR"),
        ("Random Forest", "Random Forest"),
        ("a=b", "a=b"),
        (None, ""),
        (0.123456789, 0.123457),
        (40, 40),
    ],
)
def test_escape_cell(value, expected):
    assert escape_cell(value) == expected


def test_export_escapes_injected_text_and_keeps_numbers():
    results = [
        {
            "model": "SVM",
            "paradigm": "ML",
            "status": "done",
            "accuracy": 0.9,
            "f1": 0.8,
            "roc_auc": None,
            "train_time": 0.01,
            "test_rows": 40,
        },
        {
            "model": "XGBoost",
            "paradigm": "ML",
            "status": "failed",
            "error": "=cmd|' /C calc'!A0 in column '@Balance'",
        },
    ]
    rows = list(csv.reader(io.StringIO(results_csv(results))))
    header, done, failed = rows
    assert header[0] == "Model" and header[-1] == "Error"
    assert done[3] == "0.9" and done[7] == ""
    assert failed[-1].startswith("'=")
    assert not any(cell.startswith(("=", "+", "-", "@")) for row in rows for cell in row)


def test_finished_models_come_first():
    results = [
        {"model": "Broken", "paradigm": "DL", "status": "failed", "error": "boom"},
        {"model": "NB", "paradigm": "ML", "status": "done", "accuracy": 1.0},
    ]
    rows = list(csv.reader(io.StringIO(results_csv(results))))
    assert [row[0] for row in rows[1:]] == ["NB", "Broken"]
