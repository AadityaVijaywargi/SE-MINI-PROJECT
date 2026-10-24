"""CSV export of the metrics table (SM-65).

Implements REQ-RS-9 and SEC-2: any text cell that starts with =, +, - or @ (or a tab or
carriage return) is prefixed with an apostrophe, so a spreadsheet shows it as text instead
of running it as a formula. Model names are fixed, but error messages and the dataset name
can carry text from the uploaded file.
"""

import csv
import io

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
COLUMNS = (
    ("Model", "model"),
    ("Paradigm", "paradigm"),
    ("Status", "status"),
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("ROC-AUC", "roc_auc"),
    ("Train time (s)", "train_time"),
    ("Test time (s)", "test_time"),
    ("Test rows", "test_rows"),
    ("Error", "error"),
)


def escape_cell(value):
    """A CSV-safe cell: numbers unchanged, None empty, formula-like text neutralised."""
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (int, bool)):
        return value
    text = str(value)
    return f"'{text}" if text.startswith(FORMULA_PREFIXES) else text


def results_csv(results):
    """The metrics table as CSV text, one row per model, finished models first."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([label for label, _ in COLUMNS])
    ordered = sorted(results, key=lambda result: result.get("status") != "done")
    for result in ordered:
        writer.writerow([escape_cell(result.get(key)) for _, key in COLUMNS])
    return buffer.getvalue()
