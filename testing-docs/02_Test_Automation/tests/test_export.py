"""Group H: CSV export (ST-44, ST-45)."""

import csv
import io
import re
from pathlib import Path

import pytest

from pages import MODEL_KEYS, App, td, text

HEADER = [
    "Model",
    "Paradigm",
    "Status",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "Threshold",
    "Train time (s)",
    "Test time (s)",
    "Test rows",
    "Error",
]


def export(app, evidence):
    page = app.page
    with page.expect_download() as info:
        page.click("#export-csv")
    toast = app.wait_toast("Downloading the metrics table as CSV.")
    download = info.value
    evidence.folder.mkdir(parents=True, exist_ok=True)
    saved = evidence.folder / download.suggested_filename
    download.save_as(saved)
    raw = saved.read_bytes()
    return download.suggested_filename, raw, toast


@pytest.mark.st("ST-44")
def test_st44_export_metrics_table_as_csv(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    app.train(MODEL_KEYS)
    name, raw, toast = export(app, evidence)
    evidence.shot(page, "export-toast", full_page=False)
    assert toast == "Downloading the metrics table as CSV."
    assert name == "banking_model_comparison.csv"
    content = raw.decode("utf-8")  # must be valid UTF-8 for spreadsheet import
    rows = list(csv.reader(io.StringIO(content)))
    assert rows[0] == HEADER
    data = rows[1:]
    assert len(data) == 8
    assert all(len(r) == len(HEADER) for r in data), "rows have different column counts"
    _, table = app.table()
    shown = {r["Model"]: r for r in table}
    for row in data:
        record = dict(zip(HEADER, row))
        on_screen = shown[record["Model"]]
        assert record["Status"] == "done"
        # The screen rounds to its displayed precision; allow half a last digit.
        for column in ("Accuracy", "Precision", "Recall", "F1"):
            diff = abs(float(record[column]) * 100 - float(on_screen[column].rstrip("%")))
            assert diff <= 0.051, (record["Model"], column, record[column], on_screen[column])
        for column in ("ROC-AUC", "PR-AUC", "Threshold"):
            diff = abs(float(record[column]) - float(on_screen[column]))
            assert diff <= 0.00051, (record["Model"], column, record[column], on_screen[column])
    evidence.note(f"Saved {name} ({len(raw)} bytes) in the evidence folder.")
    evidence.note(
        "Spreadsheet check automated as: valid UTF-8, comma-separated, equal column counts."
    )


@pytest.mark.partial
@pytest.mark.st("ST-45")
def test_st45_export_safe_for_formula_like_dataset_name(page, evidence):
    app = App(page).open()
    app.upload(td("TD-13_=SUM(1,1).csv"))
    shown = text(page.locator("#dataset-name"))
    evidence.shot(page, "sanitised-name", full_page=False)
    assert not set("=(,)") & set(shown), f"dataset name not sanitised: {shown}"
    app.to_configure()
    app.select_target("Exited")
    app.train(["logistic_regression"])
    name, raw, _ = export(app, evidence)
    evidence.note(f"Dataset name on screen {shown!r}; download name {name!r}.")
    assert re.fullmatch(r"[A-Za-z0-9_-]+\.csv", name), name
    for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
        for cell in row:
            if cell and cell[0] in "=+-@":
                raise AssertionError(f"unescaped formula-like cell: {cell!r}")
    assert shown == "TD-13_SUM11.csv", f"sanitised name {shown!r}"
    excel = excel_formula_check(evidence.folder / name)
    evidence.note(f"Excel check: {excel}")
    assert excel.get("opened"), f"Excel could not open the file: {excel}"
    assert excel["formulas"] == 0, f"Excel evaluated {excel['formulas']} cell(s) as formulas"
    assert excel["errors"] == 0, f"Excel shows {excel['errors']} error value(s)"


EXCEL_CHECK = r"""
$path = $args[0]
$xl = New-Object -ComObject Excel.Application; $xl.Visible = $false; $xl.DisplayAlerts = $false
try {
  $wb = $xl.Workbooks.Open($path, 0, $true)
  $cells = $wb.Worksheets.Item(1).UsedRange.Cells
  $formulas = 0; $errors = 0; $count = 0
  foreach ($c in $cells) { $count++; if ($c.HasFormula) { $formulas++ }; if ($c.Text -like '#*') { $errors++ } }
  "{""opened"": true, ""cells"": $count, ""formulas"": $formulas, ""errors"": $errors}"
  $wb.Close($false)
} catch { "{""opened"": false, ""error"": ""$($_.Exception.Message -replace '""','')""}" } finally { $xl.Quit() }
"""


def excel_formula_check(path):
    """Open the CSV in Microsoft Excel (read-only, hidden) and count formula and error cells."""
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "excel_check.ps1"
        script.write_text(EXCEL_CHECK, encoding="utf-8")
        run = subprocess.run(  # noqa: S603 - fixed local script
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    line = (run.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"opened": False, "error": run.stdout + run.stderr}
