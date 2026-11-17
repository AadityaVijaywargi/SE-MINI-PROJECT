"""Copy the blank Test_Execution_Record_Template.xlsx and fill it from one run's results.csv.

Run from testing-docs/02_Test_Automation with the application's virtual environment:

    ..\\..\\.venv\\Scripts\\python.exe fill_execution_record.py ..\\04_Test_Execution_Results\\<run-folder>

Rules (no result is invented):
* Status PASS only when pytest passed and no manual review is pending.
* Status FAIL when pytest failed or errored.
* Status UNVERIFIED (red), with a remark, when pytest passed but a person must still check
  something; blank with a remark for a case the script did not run (skipped).
The blank template is not changed; the filled copy is written into the run folder.
Needs openpyxl (use the application's .venv).
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "04_Test_Execution_Results" / "Execution_Records" / "Test_Execution_Record_Template.xlsx"


def main(run_dir, out=None):
    run = Path(run_dir).resolve()
    rows = {r["test_case_id"]: r for r in csv.DictReader((run / "results.csv").open(encoding="utf-8"))}
    info = dict(
        line.split(":", 1) for line in (run / "run-info.txt").read_text(encoding="utf-8-sig").splitlines() if ":" in line
    )
    info = {k.strip(): v.strip() for k, v in info.items()}
    commit = info.get("Commit", "")
    started = info.get("Run started", "")
    run_date = datetime.strptime(started, "%Y-%m-%d %H:%M:%S") if started else None

    wb = load_workbook(TEMPLATE)
    ri = wb["Run Information"]
    values = {
        "Test cycle / run ID": run.name,
        "Execution start date": started,
        "Execution end date": info.get("Run finished", "").split(" (")[0],
        "Executed by": "Automated: Playwright scripts (run_tests.ps1); results reviewed by Ravi Teja (Ravi S) [FILL LATER]",
        "Build under test (commit SHA)": commit,
        "Application URL": "http://127.0.0.1:5055 (local, started by the tests)",
        "COMPUTE_DEVICE setting": "cpu",
        "Operating system": info.get("Machine / OS", ""),
        "Browsers and versions": f"{info.get('Default browser', '')}; {info.get('Also used', '')}",
        "Playwright version": info.get("Packages", ""),
        "Python version": info.get("Python (tests)", ""),
        "Evidence folder": str(run),
        "Notes": "Filled by fill_execution_record.py from results.csv. Blank Status = manual review or manual execution still required.",
    }
    for r in range(4, 4 + 14):
        label = ri.cell(row=r, column=1).value
        if label in values:
            ri.cell(row=r, column=2).value = values[label]

    ex = wb["Execution"]
    for r in range(5, 55):
        case = ex.cell(row=r, column=1).value
        row = rows.get(case)
        if not row:
            continue
        outcome, pending, message = row["pytest_outcome"], row["manual_review_pending"], row["message"]
        if outcome == "passed" and not pending:
            status, remark = "PASS", "All automated checks passed."
        elif outcome in ("failed", "error"):
            status, remark = "FAIL", f"Automated check failed: {message}"
        elif outcome == "passed":
            status, remark = "UNVERIFIED", f"Automated checks passed; manual review pending: {pending}"
        else:
            status, remark = None, f"Not run by the script: {message}"
        if outcome != "skipped":
            ex.cell(row=r, column=7).value = run_date
            ex.cell(row=r, column=7).number_format = "DD-MMM-YYYY"
            ex.cell(row=r, column=8).value = "Playwright (automated)"
            ex.cell(row=r, column=9).value = "Chrome (channel chrome)" if case not in ("ST-39", "ST-49") else "Chrome + Edge" + (" + Playwright Chromium" if case == "ST-49" else "")
            ex.cell(row=r, column=10).value = commit
            ex.cell(row=r, column=11).value = remark if outcome != "passed" or pending else "As expected (see report.html)."
            ex.cell(row=r, column=14).value = f"{run.name}/report.html; screenshots/{row['test'].split('::')[-1]}"
        ex.cell(row=r, column=12).value = status
        ex.cell(row=r, column=17).value = remark
    out = Path(out) if out else run / f"Test_Execution_Record_{run.name}.xlsx"
    wb.save(out)
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
