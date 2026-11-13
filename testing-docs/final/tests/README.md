# System test scripts (Playwright for Python)

Black-box browser and HTTP tests for the 50 system test cases in
`../System_Test_Cases.xlsx`. One test function per case (`test_st01_…` to `test_st50_…`),
marked with `@pytest.mark.st("ST-nn")`. Plan: STP FSPP-MTP02.0, sections 8.8 to 8.11.

## Rules the scripts follow

- Black box: only page elements, visible text and HTTP. No application file is changed.
- Local only: the tests start their own copy of the app at `http://127.0.0.1:5055`
  (`COMPUTE_DEVICE=cpu`, uploads in the run folder). Any non-local URL is refused.
- Each test gets a fresh browser context, so it has its own `pp_session` cookie.
- Results come only from pytest. A case is never marked PASS by hand in this folder.

## Setup (once)

```powershell
cd testing-docs\final\tests
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-e2e.txt
.venv\Scripts\python.exe -m playwright install chromium
```

The application's own `.venv` (repository root) must exist, because the tests start the
app with it. Chrome and Edge must be installed (default browser channel: `chrome`).

## Run

```powershell
.\run_tests.ps1               # all 50 cases
.\run_tests.ps1 -SkipSlow     # skip ST-31, ST-34, ST-50
.\run_tests.ps1 -Select st07  # one case
.\run_tests.ps1 -Headed       # watch the browser
```

Each run writes `runs\<date-time>-<commit>\`:

| File | Content |
|---|---|
| `report.html` | pytest-html report with notes, manual-review items and step screenshots |
| `results.csv` | one row per case: pytest outcome, duration, message, manual review pending |
| `junit.xml` | machine-readable results |
| `screenshots\<test>\` | step screenshots |
| `playwright\` | trace and screenshot of every failed test (`playwright show-trace <zip>`) |
| `app-server.log` | console log of the application under test |
| `run-info.txt` | commit, browsers, package versions, start and end time |

`runs\dev-check\` holds the framework debugging runs made while writing the scripts; it is
not execution evidence.

## Files

| File | Cases |
|---|---|
| `test_upload.py` | ST-01 to ST-08 |
| `test_dataset.py` | ST-09 to ST-12 |
| `test_configure.py` | ST-13 to ST-19 |
| `test_advanced_settings.py` | ST-20 to ST-25 |
| `test_training.py` | ST-26 to ST-35 (ST-35 restarts the app with COMPUTE_DEVICE=auto, then cpu) |
| `test_progress.py` | ST-36 to ST-39 |
| `test_results.py` | ST-40 to ST-43 |
| `test_export.py` | ST-44, ST-45 |
| `test_errors.py` | ST-46, ST-47 |
| `test_ui.py` | ST-48, ST-49 |
| `test_performance.py` | ST-50 (runs the existing `tools/measure_times.py`) |
| `conftest.py`, `pages.py`, `settings.py` | server fixture, evidence capture, page helpers |

## How the harder cases are automated

All 50 cases are scripted. When a check still needs a person, it is listed under
"Manual review required" in the report and in the `manual_review_pending` column of
`results.csv`, and `fill_execution_record.py` marks the case **UNVERIFIED** (never PASS).

| Case | Measurable criterion used |
|---|---|
| ST-29, ST-30, ST-36 | stage labels sampled every 0.2 s; a TD-12 confirmation run when the Banking run is shorter than the 2 s UI refresh |
| ST-34 | warning text and SVM training rows; a 25,000-row observation for SRS REQ-MS-4 [DEV CONFIRM] |
| ST-35 | the fixture restarts the app with `COMPUTE_DEVICE=auto`, then `cpu`; `nvidia-smi` recorded |
| ST-42 | Chart.js dataset colours, labels, dashed baselines, axis types, legends |
| ST-45 | the CSV is opened in Microsoft Excel through COM: formula and error cells counted |
| ST-48 | every focus stop: outline at least 2 px and contrast at least 3:1 |
| ST-05, ST-46, ST-47 | 101 MB file generated at run time; live-server API checks; real server stop and restart |

Run 2 (`runs\20261110-132648-4988037\`, original build) left one check open: ST-36, where the
'Testing <model> (i of n)' label was never rendered on screen. Regression run 4
(`runs\20261110-225957-4988037-modified\`, build with the DEF-01 to DEF-04 fixes) passed all 50
cases with no manual check open; the ST-36 label depends on timing (see
`..\Manual_Verification_Guide.md`). A folder name ending in `-modified` means the run used
uncommitted changes to tracked files.

## Fill the execution record from a run

```powershell
..\..\..\.venv\Scripts\python.exe fill_execution_record.py runs\<run-folder>
```
