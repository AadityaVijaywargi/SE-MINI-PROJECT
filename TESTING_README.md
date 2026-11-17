# Testing Module: Full-Stack Prediction Platform (ML vs DL vs QML)

Independent black-box system testing of the Full-Stack Prediction Platform. The test plan
follows IEEE 829 (STP FSPP-MTP02.0), and the 50 system test cases (ST-01 to ST-50) are automated
with Playwright and pytest. Everything for testing is in `testing-docs/`. The application code
(`app.py`, `core/`, `static/`, `tools/`) and the developers' unit tests (`tests/`) are outside
the testing module.

| | |
|---|---|
| Tester | Ravi Teja (Ravi S), independent black-box tester |
| Build under test | Release 1.0, commit `4988037`; then the same build with the DEF-01 to DEF-04 fixes, now committed as `61fc217` on branch `testing` |
| Test plan | `testing-docs/01_Test_Documentation/STP/STP_Full_Stack_Prediction_Platform_v2.2.docx` |
| Final regression | Run 4: 50 / 50 PASS (`testing-docs/04_Test_Execution_Results/20261110-225957-4988037-modified/`); Run 9 (committed build `61fc217` on 6-part layout): **50 / 50 PASS** (`testing-docs/04_Test_Execution_Results/20261117-194428-61fc217-modified/`) |
| Final report | `testing-docs/06_Final_Reports/Test_Summary_Report_v1.1.docx` |
| Master workflow record | `FINAL_WORKFLOW.md` (repository root: complete technical project record) |

---

## 1. Testing architecture

```
 Requirements & plan            Test design                   Automation                         Evidence & reporting
 ───────────────────            ───────────                   ──────────                         ────────────────────
 SRS v1.2 ──► STP v2.2 ──►  System_Test_Cases.xlsx  ──►  run_tests.ps1                     ──►  04_…/<run folder>/
 (01_Test_Documentation)    ST-01 … ST-50,              │  └─ pytest + pytest-playwright       report.html, results.csv,
                            traceability, test data     │      ├─ conftest.py: starts the app    junit.xml, screenshots/,
                            Test_Automation_Mapping ────┘      │   on 127.0.0.1:5055 (cpu),      playwright/ (traces),
                                                               │   refuses non-local URLs,       app-server.log, run-info.txt
                            03_Test_Data/                      │   saves step screenshots              │
                            TD-01 … TD-13 (+ TD-14  ──────────►├─ pages.py: page helpers               ▼
                            generated at run time)             └─ tests/test_*.py: one test    fill_execution_record.py
                                                                   function per ST case                 │
                                                                                                        ▼
                                                               Execution record ─► Defect log ─► Developer review ─► Test Summary Report
                                                               (04_…/Execution_Records)  (05_Defect_Management)      (06_Final_Reports)
```

**Test levels** (STP section 8.1):

| Level | Who | How | Evidence |
|---|---|---|---|
| Unit and integration / API | Developers | pytest suite in `tests/` (CI: `.github/workflows/ci.yml`) | `testing-docs/04_Test_Execution_Results/Developer_Unit_Tests/` |
| **System (this module)** | Tester | 50 black-box cases, Playwright browser and HTTP checks | `testing-docs/04_Test_Execution_Results/` |
| Acceptance | Evaluator with the tester | About 30 cases chosen from the 50 | Pending (column *Selected for Acceptance* in the execution record) |

**Framework rules** (from `conftest.py` and `fill_execution_record.py`):

- **Black box.** The tests use only page elements, visible text and HTTP. They change no application file.
- **Local only.** The tests start their own copy of the app from the project `.venv` on
  `http://127.0.0.1:5055`, with `COMPUTE_DEVICE=cpu` and uploads redirected into the run folder.
  Any non-local URL is refused.
- **Isolated cases.** Every test gets a fresh browser context, so it has its own session cookie.
- **No invented results.** A case is PASS only if pytest passed with no manual check pending.
  If pytest passed but a person still has to check something, the case is UNVERIFIED.

---

## 2. Folder structure

```
TESTING_README.md                          ← this guide
testing-docs/
├── 01_Test_Documentation/
│   ├── SRS/
│   │   ├── SRS_Full_Stack_Prediction_Platform_v1.2.docx      current requirements specification
│   │   └── Previous_Versions/                                 v1.1, v1.1 draft (local only)
│   ├── STP/
│   │   ├── STP_Full_Stack_Prediction_Platform_v2.2.docx      current test plan (IEEE 829)
│   │   └── Previous_Versions/                                 v2.1, v2.0 (local only)
│   └── Test_Cases/
│       ├── System_Test_Cases.xlsx                             ST-01…ST-50, traceability, test data sheet
│       ├── Requirements_Traceability_Matrix.xlsx               FSPP-RTM-01: SRS v1.2 requirements → test cases → execution
│       ├── Test_Automation_Mapping.xlsx                       test case → script → locators
│       └── Manual_Verification_Guide.md                       how the timing-dependent checks are verified
│
├── 02_Test_Automation/
│   ├── run_tests.ps1                                          entry point: runs pytest, collects evidence
│   ├── conftest.py                                            app-server fixture, evidence capture, results.csv
│   ├── pages.py                                               page helpers and locators
│   ├── settings.py                                            folder paths, port, base URL
│   ├── pytest.ini                                             markers (st, slow), browser channel
│   ├── fill_execution_record.py                               results.csv → execution record workbook
│   ├── requirements-e2e.txt                                   pinned test dependencies
│   ├── README.md                                              framework details
│   ├── tests/                                                 11 scripts, 50 test functions
│   └── .venv/                                                 test virtual environment (local only)
│
├── 03_Test_Data/
│   ├── make_test_data.py                                      regenerates TD-01…TD-13 (seed 42); TD-14 with --oversize
│   └── TD-01_notes.txt … TD-13_=SUM(1,1).csv
│
├── 04_Test_Execution_Results/
│   ├── README.md                                              run index: what each run was and its outcome
│   ├── <date-time>-<commit>[-modified]/                       one self-contained evidence folder per run
│   ├── dev-check/                                             framework debugging runs, not evidence (local only)
│   ├── Execution_Records/                                     Test_Execution_Record_Final.xlsx, template
│   └── Developer_Unit_Tests/                                  developer pytest output before/after fixes
│
├── 05_Defect_Management/
│   ├── Defect_Log.xlsx                                        DEF-01…DEF-04, severity, fix, retest
│   ├── Developer_Review_v1.1.docx                             defects, fixes for review (3A), questions Q1–Q9
│   └── Previous_Versions/                                     Developer Review v1.0 (local only)
│
└── 06_Final_Reports/
    ├── Test_Summary_Report_v1.1.docx                          IEEE 829 Test Summary Report (run 4)
    ├── Integration_Testing_Summary.xlsx                       FSPP-ITS-01: 26 scenarios (17 system tests, 9 developer tests)
    ├── API_Testing_Summary.xlsx                               FSPP-ATS-01: 10 endpoints, 26 checks, request logs
    ├── Final_Completion_Report.md                             completion summary of 10 Nov 2026
    ├── Testing_Implementation_History.md                      full history of the testing work
    ├── Templates/Test_Summary_Report_Template.docx
    └── Previous_Versions/                                     TSR v1.0 for run 2 (local only)
```

"Local only" means the folder is kept on disk but is git-ignored (see `.gitignore`). These are
superseded versions, run 1, the `dev-check` debug runs, virtual environments and coverage
databases.

**Naming conventions:**

- Issued Word documents carry their version: `<Name>_v<major>.<minor>.docx`.
- Workbooks are registers and keep their version on their cover or revision sheet.
- Run folders are named `<yyyyMMdd-HHmmss>-<commit>[-modified]`, so a run name always shows when
  it ran and which build it tested.

---

## 3. Tools and their purpose

| Tool | Version | Purpose |
|---|---|---|
| Playwright for Python | 1.65.0 | Drives the browser: Google Chrome (default channel `chrome`), Microsoft Edge (ST-39, ST-49) and Playwright Chromium (ST-49); screenshots and traces |
| pytest | 9.2.0 | Test runner; `@pytest.mark.st("ST-nn")` ties each test to its case, `slow` marks long cases |
| pytest-playwright | 0.10.0 | `page` and `browser` fixtures, `--screenshot`, `--tracing`, `--output` |
| pytest-html | 4.2.0 | `report.html` with notes, manual-review items and step screenshots |
| Python | 3.11.9 | Test virtual environment (`testing-docs/02_Test_Automation/.venv`) and the application's `.venv` |
| PowerShell | Windows PowerShell 5.1 | `run_tests.ps1`: run folder, environment record (`run-info.txt`), pytest arguments |
| openpyxl | 3.1.5 (application `.venv`) | `fill_execution_record.py` fills the execution record workbook |
| Microsoft Excel (COM) | installed | ST-45: opens the exported CSV and counts formula and error cells |
| `tools/measure_times.py` | project tool | ST-50: API response times |
| `nvidia-smi` | NVIDIA driver | ST-35: records the GPU while the app runs with `COMPUTE_DEVICE=auto` |
| Git | — | The commit SHA in each run folder name and in `run-info.txt` |

---

## 4. Setting up the testing environment (once)

Prerequisites: Windows 10 or 11, Python 3.11, Google Chrome and Microsoft Edge. ST-45 also needs
Microsoft Excel. The recorded runs used an NVIDIA GPU for ST-35 (`COMPUTE_DEVICE=auto`).

1. **The application's environment.** The tests start the app with it. From the repository root:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   ```

2. **The test environment**, kept separate from the application's:

   ```powershell
   cd testing-docs\02_Test_Automation
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements-e2e.txt
   .venv\Scripts\python.exe -m playwright install chromium
   ```

3. **Test data.** TD-01 to TD-13 are already in `testing-docs/03_Test_Data/`. If
   `TD-12_large_40000.csv` is missing, `run_tests.ps1` regenerates the files with
   `make_test_data.py`. TD-14 (101 MB) is generated at run time by ST-05.

4. **Port 5055** must be free. Set `E2E_PORT` to use a different one.

5. **Keep the repository path short.** Use at most about 75 characters; the current location is
   68. The deepest evidence file is 184 characters below the repository root, and Windows tools
   (Git, Explorer, and Excel in ST-45) fail above 260 characters in total. Run 7 shows what
   happens otherwise: see `testing-docs/04_Test_Execution_Results/README.md`.

---

## 5. Executing the tests

From `testing-docs\02_Test_Automation`:

```powershell
.\run_tests.ps1                          # all 50 cases (about 11-13 minutes)
.\run_tests.ps1 -SkipSlow                # skip ST-31, ST-34, ST-50 (large data / timing)
.\run_tests.ps1 -Select st07             # one case (any pytest -k expression)
.\run_tests.ps1 -Select "st01 or st02"   # several cases
.\run_tests.ps1 -Headed                  # show the browser window
```

If PowerShell blocks scripts, run `powershell -ExecutionPolicy Bypass -File .\run_tests.ps1`
instead.

Each run creates `testing-docs\04_Test_Execution_Results\<date-time>-<commit>\` and
prints the folder at the end. The pytest exit code is recorded in `run-info.txt`: 0 means every
selected case passed.

**Fill the execution record for a run.** This uses openpyxl from the application's `.venv`:

```powershell
..\..\.venv\Scripts\python.exe fill_execution_record.py ..\04_Test_Execution_Results\<run-folder>
```

This writes `Test_Execution_Record_<run>.xlsx` into the run folder. The blank template in
`Execution_Records\` is never changed.

**Open a trace of a failed test:**

```powershell
.venv\Scripts\python.exe -m playwright show-trace ..\04_Test_Execution_Results\20261110-132648-4988037\playwright\test-training-py-test-st33-class-balancing-reported-chromium\trace.zip
```

---

## 6. Where to find everything

| Looking for | Location |
|---|---|
| Test plan / requirements | `testing-docs/01_Test_Documentation/STP/…v2.2.docx`, `…/SRS/…v1.2.docx` |
| The 50 test cases, traceability, test data list | `testing-docs/01_Test_Documentation/Test_Cases/System_Test_Cases.xlsx` |
| Which script and function automates each case | `testing-docs/01_Test_Documentation/Test_Cases/Test_Automation_Mapping.xlsx` |
| Test scripts | `testing-docs/02_Test_Automation/tests/test_*.py` |
| Input files | `testing-docs/03_Test_Data/` |
| Index of all runs and their outcomes | `testing-docs/04_Test_Execution_Results/README.md` |
| **Final regression evidence (run 4)** | `testing-docs/04_Test_Execution_Results/20261110-225957-4988037-modified/report.html` |
| **Screenshots** | `…/04_Test_Execution_Results/<run>/screenshots/<test name>/` |
| **Application logs** | `…/04_Test_Execution_Results/<run>/app-server.log` |
| **Traces of failures** | `…/04_Test_Execution_Results/20261110-132648-4988037/playwright/` (ST-33, original build; also in run 1) |
| Per-case results (CSV / JUnit) | `…/04_Test_Execution_Results/<run>/results.csv`, `junit.xml` |
| Formal execution record | `testing-docs/04_Test_Execution_Results/Execution_Records/Test_Execution_Record_Final.xlsx` |
| **Latest regression evidence (Run 9, 50/50 PASS)** | `testing-docs/04_Test_Execution_Results/20261117-194428-61fc217-modified/report.html` |
| Requirements Traceability Matrix (RTM) | `testing-docs/01_Test_Documentation/Test_Cases/Requirements_Traceability_Matrix.xlsx` |
| Integration Testing Summary (26 scenarios) | `testing-docs/06_Final_Reports/Integration_Testing_Summary.xlsx` |
| API Testing Summary (10 endpoints, 26 checks) | `testing-docs/06_Final_Reports/API_Testing_Summary.xlsx` |
| Developer unit-test results | `testing-docs/04_Test_Execution_Results/Developer_Unit_Tests/` |
| **Defect records** | `testing-docs/05_Defect_Management/Defect_Log.xlsx` |
| Fixes and open questions for the developers | `testing-docs/05_Defect_Management/Developer_Review_v1.1.docx` |
| Test Summary Report | `testing-docs/06_Final_Reports/Test_Summary_Report_v1.1.docx` |
| Master Workflow & Technical Record | `FINAL_WORKFLOW.md` (repository root) |
| Full history of the testing work | `testing-docs/06_Final_Reports/Testing_Implementation_History.md` |

---

## 7. Demonstrating the testing pipeline step by step

About 15 minutes, with no full run during the session. Have the folders and documents open
beforehand.

1. **Requirements to plan.** Open `STP_…_v2.2.docx`. Show these sections:
   - 4 (Test Items) and 6 (Features to Be Tested);
   - 8.1 (Testing Levels);
   - 8.8 (Retesting After Defect Fixes);
   - 8.10 (Browser Test Automation, Playwright);
   - 8.11 (ML, DL and QML Performance-Tuning Verification);
   - 9 (Item Pass/Fail Criteria).
2. **Test design.** Open `System_Test_Cases.xlsx` on the *Test Cases* sheet, at **ST-07**
   (minimum row boundary, REQ-UP-4). TD-04 with 49 rows must be rejected and TD-05 with 50 rows
   accepted. Walk through its columns: requirement, preconditions, steps, test data, expected
   result, actual result and PASS/FAIL. Then show the *Traceability* and *Test Data* sheets.
3. **Case to script.** Open `Test_Automation_Mapping.xlsx` (*Mapping* sheet) at ST-07: method,
   reason, existing developer coverage and script. Then open
   `testing-docs/02_Test_Automation/tests/test_upload.py` at `test_st07_minimum_row_boundary`. Point out the
   `@pytest.mark.st("ST-07")` marker, the `App` page helper from `pages.py`, the `td(…)` test-data
   lookup and the `evidence.shot(…)` screenshots.
4. **Framework.** In `conftest.py`, show the `AppServer` fixture (starts the app locally on port
   5055 with `COMPUTE_DEVICE=cpu`), `assert_local` (refuses non-local URLs) and the `results.csv`
   writer.
5. **Live execution.** In `testing-docs\02_Test_Automation`, run:

   ```powershell
   .\run_tests.ps1 -Select "st01 or st07 or st08" -Headed
   ```

   The browser opens, the app starts and the cases run. For scale, run 8 took 30 seconds for two
   cases, including starting the app.
   Open the new folder printed at the end and show `run-info.txt`, `report.html`, `results.csv`
   and `screenshots\`.
6. **Integration and API testing verification.** Open `Integration_Testing_Summary.xlsx`
   (26 scenarios: 17 independent system tests, 9 developer integration tests across the 8 boundaries)
   and `API_Testing_Summary.xlsx` (all 10 endpoints verified across 26 discrete checks with status
   codes verified in the server log). Show the clear distinction between independent system tests
   and developer unit/integration tests.
7. **A defect from detection to retest.** In run 2 (`…/20261110-132648-4988037/`), show the ST-33
   failure in `report.html` and its trace (`playwright show-trace …`, section 5). Then open
   `Defect_Log.xlsx` at DEF-01 (severity, root cause, fix, retest PASS) and
   `Developer_Review_v1.1.docx` section 3A.
8. **Final regression evidence, execution record, and workflow.** Open `report.html` in
   Run 9 (`…/04_Test_Execution_Results/20261117-194428-61fc217-modified/`): 50 passed (and Run 4
   baseline). Show `Test_Execution_Record_Final.xlsx` (*Execution*, *Summary*, and *Run History*
   sheets showing Runs 1 to 9). Present `Test_Summary_Report_v1.1.docx` and conclude with the master
   project record `FINAL_WORKFLOW.md` at the repository root.

Optional, if time allows: `.\run_tests.ps1 -SkipSlow` (most of the suite), or the full
`.\run_tests.ps1` (about 11 to 13 minutes).

---

## 8. Path changes from the 17 Nov 2026 reorganisation

The testing files were reorganised from `testing-docs/final/` into the numbered folders above.
File contents were not changed by the move (checked by MD5).

The Word and Excel deliverables were left byte-identical, because they are issued versions and
the test cases are not to be edited. They still cite the old paths, for example
`testing-docs/final/test-data/TD-04_rows_49.csv` in `System_Test_Cases.xlsx`, or
`tests/runs/<run>/` relative to `testing-docs/final/` in the STP and TSR. Use this table to
resolve them:

| Old path (cited in the deliverables) | New path |
|---|---|
| `testing-docs/final/SRS_Full_Stack_Prediction_Platform_v1.2.docx` | `testing-docs/01_Test_Documentation/SRS/SRS_Full_Stack_Prediction_Platform_v1.2.docx` |
| `testing-docs/final/STP_Full_Stack_Prediction_Platform_v2.2.docx` | `testing-docs/01_Test_Documentation/STP/STP_Full_Stack_Prediction_Platform_v2.2.docx` |
| `testing-docs/final/System_Test_Cases.xlsx` | `testing-docs/01_Test_Documentation/Test_Cases/System_Test_Cases.xlsx` |
| `testing-docs/final/Test_Automation_Mapping.xlsx` | `testing-docs/01_Test_Documentation/Test_Cases/Test_Automation_Mapping.xlsx` |
| `testing-docs/final/Manual_Verification_Guide.md` | `testing-docs/01_Test_Documentation/Test_Cases/Manual_Verification_Guide.md` |
| `testing-docs/final/tests/` (framework files) | `testing-docs/02_Test_Automation/` |
| `testing-docs/final/tests/test_*.py` | `testing-docs/02_Test_Automation/tests/test_*.py` |
| `testing-docs/final/tests/.venv` | `testing-docs/02_Test_Automation/.venv` |
| `testing-docs/final/test-data/` | `testing-docs/03_Test_Data/` |
| `testing-docs/final/tests/runs/<run>/` (also written `tests/runs/<run>/` or `runs/<run>/`) | `testing-docs/04_Test_Execution_Results/<run>/` (folder names unchanged) |
| `testing-docs/final/Test_Execution_Record_Final.xlsx` | `testing-docs/04_Test_Execution_Results/Execution_Records/Test_Execution_Record_Final.xlsx` |
| `testing-docs/final/Test_Execution_Record.xlsx` (blank) | `testing-docs/04_Test_Execution_Results/Execution_Records/Test_Execution_Record_Template.xlsx` |
| `testing-docs/final/evidence/` | `testing-docs/04_Test_Execution_Results/Developer_Unit_Tests/` |
| `testing-docs/final/Defect_Log.xlsx` | `testing-docs/05_Defect_Management/Defect_Log.xlsx` |
| `testing-docs/final/Developer_Review.docx` | `testing-docs/05_Defect_Management/Developer_Review_v1.1.docx` |
| `testing-docs/final/Test_Summary_Report_Final.docx` | `testing-docs/06_Final_Reports/Test_Summary_Report_v1.1.docx` |
| `testing-docs/final/Test_Summary_Report_Template.docx` | `testing-docs/06_Final_Reports/Templates/Test_Summary_Report_Template.docx` |
| `TESTING_IMPLEMENTATION.md` (root) | `testing-docs/06_Final_Reports/Testing_Implementation_History.md` |
| `FINAL_COMPLETION_REPORT.md` (root) | `testing-docs/06_Final_Reports/Final_Completion_Report.md` |
| `testing-docs/final/archive/` | `Previous_Versions/` next to each current document |
| `testing-docs/SRS_…_v1.1.docx`, `testing-docs/STP_…_v2.0.docx` (drafts) | `…/SRS/Previous_Versions/SRS_…_v1.1_draft.docx`, `…/STP/Previous_Versions/STP_…_v2.0.docx` |

**Left at the repository root:**

- `SRS.docx` and `STP.docx`, the submission copies. `SRS.docx` is byte-identical to SRS v1.2.
  `STP.docx` started as a copy of STP v2.2 but was edited afterwards (11 Nov 2026), so it now
  differs from the versioned file.
- The original v1.0 specification PDFs (git-ignored).
