# Testing Implementation Record

**Full-Stack Prediction Platform (ML vs DL vs QML)**: system testing, from first analysis to current status

> [!NOTE]
> **Folder reorganisation, 17 Nov 2026.** This record was written as `TESTING_IMPLEMENTATION.md`
> at the repository root. It now lives in `testing-docs/06_Final_Reports/`. Its content is
> unchanged, and the paths in it describe the layout at the time (`testing-docs/final/...`).
> The old-to-new path table is in `TESTING_README.md` (section 8) at the repository root.

| Field | Value |
|---|---|
| Document | `TESTING_IMPLEMENTATION.md`, a historical record of the testing work |
| Tester | **Ravi Teja (Ravi S)**, independent black-box tester |
| Developers | Aaditya Vijaywargi, Anish Kumar, Arya Devang (README Team table) |
| Repository | `github.com/AadityaVijaywargi/SE-MINI-PROJECT` |
| Build under test | Commit **`4988037`** (release 1.0): "fix(ml): find rare classes: class weights, tuned thresholds, PR-AUC ranking" |
| Test plan | STP **FSPP-MTP02.0**, version **2.1** (IEEE 829) |
| Requirements baseline | SRS version **1.1** |
| Date of record | 10 November 2026 |
| USN / institution / faculty guide | [FILL LATER] |

---

## Status legend

Every status in this document uses one of these labels. Failures and open issues are marked in red.

| Label | Meaning |
|---|---|
| 🟢 **COMPLETED** / **PASS** | Done and backed by execution evidence |
| 🔴 **FAILED** / **FAIL** | Executed; the actual result did not match the expected result |
| 🔴 **UNVERIFIED** | Executed, but a required check could not be confirmed; never counted as PASS |
| 🟠 **DEVELOPER REVIEW** | Waiting for a developer decision or fix |
| 🟡 **PENDING** | Not done yet: needs the tester, the developers or the faculty |
| ⚪ **NOT RUN** | Not executed in that run |

> [!IMPORTANT]
> **Current QA status (updated 10 Nov 2026, 23:30): ready for the evaluator's acceptance run once the developers have approved and committed the fixes. Not accepted yet.**
> - **Fixes:** the four defects (DEF-01 to DEF-04) were fixed at the project owner's request on branch `qa/defect-fixes-final-testing`. They are not committed; see [section 15](#15-defect-fixes-and-regression-cycle).
> - **Regression run 4:** all 50 cases on the fixed build passed, **50 / 50 PASS**.
> - **DEF-04 fresh-server retest (run 5):** `POST /api/train` → 202 in 18 ms.
> - **Developer suite:** 234 + 1 passed, with 96% coverage.
> - <span style="color:#C55A11">**DEVELOPER REVIEW:**</span> the developers must review and commit the fixes before the defects can be Closed. Q1–Q9 are open.
> - <span style="color:#c00000">**PENDING:**</span> the acceptance run, the approvals and the `[FILL LATER]` fields.
>
> Sections 1–14 below record the cycle on the original build (runs 1–3), where run 2 ended with 47 PASS, 2 FAIL and 1 UNVERIFIED.

> [!CAUTION]
> **Status before the fixes (original build, run 2): <span style="color:#c00000">NOT READY FOR ACCEPTANCE</span>.**
> In run 2, all 50 cases were automated and executed: **47 PASS**, **<span style="color:#c00000">2 FAIL</span>** (ST-33, ST-50) and **<span style="color:#c00000">1 UNVERIFIED</span>** (ST-36).
> Two Major defects, **<span style="color:#c00000">DEF-01</span>** and **<span style="color:#c00000">DEF-04</span>**, were open. Under STP section 9, an open Critical or Major defect blocks acceptance.

---

> [!NOTE]
> **Files not in the GitHub submission** (see `.gitignore`): run 1 (`tests/runs/20261110-122004-4988037/`), `testing-docs/final/archive/`, the drafts `testing-docs/SRS_…_v1.1.docx` and `testing-docs/STP_…_v2.0.docx`, and the original PDFs in the root. References to them are marked: *Local-only historical evidence; not included in GitHub submission.* The final versions are in the submission.

## Contents

1. [Project overview and the tester's role](#1-project-overview-and-the-testers-role)
2. [Initial repository analysis and existing testing infrastructure](#2-initial-repository-analysis-and-existing-testing-infrastructure)
3. [SRS and IEEE 829 STP corrections](#3-srs-and-ieee-829-stp-corrections)
4. [System test cases ST-01 to ST-50](#4-system-test-cases-st-01-to-st-50)
5. [Playwright automation architecture and implementation](#5-playwright-automation-architecture-and-implementation)
6. [Environment setup and execution commands](#6-environment-setup-and-execution-commands)
7. [Execution results: run 1, run 2 and run 3](#7-execution-results-run-1-run-2-and-run-3)
8. [Screenshots, evidence and generated reports](#8-screenshots-evidence-and-generated-reports)
9. [Defects](#9-defects)
10. [ML, DL and QML performance-tuning verification](#10-ml-dl-and-qml-performance-tuning-verification)
11. [Manual verification and pending activities](#11-manual-verification-and-pending-activities)
12. [Developer review requirements and open decisions](#12-developer-review-requirements-and-open-decisions)
13. [Deliverables and locations](#13-deliverables-and-locations)
14. [Remaining work, retesting procedure and acceptance status](#14-remaining-work-retesting-procedure-and-acceptance-status)
15. [Defect fixes and regression cycle](#15-defect-fixes-and-regression-cycle)
- [Appendix A: Testing timeline](#appendix-a-testing-timeline)
- [Appendix B: Integrity of originals and application code](#appendix-b-integrity-of-originals-and-application-code)

---

## 1. Project overview and the tester's role

### 1.1 System under test

The platform compares classical machine learning (ML), deep learning (DL) and quantum machine learning (QML) classifiers on any tabular dataset the user uploads.

| Layer | Implementation |
|---|---|
| Backend | Flask `app.py` with 10 `/api/*` endpoints and an in-process JobRunner thread for asynchronous training |
| Core pipeline | `core/` modules for profiling, preprocessing, targets, trainers, the Keras DNN (`dnn.py`), the PennyLane VQC on `default.qubit` (`qml.py`), the evaluator, export, storage, jobs and GPU detection |
| Frontend | `static/`: an ES-module single-page app with Chart.js charts. The status display refreshes every **2 s** |
| Models | 6 ML (Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, XGBoost), 1 DL (Deep Neural Network) and 1 QML (Variational Quantum Classifier) |
| Configuration | `.env`: `PORT=5000`, `COMPUTE_DEVICE=auto` |

### 1.2 The tester's role

Ravi Teja (Ravi S) is the **independent black-box tester**. The developers own the unit and integration tests in `tests/`. The tester owns:

- correcting the SRS and the IEEE 829 STP against the implemented build;
- designing 50 system test cases, from which the evaluator selects about 30 for acceptance;
- implementing the Playwright automation and collecting execution evidence;
- verifying UI/UX behaviour and ML/DL/QML performance tuning;
- logging defects and sending them to the developers for review;
- writing the IEEE 829 Test Summary Report.

### 1.3 Ground rules followed throughout

| Rule | How it was applied |
|---|---|
| Do not modify application source (`app.py`, `core/`, `static/`, …) | Held through runs 1–3: no tracked file changed. Afterwards the project owner explicitly asked for the defects to be fixed; the changes are on a separate branch, are not committed, and every changed file is listed in section 15 and Appendix B. |
| Keep original documents and evidence untouched | The original PDFs and the earlier drafts have unchanged MD5 checksums (Appendix B). Final versions are saved separately in `testing-docs/final/`. |
| Never invent results; never mark unverified cases PASS | Statuses come only from pytest output (`results.csv`). A case with an open manual check is **UNVERIFIED**. |
| Run tests only against the local application | `conftest.py` refuses any host other than `127.0.0.1` or `localhost`. |
| Keep personal and approval fields as placeholders | USN, institution, faculty name and signatures stay `[FILL LATER]`. Developer questions are marked `[DEV CONFIRM]`. |

---

## 2. Initial repository analysis and existing testing infrastructure

The first phase was **analysis only**: no file changes, installs or test runs. It covered the code, the documents and the professor's requirements.

### 2.1 Professor's requirements for the testing phase

1. A Software Test Plan in **IEEE 829** format, with section 8.8 renamed **"Retesting After Defect Fixes"**.
2. About **50 test cases**, each with ID, input, expected result, actual result and PASS/FAIL. The evaluator selects about **30** for acceptance.
3. **Playwright** automation, with scripts and execution evidence.
4. Verification of **UI/UX** and of **ML/DL/QML performance tuning**.
5. Acceptance, suspension and resumption criteria; environment, staffing and schedule.
6. Five deliverables: **Test Plan, Test Cases, Test Scripts, Automated Execution Evidence, Test Summary Report**.

### 2.2 Existing testing infrastructure (developer-owned, reused and not duplicated)

| Item | Location | Notes |
|---|---|---|
| Unit and integration tests | `tests/` (15 files) | Test files: `test_api.py`, `test_dnn.py`, `test_evaluator.py`, `test_export.py`, `test_gpu.py`, `test_jobs.py`, `test_preprocessing.py`, `test_profiling.py`, `test_qml.py`, `test_security.py`, `test_storage.py`, `test_targets.py`, `test_trainers.py`; also `conftest.py` and `__init__.py` |
| Timing tool | `tools/measure_times.py` | Reused unchanged by ST-50 |
| Sample generator | `tools/make_samples.py` | Produces the bundled samples in `data/samples/` |
| CI | `.github/workflows/ci.yml` | Developer pipeline; not changed |
| Reference documents | `docs/deviations.md` (29 changes against the SRS), `docs/performance.md` (response-time targets), `docs/demo-script.md` (demo checklist) | Used as test oracles alongside the README |
| Original specifications | `SRS_Full_Stack_Prediction_Platform (1).pdf`, `STP_Full_Stack_Prediction_Platform.pdf` (repository root) | Kept unchanged. *Local-only historical evidence; not included in GitHub submission.* Final versions: `SRS.docx`, `STP.docx` |
| Earlier test cases | "Test Cases: ML/DL/QML Training Dashboard" (TC-01 to TC-14, 12 Oct 2026) | Mapped to the new cases in STP Appendix D |

**Decision:** the system tests live in a separate folder, `testing-docs/final/tests/`, with their own virtual environment. The developers' `tests/` folder and its CI stay unchanged, and checks that developer pytest already makes at API level are not re-automated (Test_Automation_Mapping.xlsx, column G).

### 2.3 Gap analysis

The SRS 1.0 and STP 1.0 were compared with commit `4988037`, and the result was a correction checklist (document, section, problem, correction, replacement text) that the tester approved before any edits. Main findings:

- 29 implemented deviations (`docs/deviations.md`) were missing from SRS 1.0.
- STP 1.0 referenced files that don't exist (`splitting.py`, `preprocessing_reference.py`) and had no Playwright, system-test-case or tuning-verification sections.
- Response-time targets were TBD in the SRS; they are resolved by `docs/performance.md`.

---

## 3. SRS and IEEE 829 STP corrections

### 3.1 SRS version 1.0 to 1.1

File: `testing-docs/final/archive/SRS_Full_Stack_Prediction_Platform_v1.1.docx` (*Local-only historical evidence; not included in GitHub submission.*) (20 pages; Wiegers/IEEE 830 structure kept). Superseded by **v1.2** in `testing-docs/final/` (S25: REQ-MS-4 restated on SVM training rows after DEF-02; reference to STP 2.2).

The correction log has **24 entries (S1 to S24)**. Highlights:

| # | Section | Change |
|---|---|---|
| S1 | Cover, revision history | Team names added, with Ravi Teja (Ravi S) as tester; institution details left as placeholders |
| S9 | 3.3 | Full API table (10 endpoints), result fields, simulator |
| S12 | 4.3 REQ-MS-5 | Ranges and defaults; threshold and class-balancing settings |
| S13–S14 | 4.4, 4.5 | New REQ-PP-11 to 13 and REQ-TR-9 (validation rows, positive class, missing targets, class weights) |
| S15 | 4.6 | `cancelled` status, per-epoch progress, testing stage, session cookie |
| S16 | 4.7 | PR-AUC, test time, ranking rule; new REQ-EV-6 for threshold tuning |
| S18 | 5.1 | Measured performance targets; PERF-8 added; TBD-2 resolved |
| S20 | 5.3 | SEC-6 ("model timeouts") flagged for the developers |
| S24 | 4.3 REQ-MS-4 | 🟠 The SVM warning threshold and the actual subsampling rule disagree: `[DEV CONFIRM]` (became DEF-02) |

Reference 11 points to STP FSPP-MTP02.0 version 2.1.

### 3.2 STP version 1.0 to 2.0 to 2.1

File: `testing-docs/final/archive/STP_Full_Stack_Prediction_Platform_v2.1.docx` (*Local-only historical evidence; not included in GitHub submission.*) (23 pages), superseded by **v2.2** in `testing-docs/final/` (T36–T39: fix and retest status, SRS 1.2, Appendix A title). All **19 IEEE 829 sections** are kept, plus appendices A to E. Appendix A logs **35 corrections (T1 to T35)**.

| Section | Change in this test cycle |
|---|---|
| 1, 2 | Plan ID FSPP-MTP02.0, build SHA; references updated to SRS 1.1 and repository documents |
| **3** | Split into **3.1 Purpose**, **3.2 Scope and Approach**, **3.3 Architecture of the System Under Test** (diagram and test-level table) |
| **4** | Corrected module paths; item **L** testware versions; item **M** test data |
| 5 | Risks R2 and R9 updated; R10 to R13 added (GPU laptop, CDN, flaky automation, requirements lag) |
| **6** | New "System test cases" column for every feature |
| 8.1–8.7 | Levels reassigned between developers and tester; tools; metrics; test data; defect severity |
| **8.8** | Renamed **"Retesting After Defect Fixes"** (faculty instruction) |
| **8.9** | Test case design and format (the 10-field template) |
| **8.10** | Browser test automation with Playwright; all 50 cases scripted |
| **8.11** | ML, DL and QML performance-tuning verification: 14 measurable checks, **P1 to P14** |
| 9, 10 | Item pass/fail and acceptance criteria; six suspension and resumption rows for the tester |
| 11–13 | The five deliverables with real file names; task status after runs 1 and 2; actual hardware, OS and tool versions |
| 14–17 | Named staffing and responsibilities; dated, adjustable schedule with a status column; contingencies |
| 18 | Approvals: tester, development lead, project manager, customer/evaluator (all `[FILL LATER]`) |
| App. B–E | Test case template; index of the 50 cases; mapping of TC-01 to TC-14; requirement coverage |

Drafts from the earlier step are kept unchanged in `testing-docs/` (SRS v1.1 draft, STP v2.0 draft; *Local-only historical evidence; not included in GitHub submission.*) The final STP v2.0 was replaced by v2.1, and v2.1 by v2.2 (v2.1 kept in `testing-docs/final/archive/`).

---

## 4. System test cases ST-01 to ST-50

File: `testing-docs/final/System_Test_Cases.xlsx` (FSPP-STC-01, **revision 1.2**). Sheets: Cover, Test Cases, Traceability, Test Data, Template.

Each case has 10 fields (STP 8.9), plus Execution Type:
1. Test Case ID
2. Requirement ID
3. Test Scenario
4. Preconditions
5. Test Steps
6. Test Data
7. Expected Result
8. Actual Result
9. PASS/FAIL
10. Priority

The *Actual Result* and *PASS/FAIL* columns stay blank in the design workbook by design. Results are recorded per run in the execution record (section 7).

### 4.1 Groups

| Group | Cases | Count |
|---|---|---|
| A. Dataset upload and validation | ST-01 to ST-08 | 8 |
| B. Dataset selection and preprocessing | ST-09 to ST-12 | 4 |
| C. Target selection and configuration | ST-13 to ST-19 | 7 |
| D. Advanced performance-tuning settings | ST-20 to ST-25 | 6 |
| E. ML, DL and QML training | ST-26 to ST-35 | 10 |
| F. Training progress and cancellation | ST-36 to ST-39 | 4 |
| G. Results dashboard, metrics and visualisations | ST-40 to ST-43 | 4 |
| H. CSV export | ST-44 to ST-45 | 2 |
| I. Security, invalid input and error handling | ST-46 to ST-47 | 2 |
| J. UI/UX and browser compatibility | ST-48 to ST-49 | 2 |
| K. Performance | ST-50 | 1 |
| **Total** | | **50** |

### 4.2 Revisions of the test-case workbook

| Revision | Change | Reason |
|---|---|---|
| 1.0 | 50 cases designed from SRS 1.1 | Initial design |
| 1.1 | **ST-45:** the sanitised name is `TD-13_SUM11.csv`. **ST-46:** a traversal request returns 404 or 405 (the developer test accepts both) | The original text was the tester's assumption, not a requirement; corrected after run 1 and logged |
| 1.2 | **ST-29, ST-30, ST-36:** add a TD-12 confirmation run. **ST-35:** automated, with app restarts | Stages shorter than the 2 s UI refresh are not drawn on the Banking sample; steps aligned with the implemented scripts |

### 4.3 Test data

Generated by `testing-docs/final/test-data/make_test_data.py` (seed 42):

| ID | File | Purpose |
|---|---|---|
| TD-01 | `TD-01_notes.txt` | Disallowed extension |
| TD-02 | `TD-02_lookalike.csv.exe` | Look-alike double extension |
| TD-03 | `TD-03_banking.xlsx` | Valid Excel upload |
| TD-04 / TD-05 | `TD-04_rows_49.csv` / `TD-05_rows_50.csv` | Minimum-row boundary |
| TD-06 | `TD-06_one_column.csv` | Single column |
| TD-07 / TD-08 | `TD-07_empty.csv` / `TD-08_corrupt.xlsx` | Empty and corrupt files |
| TD-09 / TD-10 | `TD-09_imbalanced_4to1.csv` / `TD-10_imbalanced_3to1.csv` | Imbalance warning boundary |
| TD-11 | `TD-11_single_row_class.csv` | Single-row class blocks training |
| TD-12 | `TD-12_large_40000.csv` | 40,000 rows: QML row cap, SVM subsampling, long runs |
| TD-13 | `TD-13_=SUM(1,1).csv` | CSV formula-injection name |
| TD-14 | generated at run time | 105,906,989 bytes (> 100 MB), for ST-05 |
| TD-15 | generated at run time | First 25,000 rows of TD-12, for the REQ-MS-4 observation |

The bundled samples in `data/samples/` (`banking.csv`, `healthcare.csv` and the others) are used unchanged.

---

## 5. Playwright automation architecture and implementation

### 5.1 Architecture

```
testing-docs/final/tests/          (own .venv; separate from the developers' tests/)
├── run_tests.ps1                  entry point: makes the run folder, writes run-info.txt, runs pytest
├── pytest.ini                     --browser-channel chrome, strict markers (st, slow, partial, manual)
├── requirements-e2e.txt           playwright 1.65.0, pytest 9.2.0, pytest-playwright 0.10.0, pytest-html 4.2.0
├── settings.py                    paths, PORT 5055, BASE_URL, local-only host list
├── conftest.py                    AppServer fixture (start / stop / restart(compute_device)),
│                                  local-only guard, Evidence (shot / note / manual),
│                                  results.csv writer, pytest-html extras
├── pages.py                       App page helper: record_training, server_stages,
│                                  wait_model_status, start_and_get_job, text(), number()
├── test_upload.py … test_performance.py   one test function per case (test_st01_… to test_st50_…)
├── fill_execution_record.py       results.csv → execution record (PASS / FAIL / UNVERIFIED rules)
├── README.md
└── runs/<yyyyMMdd-HHmmss>-<commit>/   evidence for each execution
```

**Flow of one run:**
1. `run_tests.ps1` creates the run folder and the test data (if missing), then records the commit, browser versions and package versions.
2. The `AppServer` fixture starts the application with the project's own `.venv` on **127.0.0.1:5055**, with `COMPUTE_DEVICE=cpu` and `UPLOAD_DIR` set to the run folder.
3. Each test gets a fresh browser context, so it has its own `pp_session` cookie.
4. Tests wait on page state, never on fixed sleeps. Screenshots and notes are attached to the pytest-html report.
5. `results.csv` records the pytest outcome, duration, message, any pending manual review and the screenshot count for each case.
6. `fill_execution_record.py` turns the CSV into the execution workbook.

**Status rules in `fill_execution_record.py`:**
- `passed` with no manual review pending → **PASS**
- `failed` → **FAIL**
- `passed` with a manual review pending → **UNVERIFIED** (never PASS)
- `skipped` → left blank (not run)

### 5.2 Modules

| File | Cases |
|---|---|
| `test_upload.py` | ST-01 to ST-08 |
| `test_dataset.py` | ST-09 to ST-12 |
| `test_configure.py` | ST-13 to ST-19 |
| `test_advanced_settings.py` | ST-20 to ST-25 |
| `test_training.py` | ST-26 to ST-35 (ST-35 restarts the app with `COMPUTE_DEVICE=auto`, then `cpu`) |
| `test_progress.py` | ST-36 to ST-39 |
| `test_results.py` | ST-40 to ST-43 |
| `test_export.py` | ST-44, ST-45 (Excel check through COM, using a temporary `.ps1`) |
| `test_errors.py` | ST-46, ST-47 |
| `test_ui.py` | ST-48 (focus contrast), ST-49 (responsive layout and stamp overlap) |
| `test_performance.py` | ST-50 (runs `tools/measure_times.py` unchanged) |

### 5.3 How the hard-to-automate cases were made measurable

| Case | Measurable criterion |
|---|---|
| ST-29, ST-30, ST-36 | Server stage labels sampled every 0.1 s; a TD-12 confirmation run when the Banking run is shorter than the 2 s UI refresh |
| ST-34 | Warning text and SVM training rows; plus a 25,000-row observation for REQ-MS-4 |
| ST-35 | The fixture restarts the app with `COMPUTE_DEVICE=auto`, then `cpu`; `nvidia-smi` is recorded |
| ST-42 | Chart.js dataset colours, labels, dashed baselines, axis types and legends |
| ST-45 | The exported CSV is opened in Microsoft Excel through COM; formula and error cells are counted |
| ST-48 | Every focus stop: outline at least 2 px and contrast at least 3:1 (WCAG 2.2 SC 1.4.11 / 2.4.13) |
| ST-05, ST-46, ST-47 | 101 MB file generated at run time; API checks against the live server; a real server stop and restart |

### 5.4 Problems met while building the framework

| Problem | Fix |
|---|---|
| `inner_text` returned CSS-uppercased labels | Used `text_content` through the `text()` helper |
| The 2 s UI refresh skipped short stages (ST-29, ST-30, ST-36) | 0.1 s server-stage polling plus TD-12 confirmation runs. ST-36 "Testing" was still never drawn → stays 🔴 UNVERIFIED |
| ST-40 font-size check tripped on the visually hidden `h1` | Visually hidden and 1 px elements filtered out |
| ST-48 `:focus` filter unreliable; BODY counted as a focus stop | Element index read through `evaluate`; BODY/HTML excluded |
| PowerShell `-Command` doesn't pass `$args` (ST-45 Excel COM) | A temporary `.ps1` run with `-File` |
| The cp1252 console couldn't print some characters | `PYTHONIOENCODING=utf-8` |

`runs/dev-check/` holds the debugging runs from building the framework. **It is not execution evidence.**

---

## 6. Environment setup and execution commands

### 6.1 Test environment

| Item | Value |
|---|---|
| Machine | RAVI_TEJA: AMD Ryzen 5 7535HS, 15.3 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (driver 566.07) |
| OS | Microsoft Windows 11 Home Single Language (build 26300) |
| Application | Flask development server started by the tests at `http://127.0.0.1:5055`, `COMPUTE_DEVICE=cpu`. The user's own instance uses port 5000 from `.env` |
| Python | 3.11.9 (test venv and app venv) |
| Test packages | playwright==1.65.0, pytest==9.2.0, pytest-playwright==0.10.0, pytest-html==4.2.0 |
| Browsers | Google Chrome 156.0.8078.81 (default, channel `chrome`); Microsoft Edge 157.0.4341.42 (ST-39, ST-49); Playwright Chromium 158.0.8127.18 (ST-49) |
| Document tooling | python-docx, openpyxl, Microsoft Word and Excel (COM) for TOC refresh, validation and the ST-45 Excel check |

### 6.2 One-time setup

```powershell
cd testing-docs\final\tests
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-e2e.txt
.venv\Scripts\python.exe -m playwright install chromium
```

The application's own `.venv` at the repository root must exist, because the tests start the app with it.

### 6.3 Running the system tests

```powershell
cd testing-docs\final\tests
.\run_tests.ps1               # all 50 cases
.\run_tests.ps1 -SkipSlow     # skip ST-31, ST-34, ST-50
.\run_tests.ps1 -Select st33  # one case (pytest -k expression)
.\run_tests.ps1 -Headed       # watch the browser
```

### 6.4 Filling the execution record from a run

```powershell
..\..\..\.venv\Scripts\python.exe fill_execution_record.py runs\<run-folder>
```

### 6.5 Developer test suite, run by the tester as a check (repository root)

```powershell
.venv\Scripts\python.exe -m pytest -m "not slow" -q --cov=core --cov=app --cov-report=term -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -m slow -q -p no:cacheprovider
```

---

## 7. Execution results: run 1, run 2 and run 3

### 7.1 Run overview

| Run | Folder (`testing-docs/final/tests/runs/`) | Time (10 Nov 2026) | Scope | pytest outcome | Execution record |
|---|---|---|---|---|---|
| Run 1 | `20261110-122004-4988037` (*Local-only historical evidence; not included in GitHub submission.*) | 12:20:04 – 12:27:22 | All 50 | 48 passed, <span style="color:#c00000">**1 failed**</span>, 1 skipped | 41 PASS, 🔴 1 FAIL, 8 left blank (7 manual reviews pending, 1 manual case skipped). *Superseded by run 2* |
| **Run 2 (final)** | **`20261110-132648-4988037`** | 13:26:48 – 13:34:07 | All 50 | 48 passed, <span style="color:#c00000">**2 failed**</span> | **47 PASS, 🔴 2 FAIL, 🔴 1 UNVERIFIED** |
| Run 3 (investigation) | `20261110-133444-4988037` | 13:34:44 – 13:35:18 | ST-50 only | <span style="color:#c00000">**1 failed**</span> | Confirms DEF-04 on a fresh server |
| Developer suite (not slow) | `evidence/developer-pytest-not-slow.txt` | 13:35:41 – 13:36:31 | `tests/` | 🟢 **231 passed**, 0 failed, 1 deselected | Coverage **96%** (1,145 statements, 44 missed) |
| Developer suite (slow) | `evidence/developer-pytest-slow.txt` | 13:36:39 – 13:36:50 | `tests/ -m slow` | 🟢 **1 passed** | Default QML run |

All three runs used commit `4988037`, with no local modifications.

> [!NOTE]
> The README states 199 developer tests and 97% coverage. The figures measured on commit `4988037` are **232 tests (231 + 1 slow)** and **96%**.

### 7.2 Run 2 summary against the plan

| Metric | Value |
|---|---|
| Planned cases | 50 |
| Automated and executed | 50 (100%) |
| PASS | 47 |
| <span style="color:#c00000">FAIL</span> | <span style="color:#c00000">**2**</span>: ST-33, ST-50 |
| <span style="color:#c00000">UNVERIFIED</span> | <span style="color:#c00000">**1**</span>: ST-36 |
| BLOCKED / NOT RUN | 0 |
| Pass rate (all 50) | 94% (target `[PROPOSED: 90%]`, STP 9; pending faculty confirmation) |
| Open Major defects | <span style="color:#c00000">**2** (DEF-01, DEF-04): block acceptance under STP 9</span> |

### 7.3 Per-case results: run 1 and run 2

Final status = run 2, from `Test_Execution_Record_Final.xlsx`. Durations are run 2 pytest durations.

| ID | Scenario | Pri. | Run 1 | Run 2 (final) | Run 2 (s) |
|---|---|---|---|---|---|
| ST-01 | Upload valid CSV with the file picker | High | PASS | 🟢 PASS | 1.0 |
| ST-02 | Upload valid CSV by drag-and-drop | Med | PASS | 🟢 PASS | 1.0 |
| ST-03 | Upload valid Excel (.xlsx) | High | PASS | 🟢 PASS | 2.0 |
| ST-04 | Reject disallowed and look-alike extensions | High | PASS | 🟢 PASS | 1.2 |
| ST-05 | Reject file > 100 MB in the browser | High | PASS | 🟢 PASS | 1.3 |
| ST-06 | Reject empty and corrupt files | Med | PASS | 🟢 PASS | 1.3 |
| ST-07 | Minimum row boundary (49 rejected / 50 accepted) | High | PASS | 🟢 PASS | 1.0 |
| ST-08 | Reject single-column file | Med | PASS | 🟢 PASS | 0.7 |
| ST-09 | Load each of the four sample datasets | High | PASS | 🟢 PASS | 1.7 |
| ST-10 | Identifier and constant columns flagged | High | PASS | 🟢 PASS | 2.0 |
| ST-11 | Missing values counted and blanked | Med | PASS | 🟢 PASS | 1.6 |
| ST-12 | Preprocessing reported in the metadata | High | PASS | 🟢 PASS | 4.7 |
| ST-13 | Suggested target preselected and starred | High | PASS | 🟢 PASS | 3.6 |
| ST-14 | Binary task analysis | Med | PASS | 🟢 PASS | 0.9 |
| ST-15 | Imbalance warning only above 3:1 | High | PASS | 🟢 PASS | 1.1 |
| ST-16 | Training blocked for unsupported targets | High | PASS | 🟢 PASS | 1.2 |
| ST-17 | Multi-class target disables QML | High | PASS | 🟢 PASS | 0.8 |
| ST-18 | Back to binary re-enables QML | Med | PASS | 🟢 PASS | 0.9 |
| ST-19 | Model grid defaults, QML note, no-model block | Med | PASS | 🟢 PASS | 1.3 |
| ST-20 | Advanced panel defaults | Med | PASS | 🟢 PASS | 0.2 |
| ST-21 | Qubit boundaries (2 to 8) | High | PASS | 🟢 PASS | 0.5 |
| ST-22 | DNN epoch boundaries (1 to 500) | High | PASS | 🟢 PASS | 0.4 |
| ST-23 | QML epoch boundaries (1 to 100) | Med | PASS | 🟢 PASS | 0.4 |
| ST-24 | Fixed threshold boundaries (0.01 to 0.99) | High | PASS | 🟢 PASS | 0.7 |
| ST-25 | Seed validation and test split choices | Med | PASS | 🟢 PASS | 0.6 |
| ST-26 | Train all 8 models (binary) | High | PASS | 🟢 PASS | 11.1 |
| ST-27 | Train 7 models (multi-class) | High | PASS | 🟢 PASS | 9.1 |
| ST-28 | Seed reproducibility | High | PASS | 🟢 PASS | 11.5 |
| ST-29 | DNN epoch limit and early stopping | High | UNVERIFIED (manual review) | 🟢 PASS | 14.5 |
| ST-30 | QML settings applied and reported | High | UNVERIFIED (manual review) | 🟢 PASS | 11.0 |
| ST-31 | QML row cap 2,000; default QML run time | High | PASS | 🟢 PASS | 29.2 |
| ST-32 | Threshold modes reported | High | PASS | 🟢 PASS | 9.0 |
| **ST-33** | **Class balancing applied and reported** | Med | 🔴 **FAIL** (DEF-01) | 🔴 **FAIL (DEF-01)** | 8.2 |
| ST-34 | SVM subsampling on a large dataset | Med | UNVERIFIED (manual review) | 🟢 PASS (observation → DEF-02) | 19.0 |
| ST-35 | Compute device reported per model | Low | NOT RUN (manual at the time) | 🟢 PASS | 38.2 |
| **ST-36** | **Progress stages, checklist, log and timer** | High | UNVERIFIED (manual review) | 🔴 **UNVERIFIED** | 47.8 |
| ST-37 | Cancel during the DNN keeps finished results | High | PASS | 🟢 PASS | 5.0 |
| ST-38 | Cancel before any model finishes | Med | PASS | 🟢 PASS | 3.7 |
| ST-39 | One training job per browser | High | PASS | 🟢 PASS | 15.0 |
| ST-40 | Winner banner and paradigm cards | High | PASS | 🟢 PASS | 0.3 |
| ST-41 | Metrics table sorting and highlight | Med | PASS | 🟢 PASS | 4.7 |
| ST-42 | Charts render and degrade without Chart.js | Med | UNVERIFIED (manual review) | 🟢 PASS | 4.5 |
| ST-43 | Confusion matrices and metadata | High | PASS | 🟢 PASS | 0.4 |
| ST-44 | Export metrics as CSV | High | PASS | 🟢 PASS | 16.2 |
| ST-45 | Export safe for a formula-like name | High | UNVERIFIED (manual review) | 🟢 PASS | 7.5 |
| ST-46 | Invalid API requests return safe JSON errors | High | PASS | 🟢 PASS | 0.6 |
| ST-47 | Server unreachable / dataset expired messages | Med | PASS | 🟢 PASS | 16.4 |
| ST-48 | Keyboard-only flow and accessibility | Med | UNVERIFIED (manual review) | 🟢 PASS | 5.1 |
| ST-49 | Responsive layout and browsers | Med | PASS | 🟢 PASS (observation → DEF-03) | 56.9 |
| **ST-50** | **API response times** | High | PASS (warm server, train 202 = 15 ms) | 🔴 **FAIL (DEF-04)**, train 202 = 1,043 ms | 18.7 |

*Run 1 "UNVERIFIED (manual review)" rows were left blank in the run 1 record, as the status rules require. Between the runs the scripts were extended (section 5.3), not the application.*

### 7.4 Recorded failure messages (run 2, verbatim)

> [!CAUTION]
> **ST-33:** `Failed: Step 2 failed: after '← Change models', unticking 'Balance class weights' left Start training disabled, so the run could not be started as specified. Class balancing itself was reported correctly after a workaround (toggling a model).`
>
> **ST-50:** `AssertionError: ['train 202: 1043 ms (target < 200 ms)']`
>
> **Run 3, ST-50:** `AssertionError: ['train 202: 5641 ms (target < 200 ms)']`
>
> **ST-36, manual review:** `The UI refreshes every 2 s, so short stages can be skipped. Never rendered on screen: 'Testing <model> (i of n)'. Confirm by watching a run.`

### 7.5 Selected observations from run 2 (from the test notes)

| Case | Observation |
|---|---|
| ST-05 | TD-14 generated at 105,906,989 bytes; rejected in the browser |
| ST-12 / ST-43 | Train / validation / test rows 128 / 32 / 40; split fingerprint `ba7ba92f3866e20f`; PCA variance retained 68.7% |
| ST-21 – ST-25 | Every out-of-range value showed the documented message and disabled Start training. Examples: qubits `1`, `9`, `4.5`; DNN epochs `0`, `501`; threshold `0`, `1`, empty; seed `-1`, `2.5`, empty |
| ST-26 | 8 models on Banking: elapsed timer 8 s, wall clock 9.1 s |
| ST-37 | Results opened 2.5 s after Cancel |
| ST-46 | Unknown job → 404; `POST /api/samples/..%2F..%2Fconfig` → 405; unknown route → 404; expired dataset → 404; cancel a finished job → 409. All bodies are JSON `{"error": …}` |
| ST-49 | Chrome 156, Edge 157 and Chromium 158 at 1366×768 and 375×812: all `ok` |

---

## 8. Screenshots, evidence and generated reports

### 8.1 Contents of each run folder

| File | Content |
|---|---|
| `report.html` | pytest-html report with notes, manual-review items and step screenshots |
| `results.csv` | One row per case: outcome, duration, message, manual review pending, screenshot count |
| `junit.xml` | Machine-readable results |
| `screenshots/<test>/` | Step screenshots and attached text notes |
| `playwright/` | Trace (`.zip`) and screenshot of each failed browser test (`playwright show-trace <zip>`) |
| `app-server.log` | Console log of the application under test |
| `run-info.txt` | Commit, machine, browsers, package versions, start and end time, pytest exit code |
| `Test_Execution_Record_<run>.xlsx` | Execution record filled from `results.csv` (runs 1, 2 and 4; the run 1 copy is local only) |
| `uploads/`, `assets/` | Uploads written by the app under test; report assets |

### 8.2 Evidence volume

| Run | Files in `screenshots/` | `report.html` | `app-server.log` | Failure traces |
|---|---|---|---|---|
| Run 1 (*Local-only historical evidence; not included in GitHub submission.*) | 95 | 110,781 bytes | 155,776 bytes | ST-33 |
| Run 2 | 103 | 114,960 bytes | 165,219 bytes | ST-33 (`playwright/test-training-py-test-st33-class-balancing-reported-chromium/`) |
| Run 3 | 1 | 31,309 bytes | 11,444 bytes | none (ST-50 is not a browser test) |

### 8.3 Key evidence files

| Topic | Path (under `testing-docs/final/tests/runs/`) |
|---|---|
| DEF-01 | `20261110-132648-4988037/screenshots/test_st33_class_balancing_reported[chromium]/02-start-after-change-models.png`, plus the trace in `playwright/` |
| DEF-02 | `20261110-132648-4988037/screenshots/test_st34_svm_subsampling_on_large_dataset[chromium]/03-td15-warning.png`, `04-td15-results.png`, `observation-REQ-MS-4.txt` |
| DEF-03 | `…/screenshots/test_st49_responsive_layout_and_browser_compatibility/02-chrome-375-results.png`, `04-msedge-375-results.png`, `06-chromium-375-results.png` |
| DEF-04 | `20261110-132648-4988037/…/test_st50_api_response_times/measure_times-output.txt` and the same file in `20261110-133444-4988037` |
| Developer suite | `testing-docs/final/evidence/developer-pytest-not-slow.txt`, `developer-pytest-slow.txt`, `.coverage-not-slow` |

No screenshot, result or approval in any deliverable was produced by hand. All of them come from the files above.

---

## 9. Defects

Log: `testing-docs/final/Defect_Log.xlsx`. Detail and the developer response form: `testing-docs/final/Developer_Review.docx`. Severity follows STP 8.7: Critical, Major (a feature fails, but a workaround exists), Minor (cosmetic or usability).

| ID | Case | Severity | Priority | Summary | Assigned to | Status |
|---|---|---|---|---|---|---|
| <span style="color:#c00000">**DEF-01**</span> | ST-33 | **Major** | Medium | Start training stays disabled after "← Change models" when only *Balance class weights* (or *Test split*) changes | `[DEV CONFIRM]` frontend owner | 🟠 **DEVELOPER REVIEW** |
| DEF-02 | ST-34 (observation) | Minor | Low | SVM subsampling warning shown for 25,000 rows, although SVM trains on all training rows | Aaditya Vijaywargi (SVM subsampling, README) | 🟠 **DEVELOPER REVIEW** |
| DEF-03 | ST-49 (observation) | Minor | Low | At 375 px, the round "Best" stamp covers part of the winner-banner heading (691 px²) | `[DEV CONFIRM]` frontend owner | 🟠 **DEVELOPER REVIEW** |
| <span style="color:#c00000">**DEF-04**</span> | ST-50 | **Major** | High | `POST /api/train` takes seconds, not < 200 ms, on the first job after a server start that selects the DNN or VQC | Anish Kumar (API layer, README) | 🟠 **DEVELOPER REVIEW** |

### 9.1 <span style="color:#c00000">DEF-01: Start training disabled after "← Change models"</span>

- **Steps:**
  1. Load the Banking sample → **Continue to configure**, and tick only Logistic Regression.
  2. **Start training**, and wait for the results.
  3. Click **← Change models**.
  4. Open Advanced settings and untick *Balance class weights*.
  5. Try **Start training**.
- **Expected:** Start training is enabled, and a second run starts with class balancing off.
- **Actual:** <span style="color:#c00000">Start training stays disabled. It is enabled only after another input with a change handler is used (for example, toggling a model).</span>
- **Reproduced:** runs 1 and 2.
- **Suspected cause (code reading only):**
  - `onStartTraining` in `static/js/config.js` disables the button and never re-enables it after a successful start.
  - **← Change models** (`static/js/results.js`) doesn't refresh it.
  - `#opt-balance` and `#opt-test-ratio` have no change listener.
- **Needs an application change** → the developers decide. The tester did not change any code.

### 9.2 DEF-02: SVM warning and subsampling disagree (REQ-MS-4)

- **Input:** TD-15, 25,000 rows.
- **Hint shown:** `⚠ 25,000 rows: SVM will train on a stratified sample of 20,000 rows.`
- **Actual SVM details:** `Class weights balanced · Device CPU`, with no "Subsampled" entry. SVM trained on all 16,000 training rows.
- **Cause (code reading):**
  - The warning counts all rows (`static/js/config.js`).
  - Subsampling applies to training rows after the test and validation splits (`core/trainers.py`).
- **On 40,000 rows (ST-34 proper):** `Subsampled: from 25600 rows`, `Training rows: 20000` → the case PASSED.
- **Decision needed:** align the warning or the requirement (Q2).

### 9.3 DEF-03: "Best" stamp overlaps the heading at 375 px

- **Measured:** 691 px² overlap of the stamp with the text of the heading "Best model · ranked by PR-AUC, then F1", in Chrome, Edge and Chromium at 375×812. No overlap at 1366×768.
- **Effect:** cosmetic; it doesn't affect results. It isn't part of ST-49's expected result, so ST-49 passed.

### 9.4 <span style="color:#c00000">DEF-04: `/api/train` is slow on a fresh server (PERF-1)</span>

| Measurement | `POST /api/train` → 202 | Target |
|---|---|---|
| Run 1 (server had already trained a DNN and VQC) | 15 ms | < 200 ms |
| <span style="color:#c00000">Run 2</span> | <span style="color:#c00000">**1,043 ms**</span> | < 200 ms |
| <span style="color:#c00000">Run 3 (fresh server, ST-50 only)</span> | <span style="color:#c00000">**5,641 ms**</span> | < 200 ms |
| `docs/performance.md` (developers, warm server) | 15 ms | < 200 ms |

- **Suspected cause (code reading, not instrumented):** `/api/train` calls `build_trainers()` before returning 202. That function imports `core.dnn` (TensorFlow/Keras) and `core.qml` (PennyLane) inside the request, on first use.
- **Needs an application change** → the developers decide.

---

## 10. ML, DL and QML performance-tuning verification

Criteria: STP section 8.11 (P1 to P14). Measurements: run 2.

| Check | Criterion | Measured (run 2) | Result |
|---|---|---|---|
| P1 Split reproducibility (ST-28) | Two seed-42 fingerprints identical; seed 7 differs | 42 → `ba7ba92f3866e20f` twice; 7 → `e5a92d58be4a5d2d`; rows (128, 32, 40, 20) | 🟢 PASS |
| P2 Advanced panel limits (ST-20 – ST-25) | In range accepted; out of range rejected with the documented message | All boundary values behaved as specified | 🟢 PASS |
| P3 DNN epoch limit (ST-29) | Epochs run ≤ 5; "Stopped early" consistent | Epochs run 5, best epoch 5, stopped early `no`, 12,289 parameters | 🟢 PASS |
| P4 QML settings (ST-30) | Qubits 2, epochs 3, training rows ≤ 2,000 | Qubits 2, epochs 3, layers 2, 12 trainable parameters, 128 training rows, `default.qubit` | 🟢 PASS |
| P5 QML row cap (ST-31) | Training rows 2,000; test rows 8,000 | Training rows 2000 (subsampled), test rows 8000 | 🟢 PASS |
| P6 Threshold modes (ST-32) | Auto within 0.01 – 0.99; fixed = 0.500 | Auto: Logistic Regression 0.328, XGBoost 0.574; fixed 0.500 | 🟢 PASS |
| P7 Class balancing (ST-33) | "balanced", then "none" | Reported correctly, **but only after a workaround** | 🔴 **FAIL (DEF-01)** |
| P8 Winner ranking (ST-40, ST-27) | Top PR-AUC (binary) / top F1 (multi-class) | Winner Deep Neural Network (PR-AUC 0.935, F1 80.0%) | 🟢 PASS |
| P9 SVM subsampling (ST-34) | Training rows 20,000 on 40,000 rows | Subsampled from 25,600 rows → 20,000 | 🟢 PASS (boundary → DEF-02) |
| P10 Default QML run time (ST-31) | ≤ 180 s | 25.28 s (4 qubits, 20 epochs, 2,000 rows) | 🟢 PASS |
| P10 All 8 models on Banking (ST-26) | < 120 s | 8 s elapsed timer (9.1 s wall clock) | 🟢 PASS |
| **P11 `POST /api/train` (ST-50)** | < 200 ms | <span style="color:#c00000">**1,043 ms** (fresh-server rerun: 5,641 ms)</span> | 🔴 **FAIL (DEF-04)** |
| P11 status, configure, results, upload (ST-50) | < 200 / 500 / 500 ms / 2 s | Status median 15 ms (max 88 over 30 polls); results 16 ms; configure medians 15 / 32 / 51 ms; upload medians 47 / 62 / 111 ms (200 rows / 10,000 rows / 50,000 rows) | 🟢 Met |
| P12 Device reporting (ST-35) | XGBoost "GPU (CUDA)" with `auto`; all "CPU" with `cpu` | `auto`: XGBoost GPU (CUDA), Logistic Regression and DNN CPU. `cpu`: all CPU. `nvidia-smi`: RTX 3050 Laptop GPU, 566.07 | 🟢 PASS |
| P13 Model quality baseline (ST-26) | Recorded only | See table below | Recorded |
| P14 Better than random `[PROPOSED]` | Every binary ROC-AUC > 0.5 on Banking | Lowest ROC-AUC 0.675 (VQC) | 🟢 Met (criterion awaiting faculty confirmation) |

**Per-model training time in the ST-50 job (run 2):** Random Forest 0.4 s, SVM 6.2 s, DNN 3.3 s, VQC 3.6 s; the whole job took 15.0 s.

### 10.1 Observed scores, Banking sample (run 2, ST-26; baseline only, not pass/fail)

| Model | Paradigm | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Deep Neural Network** (winner) | DL | 82.5% | 80.0% | 0.930 | **0.935** |
| Logistic Regression | ML | 80.0% | 78.9% | 0.913 | 0.914 |
| SVM | ML | 72.5% | 74.4% | 0.885 | 0.911 |
| Random Forest | ML | 82.5% | 81.1% | 0.872 | 0.904 |
| Naive Bayes | ML | 82.5% | 82.1% | 0.925 | 0.903 |
| XGBoost | ML | 67.5% | 64.9% | 0.800 | 0.840 |
| Variational Quantum Classifier | QML | 60.0% | 68.0% | 0.675 | 0.730 |
| Decision Tree | ML | 72.5% | 74.4% | 0.725 | 0.657 |

The test set is 40 rows (positive rate 50.0%), so these scores are a baseline, not a quality claim (STP section 7).

### 10.2 UI/UX verification

| Area | Evidence | Result |
|---|---|---|
| Keyboard-only flow (ST-48) | 52 focus stops; none without an outline; contrast 6.09:1 to 6.64:1 (≥ 3:1 required); width ≥ 2 px | 🟢 PASS |
| Charts (ST-42) | Paradigm colours ML `#4c72b0`, DL `#c94f6d`, QML `#8172b3`; every curve labelled; one dashed random baseline in ROC and PR; horizontal log-axis time chart; safe fallback without Chart.js | 🟢 PASS |
| Responsive layout and browsers (ST-49) | 3 browsers × 2 widths, all `ok` | 🟢 PASS; 🟠 DEF-03 observation |
| Export safety (ST-45) | Opened in Excel: 28 cells, 0 formulas, 0 errors; download name `TD-13_SUM11_model_comparison.csv` | 🟢 PASS |
| Progress screen (ST-36) | 6 distinct stages seen; status poll median gap 2.02 s; "Testing <model> (i of n)" never drawn | 🔴 **UNVERIFIED** |
| Re-run after results (ST-33) | Start training stays disabled | 🔴 **FAIL (DEF-01)** |

---

## 11. Manual verification and pending activities

Guide: `testing-docs/final/Manual_Verification_Guide.md`.

After run 1, eight cases needed a person: ST-29, ST-30, ST-34, ST-35, ST-36, ST-42, ST-45 and ST-48. **Seven of them were then automated with measurable criteria and passed in run 2.** One remains open.

| Case | Status | What is needed |
|---|---|---|
| <span style="color:#c00000">**ST-36**</span> | 🔴 **UNVERIFIED** / 🟡 PENDING | See the steps below, after the developers answer **Q9** |

**ST-36 manual check:**
1. At the repository root: `$env:COMPUTE_DEVICE = "cpu"; .venv\Scripts\python.exe app.py`, then open `http://127.0.0.1:5000` in Chrome.
2. Upload `testing-docs\final\test-data\TD-12_large_40000.csv` → **Continue to configure →** → target **Exited**.
3. Tick Random Forest, SVM and Deep Neural Network → **Start training**.
4. **Expected:**
   - The label shows `Testing <model> (i of 3)` at least once.
   - The checklist icons go ○ → ◐ → ◑ → ✓.
   - The log has one "… tested in …" line per model.
5. **Evidence:** a screenshot saved as `tests\runs\manual-<date>\ST-36_testing-stage.png`.
6. **Record:** set **PASS** in `Test_Execution_Record_Final.xlsx` only if the tester sees the label. Otherwise leave it UNVERIFIED and follow the developers' answer to Q9.

---

## 12. Developer review requirements and open decisions

Document: `testing-docs/final/Developer_Review.docx` (FSPP-DR01, 8 pages). It contains, for each defect:
- reproduction steps;
- expected and actual results, with the actual result in red;
- evidence and embedded screenshots;
- the suspected cause;
- the requested action;
- the retest command;
- response rows for the developers.

It ends with a sign-off table.

### 12.1 Questions for the developers (`[DEV CONFIRM]`)

| # | Reference | Question | Status |
|---|---|---|---|
| Q1 | SRS 5.3 SEC-6 | SEC-6 promises "model timeouts"; the code has none (only Gunicorn's 600 s request timeout). Remove the words or implement timeouts? | 🟠 DEVELOPER REVIEW |
| Q2 | SRS 4.3 REQ-MS-4 | Align the SVM warning and the subsampling rule (DEF-02) | 🟠 DEVELOPER REVIEW |
| Q3 | SRS 4.4 – 4.7 | Confirm the IDs and wording of REQ-PP-11 to 13, REQ-TR-9 and REQ-EV-6 | 🟠 DEVELOPER REVIEW |
| Q4 | SRS 5.1 PERF-8 | Keep "all 8 models on Banking under 2 minutes" as a requirement? (Measured: 8 s) | 🟠 DEVELOPER REVIEW |
| Q5 | SRS 4.7 REQ-EV-4 | `docs/deviations.md` item 8 says accuracy then F1; the code ranks binary targets by PR-AUC then F1 | 🟠 DEVELOPER REVIEW |
| Q6 | SRS 2.6 | The README has no dashboard screenshot, although the SRS lists one | 🟠 DEVELOPER REVIEW |
| Q7 | STP 14, 18 | Who owns frontend defects? Who are the development lead and the project manager? | 🟠 DEVELOPER REVIEW |
| Q8 | STP 12 | Confirm the owners of the tasks marked done | 🟠 DEVELOPER REVIEW |
| Q9 | SRS REQ-JB-5, COM-2 | Stages shorter than the 2 s refresh are never drawn ("Testing <model> (i of n)" never appeared, even on 40,000 rows). Is a skipped label acceptable, or should every stage be shown? | 🟠 DEVELOPER REVIEW |

### 12.2 Decisions needed

| Decision | Owner |
|---|---|
| Fix DEF-01 and DEF-04, or accept them as known limitations | Developers, then faculty |
| DEF-02: change the warning or the requirement | Developers |
| DEF-03: fix or defer (cosmetic) | Developers |
| Name the frontend owner, development lead and project manager | Developers |

---

## 13. Deliverables and locations

All final deliverables are in `testing-docs/final/`. The previous final versions are kept in `testing-docs/final/archive/` (*Local-only historical evidence; not included in GitHub submission.*) Submission copies of the final requirements and plan are at the repository root as **`SRS.docx`** (= SRS v1.2) and **`STP.docx`** (= STP v2.2, IEEE 829); they are byte-identical to the versioned files.

### 13.1 The five faculty deliverables

| # | Deliverable | File(s) | Status |
|---|---|---|---|
| 1 | Test Plan | `STP_Full_Stack_Prediction_Platform_v2.2.docx` (FSPP-MTP02.0 v2.2, 23 pages, IEEE 829 sections 1–19 plus appendices A–E; correction log T1–T39) | 🟢 COMPLETED; 🟡 approvals `[FILL LATER]` |
| 2 | Test Cases | `System_Test_Cases.xlsx` (50 cases, revision 1.3, Actual Result and PASS/FAIL from run 4) | 🟢 COMPLETED |
| 3 | Test Scripts | `tests/` (11 test modules, framework, `run_tests.ps1`, README) | 🟢 COMPLETED: 50 of 50 scripted |
| 4 | Automated Execution Evidence | `tests/runs/` runs 2–5 (run 1 is local only; final: **`20261110-225957-4988037-modified`**); `evidence/` | 🟢 COMPLETED |
| 5 | Test Summary Report | `Test_Summary_Report_Final.docx` (FSPP-TSR01.0 **version 1.1**, 10 pages, for run 4) | 🟢 COMPLETED; 🟠 fixes await developer review; 🟡 acceptance and approvals pending |

### 13.2 Supporting files

| File | Content | Status |
|---|---|---|
| `SRS_Full_Stack_Prediction_Platform_v1.2.docx` | Corrected requirements (S1–S25); REQ-MS-4 restated on SVM training rows | 🟢 COMPLETED; 🟠 `[DEV CONFIRM]` items open |
| `Test_Automation_Mapping.xlsx` | 50 cases → automation approach; run 4 status: 50 passing | 🟢 COMPLETED |
| `Test_Execution_Record_Final.xlsx` | Run 4 statuses (50 PASS); retest columns for ST-33, 34, 49, 50; Run History of every execution | 🟢 COMPLETED; 🟡 evaluator selection column empty |
| `Test_Execution_Record.xlsx` | Blank template for later cycles | 🟢 COMPLETED |
| `Defect_Log.xlsx` | DEF-01 to DEF-04: status **Fixed**, fix description, retest date and result (PASS) | 🟢 COMPLETED; 🟠 Closed after developer commit |
| `Developer_Review.docx` | FSPP-DR01 **version 1.1**: defects, section 3A (fixes to review), Q1–Q9, sign-off | 🟢 COMPLETED; 🟠 waiting for responses |
| `Test_Summary_Report_Template.docx` | Blank TSR template | 🟢 COMPLETED |
| `Manual_Verification_Guide.md` | No manual check open; optional ST-36 observation steps; people actions | 🟢 COMPLETED |
| `test-data/` | `make_test_data.py`, TD-01 to TD-13 | 🟢 COMPLETED |
| `evidence/` | Developer pytest output before and after the fixes | 🟢 COMPLETED |
| `archive/` | STP v2.1, SRS v1.1, TSR for run 2, Developer Review v1.0, execution record for run 2 | 🟢 Kept for history. *Local-only historical evidence; not included in GitHub submission.* |

### 13.3 Inconsistencies found earlier: resolved

| File | Was | Now |
|---|---|---|
| `Defect_Log.xlsx`, subtitle | "no defects recorded yet" | 4 defects recorded; fixed and retested |
| `Test_Automation_Mapping.xlsx`, subtitle | "prepared, not implemented" | Implemented: 50 of 50 cases scripted |
| `System_Test_Cases.xlsx`, Cover "Test plan" | "STP FSPP-MTP02.0 v2.0" | v2.2 |
| STP, Appendix A title | "Correction Log (v1.0 to v2.0)" | "Correction Log (v1.0 to v2.2)" |

---

## 14. Remaining work, retesting procedure and acceptance status

### 14.1 Final checklist

| Item | Status |
|---|---|
| Repository analysis and gap analysis | 🟢 COMPLETED |
| SRS v1.2 and STP v2.2 (IEEE 829 structure kept) | 🟢 COMPLETED |
| 50 system test cases with traceability, test data and recorded results | 🟢 COMPLETED |
| Playwright automation of all 50 cases | 🟢 COMPLETED |
| Runs 1–3 on the original build; developer suite (231 + 1 passed) | 🟢 COMPLETED |
| DEF-01 to DEF-04 fixed (3 application files, 1 test file) | 🟢 COMPLETED → 🟠 DEVELOPER REVIEW (approve and commit) |
| Regression run 4: 50 / 50 PASS; run 5: DEF-04 fresh-server retest PASS (18 ms) | 🟢 COMPLETED |
| Developer suite after the fixes: 234 + 1 passed, 96% | 🟢 COMPLETED |
| ST-36 "Testing" label | 🟢 PASS in run 4 (timing-dependent) → 🟠 Q9 |
| Q1 – Q9 | 🟠 DEVELOPER REVIEW |
| <span style="color:#c00000">Lint (flake8, pylint, black) on the changed files</span> | 🔴 **NOT RUN**: tools not installed in the project `.venv`; CI will run them on the commit |
| <span style="color:#c00000">Rerun on the committed fix SHA (run folders currently end in `-modified`)</span> | 🟡 PENDING (needs the commit) |
| <span style="color:#c00000">Evaluator selection of about 30 cases and the acceptance run</span> | 🟡 PENDING |
| Names, USN, institution, faculty guide, signatures | 🟡 PENDING (`[FILL LATER]`) |

### 14.2 Who acts next

**Developers:**
1. Review `git diff main` on branch `qa/defect-fixes-final-testing` (section 15). Accept and commit the changes, or replace them with your own fix (`Developer_Review.docx`, section 3A).
2. Answer Q1–Q9, Q9 in particular: is a timing-dependent "Testing" label acceptable?
3. Name the frontend owner, the development lead and the project manager.

**Tester (Ravi Teja):**
1. After the commit, rerun on the commit SHA (14.3) and set DEF-01 to DEF-04 to **Closed** in `Defect_Log.xlsx`.
2. Fill the `[FILL LATER]` fields and sign the Test Summary Report.

**Faculty / evaluator:**
1. Confirm the proposed criteria: 90% pass rate, at least 30 automated cases, ROC-AUC > 0.5.
2. Select about 30 cases (column *Selected for Acceptance* in `Test_Execution_Record_Final.xlsx`) and attend the acceptance run.
3. Sign the approvals in STP section 18 and in the Test Summary Report.

### 14.3 Retesting procedure (STP 8.8, "Retesting After Defect Fixes")

1. The developers commit the fix and record the commit SHA in `Defect_Log.xlsx` (*Fix Commit*).
2. Check out that commit. Check that `git status` shows no changes to tracked files, so the run folder carries the plain SHA without `-modified`.
3. Run the full regression and the fresh-server ST-50 check:

   ```powershell
   cd testing-docs\final\tests
   .\run_tests.ps1                 # all 50 cases
   .\run_tests.ps1 -Select st50    # DEF-04 on a freshly started server
   ..\..\..\.venv\Scripts\python.exe fill_execution_record.py runs\<run-folder>
   ```

4. If everything passes, set each defect to **Closed**; if anything fails, set it to **Reopened**. Reissue the Test Summary Report. Earlier run folders stay as history.

### 14.4 Acceptance status

| Criterion (STP 9) | Current value | Met? |
|---|---|---|
| All 50 cases executed | 50 / 50 (run 4) | 🟢 Yes |
| Pass rate ≥ `[PROPOSED: 90%]` | 100% (50 / 50) | 🟢 Yes (target awaiting faculty confirmation) |
| At least 30 automated cases `[PROPOSED]` | 50 | 🟢 Yes |
| No open Critical or Major defect | DEF-01 and DEF-04 fixed and retested, but not Closed until committed | 🟠 **Pending developer commit** |
| No UNVERIFIED case | None in run 4 | 🟢 Yes |
| Execution on a committed build | Runs 4 and 5 used uncommitted changes (`-modified`) | 🟡 Pending |
| Evaluator selection run and approvals | Not done | 🟡 Pending |

> [!IMPORTANT]
> **Final QA acceptance status: <span style="color:#C55A11">ready for the evaluator's acceptance run after the developers commit the fixes and the regression is repeated on that commit</span>.**
> The build is <span style="color:#c00000">**not accepted yet**</span>. This is **not** "100% complete": the developer approval, the acceptance run, the approvals and the lint check are still outstanding.

---

## 15. Defect fixes and regression cycle

### 15.1 Why the application was changed

During runs 1–3 the tester did not modify application code. Afterwards the project owner explicitly asked for the defects to be fixed, on a separate Git branch with every change documented and nothing committed without permission. Branch: **`qa/defect-fixes-final-testing`**, created from `main` at `4988037`. **Nothing is committed, pushed or merged.**

### 15.2 Files changed

| File | Defect | Change | Lines |
|---|---|---|---|
| `static/js/config.js` | DEF-01 | `updateTrainButton()` runs whenever the configure screen is shown (`onStateChange`). Change listeners added to `#opt-balance` and `#opt-test-ratio`. | +40 / −5 (with DEF-02) |
| `static/js/config.js` | DEF-02 | The warning is shown only when the **SVM training rows** exceed 20,000. Training rows = rows − ⌈rows × test split⌉ − ⌈20% validation⌉, the same rule as `core/preprocessing.py` and `core/trainers.py`. The hint `#svm-note` updates when the test split changes. Wording is unchanged. | (as above) |
| `static/css/styles.css` | DEF-03 | At ≤ 768 px: `.winner-kicker { display: block; padding-right: 52px; }`, matching the existing `.winner-name` rule. | +2 |
| `core/trainers.py` | DEF-04 | `build_one()` returns a `DeferredTrainer` for `dnn` and `vqc`. Key, name and paradigm come from `DEFERRED_MODELS`; the real trainer, with its TensorFlow or PennyLane import, is built on first attribute access in the job thread. Attributes pass through, so callers and existing tests are unchanged. | +65 / −11 |
| `tests/test_trainers.py` | DEF-04 | 3 new developer tests: trainers built only on first use; names match the real classes; a fresh interpreter imports no TensorFlow, Keras or PennyLane when building trainers. | +45 / −1 |
| `.gitignore` | cleanup | Keep each run's `app-server.log`; ignore `runs/dev-check/`, `.coverage-*` and Office lock files `~$*`. | +11 |

The exact line counts are in `git diff --stat` (see `FINAL_COMPLETION_REPORT.md`).

### 15.3 ST-36: investigated, not changed

- **Server side:** the server publishes `Testing <model> (i of n)` correctly; 0.1 s polling saw it in runs 2 and 4.
- **UI side:** the UI shows the latest poll every 2 s, as COM-2 requires. A testing step shorter than 2 s may therefore be skipped.
- **Observed:** the label was drawn in run 4 but not in runs 1 and 2.

This is a conflict between REQ-JB-5 and COM-2, not a coding error, so no code was changed. ST-36 passed in run 4; Q9 asks the developers whether a timing-dependent label is acceptable.

### 15.4 Regression and retest results

| Execution | Folder / file | Result |
|---|---|---|
| Developer suite after fixes (not slow) | `evidence/developer-pytest-not-slow-after-fixes.txt` | 🟢 **234 passed**, 1 deselected; coverage 96% (1,165 statements, 45 missed) |
| Developer suite after fixes (slow) | `evidence/developer-pytest-slow-after-fixes.txt` | 🟢 **1 passed** |
| **Run 4**, 22:59:57 – 23:10:57, all 50 | `tests/runs/20261110-225957-4988037-modified/` | 🟢 **50 passed, 0 failed, 0 manual reviews pending** |
| **Run 5**, 23:11:35, ST-50 on a fresh server | `tests/runs/20261110-231135-4988037-modified/` | 🟢 **PASS**: train 202 = **18 ms** |

| Defect | Before | After |
|---|---|---|
| DEF-01 (ST-33) | 🔴 FAIL in runs 1 and 2 | 🟢 PASS: Start training enabled after "← Change models"; the second run reports class weights "none" |
| DEF-02 (ST-34) | Warning at 25,000 rows, no subsampling | 🟢 25,000 rows: no warning, no subsampling. 40,000 rows: warning shown, 20,000 of 25,600 rows used |
| DEF-03 (ST-49) | Overlap of 691 px² at 375 px | 🟢 0 px² in Chrome 157, Edge 157 and Chromium 158 at 375 and 1366 px |
| DEF-04 (ST-50) | 🔴 1,043 ms (run 2); 5,641 ms on a fresh server (run 3) | 🟢 20 ms (run 4); 18 ms on a fresh server (run 5) |

### 15.5 Performance after the fixes (run 4, ST-50, `tools/measure_times.py`)

| Endpoint | Target (SRS PERF-1, 2, 7) | Run 2 (before) | Run 4 (after) | Run 5 (fresh server) |
|---|---|---|---|---|
| `POST /api/upload` median (200 / 10k / 50k rows) | < 2 s up to 10 MB | 47 / 62 / 111 ms | 66 / 83 / 171 ms | 36 / 43 / 135 ms |
| `POST /api/configure` median | < 500 ms | 15 / 32 / 51 ms | 8 / 19 / 77 ms | 7 / 24 / 59 ms |
| `POST /api/train` → 202 | < 200 ms | <span style="color:#c00000">1,043 ms</span> | **20 ms** | **18 ms** |
| `GET /api/status` median (max) | < 200 ms | 15 ms (88) | 6 ms (164) | 9 ms (198) |
| `GET /api/results` | < 500 ms | 16 ms | 24 ms | 4 ms |

**ML/DL/QML tuning in run 4 (P1–P14):**
- **Identical to run 2:** split fingerprint `ba7ba92f3866e20f`; DNN epochs 5 with no early stop; VQC qubits 2, epochs 3; QML row cap 2,000 with 8,000 test rows; auto thresholds LR 0.328 and XGBoost 0.574; winner DNN with PR-AUC 0.935; lowest ROC-AUC 0.675. All P checks met.
- **Hardware:** `auto` → XGBoost GPU (CUDA); `cpu` → all CPU; RTX 3050, driver 566.07.
- **Run times:** run 4 was slower than run 2 on the same laptop. The default QML run took 47.90 s (25.28 s in run 2) and all 8 models took 17 s (8 s). Both are still well inside their targets (≤ 180 s and < 120 s), but the times are not a like-for-like comparison.
- **Browser:** Chrome updated itself to 157.0.8103.39 before run 4.

### 15.6 Cleanup

| Action | Detail |
|---|---|
| Removed | `__pycache__/` in the repository root, `core/`, `tests/` and `testing-docs/final/tests/`; two stale Word lock files (`~$…docx`) in `testing-docs/final/` |
| Kept | All run folders (including `runs/dev-check/`, now git-ignored), the earlier drafts, the root PDFs, `.env`, `uploads/`, test data, evidence |
| `.gitignore` | See 15.2 |

---

## Appendix A: Testing timeline

All dates are 10 November 2026 unless stated otherwise.

| # | Phase | Output |
|---|---|---|
| 1 | Analysis only: project, existing tests, SRS/STP, professor's requirements (no file changes) | Findings report |
| 2 | Comparison of SRS/STP against commit `4988037` (no file changes) | Correction checklist, approved by the tester |
| 3 | Corrected drafts | `testing-docs/SRS_…_v1.1.docx`, `testing-docs/STP_…_v2.0.docx` (*Local-only historical evidence; not included in GitHub submission.*) |
| 4 | Submission-ready documents | `testing-docs/final/`: SRS v1.1, STP v2.0, 50 test cases, automation mapping, defect log, execution record and TSR templates, test data |
| 5 | Playwright implementation plan, then implementation | `testing-docs/final/tests/` |
| 6 | **Run 1** (local only), 12:20:04 – 12:27:22 | 48 passed, 1 failed (ST-33), 1 skipped (ST-35); 7 manual reviews pending; DEF-01 raised |
| 7 | Manual verification guide for the 7 pending reviews and ST-35 | `Manual_Verification_Guide.md` (first version) |
| 8 | Scripts extended so the manual checks became measurable; test cases revision 1.1 and 1.2 | ST-29, 30, 34, 35, 42, 45, 48 automated |
| 9 | **Run 2**, 13:26:48 – 13:34:07 | 47 PASS, 2 FAIL, 1 UNVERIFIED; DEF-02, 03, 04 raised |
| 10 | **Run 3**, 13:34:44 – 13:35:18 (ST-50 on a fresh server) | 5,641 ms; DEF-04 confirmed |
| 11 | Developer suite, 13:35:41 – 13:36:50 | 231 + 1 passed, 96% coverage |
| 12 | Final documents | STP v2.1, `Developer_Review.docx`, `Test_Execution_Record_Final.xlsx`, `Test_Summary_Report_Final.docx`, updated guide |
| 13 | This record | `TESTING_IMPLEMENTATION.md` |
| 14 | Defect fixes at the project owner's request (branch `qa/defect-fixes-final-testing`, 22:30 – 22:55) | 3 application files, 1 test file changed; ST-36 investigated, not changed |
| 15 | Developer suite after fixes, 22:57 – 22:59 | 234 + 1 passed, 96% |
| 16 | **Run 4 (final, fixed build)**, 22:59:57 – 23:10:57 | 50 / 50 PASS |
| 17 | **Run 5**, 23:11:35 (ST-50 on a fresh server) | 18 ms; DEF-04 retest PASS |
| 18 | Documents updated | STP v2.2, SRS v1.2, test cases rev 1.3, TSR v1.1, Developer Review v1.1, workbooks, this record, `FINAL_COMPLETION_REPORT.md` |

## Appendix B: Integrity of originals and application code

Checked when this record was last updated (10 Nov 2026, after run 5):

| Item | Check | Result |
|---|---|---|
| Application source during runs 1–3 | `git diff --stat HEAD` | 🟢 Empty: no tracked file modified |
| Application source after the fix request | `git diff --stat` on branch `qa/defect-fixes-final-testing` | 🟠 Changed (uncommitted): `static/js/config.js`, `static/css/styles.css`, `core/trainers.py`, `tests/test_trainers.py`, `.gitignore`. `app.py`, `config.py`, `tools/` and the other `core/` modules are unchanged |
| `SRS_Full_Stack_Prediction_Platform (1).pdf` | MD5 prefix | 🟢 `ce3efb2bbedd`, unchanged. *Local-only historical evidence; not included in GitHub submission.* |
| `STP_Full_Stack_Prediction_Platform.pdf` | MD5 prefix | 🟢 `ba847c370873`, unchanged. *Local-only historical evidence; not included in GitHub submission.* |
| `testing-docs/SRS_Full_Stack_Prediction_Platform_v1.1.docx` (draft) | MD5 prefix | 🟢 `bbd844b4e5b8`, unchanged. *Local-only historical evidence; not included in GitHub submission.* |
| `testing-docs/STP_Full_Stack_Prediction_Platform_v2.0.docx` (draft) | MD5 prefix | 🟢 `a8c3af3207a8`, unchanged. *Local-only historical evidence; not included in GitHub submission.* |
| Run evidence folders | Not edited after execution | 🟢 Preserved |

---

*Prepared by Ravi Teja (Ravi S), Tester. Signature: [FILL LATER]. Date: [FILL LATER].*
