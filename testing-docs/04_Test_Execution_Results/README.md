# 04 Test Execution Results

Evidence of every test execution. Each automated run is one self-contained folder: its
`report.html` links to its own screenshots by relative path, so a run folder is always kept whole
and never split up.

The run folders sit directly in this folder, without an extra sub-folder level, to keep the
deepest screenshot paths within the Windows 260-character path limit. Git for Windows, Explorer
and Office cannot open longer paths by default.

| Folder | Content |
|---|---|
| `<date-time>-<commit>[-modified]\` | One folder per Playwright run, written by `..\02_Test_Automation\run_tests.ps1` |
| `dev-check\` | Framework debugging runs made while the scripts were written; **not execution evidence** (local only, git-ignored) |
| `Execution_Records\` | `Test_Execution_Record_Final.xlsx`, the formal execution record (run 4, with the run history for runs 1 to 5), and `Test_Execution_Record_Template.xlsx`, the blank template used by `fill_execution_record.py` |
| `Execution_Records\Previous_Versions\` | The execution record issued for run 2 (local only, git-ignored) |
| `Developer_Unit_Tests\` | Output of the developers' pytest suite (`tests\` at the repository root), before and after the defect fixes |

## Inside a run folder

| Evidence type | Where | Present in |
|---|---|---|
| HTML execution report (notes, manual-review items, step screenshots) | `report.html` | every run |
| Per-case outcome, duration, message, manual review pending | `results.csv` | every run |
| Machine-readable results | `junit.xml` | every run |
| **Screenshots** | `screenshots\<test name>\NN-<step>.png` | every run |
| **Logs**: console log of the application under test | `app-server.log` | every run |
| **Traces** of failed browser tests (`playwright show-trace trace.zip`) | `playwright\<test>\trace.zip` and `test-failed-1.png` | runs that had a failing browser test (runs 1 and 2: ST-33) |
| Environment: commit, browsers, package versions, start and end time, pytest exit code | `run-info.txt` | every run |
| Filled execution record for that run | `Test_Execution_Record_<run>.xlsx` | runs 1, 2, 4 and 7 (made by `fill_execution_record.py`) |

A folder name ending in `-modified` means the working tree had uncommitted changes to tracked
files when the run started.

## Run index

The counts below are taken from each run's own `results.csv` and `run-info.txt`. "Manual review"
means pytest passed but the script left a check for a person, so the case counts as
**UNVERIFIED**, not PASS (rule in `fill_execution_record.py`).

| Run | Folder | Build | Scope | Outcome from `results.csv` | Role |
|---|---|---|---|---|---|
| — | `dev-check\` | — | 6 cases | 6 passed | Framework debugging while the scripts were written. **Not execution evidence.** Local only (git-ignored). |
| 1 | `20261110-122004-4988037\` | `4988037` | all 50 | 48 passed, 1 failed (ST-33), 1 skipped (ST-35); 7 with manual review pending | First execution. Superseded by run 2 (same build, extended scripts). Local only (git-ignored). |
| 2 | `20261110-132648-4988037\` | `4988037` | all 50 | 48 passed, 2 failed (ST-33, ST-50); 1 with manual review pending (ST-36) → **47 PASS, 2 FAIL, 1 UNVERIFIED** | Original-build evidence for DEF-01 (ST-33 trace in `playwright\`) and DEF-04 |
| 3 | `20261110-133444-4988037\` | `4988037` | ST-50 | 1 failed (ST-50) | ST-50 on a fresh server: DEF-04 confirmation |
| 4 | `20261110-225957-4988037-modified\` | `4988037` + fixes | all 50 | **50 passed**, 0 failed, 0 manual review | **Final regression run** cited in the Test Summary Report v1.1 and the final execution record |
| 5 | `20261110-231135-4988037-modified\` | `4988037` + fixes | ST-50 | 1 passed | ST-50 retest on a fresh server (DEF-04 fixed) |
| 6 | `20261111-102718-4988037-modified\` | `4988037` + fixes | all 50 | 50 passed | Additional full run on 11 Nov 2026; not cited in the reports |
| 7 | `20261117-191808-61fc217-modified\` | `61fc217` (fixes committed) | all 50 | 49 passed, **1 failed (ST-45)** | Verification after the 17 Nov folder reorganisation. ST-45's application checks passed, but its Excel step failed with `Exception from HRESULT: 0xFFF40015`. The cause was path length: the run was written under an interim extra `Automated_Runs\` level, making the exported CSV's path 261 characters (it was 237 in the old layout). That level was removed and the run folder moved here afterwards; see run 8. |
| 8 | `20261117-193001-61fc217-modified\` | `61fc217` (fixes committed) | ST-07, ST-45 | 2 passed | ST-45 retest on the final layout (CSV path 246 characters): Excel opened the file, with 0 formula and 0 error cells. ST-07 is the demonstration case. |
| 9 | `20261117-194428-61fc217-modified\` | `61fc217` (fixes committed) | all 50 | **50 passed**, 0 failed, 0 manual review | Full 50-case regression run on the final 6-part folder layout. All 50 cases PASS. Confirms end-to-end stability and reproducibility across the entire system test suite. |

For runs 7, 8 and 9, `-modified` reflects only the uncommitted folder reorganisation of the testing
files. The application source was identical to commit `61fc217` (checked with `git diff`).

## Developer unit tests (`Developer_Unit_Tests\`)

| File | Build | Result (last lines of the file) |
|---|---|---|
| `developer-pytest-not-slow.txt` | `4988037` | 231 passed, 1 deselected; coverage 96% |
| `developer-pytest-slow.txt` | `4988037` | 1 passed |
| `developer-pytest-not-slow-after-fixes.txt` | with fixes | 234 passed, 1 deselected; coverage 96% |
| `developer-pytest-slow-after-fixes.txt` | with fixes | 1 passed |

The `.coverage-*` files beside them are the binary coverage databases (local only, git-ignored).
