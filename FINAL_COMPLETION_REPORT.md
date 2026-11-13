# Final Completion Report: System Testing and Release Preparation

**Full-Stack Prediction Platform (ML vs DL vs QML)**

| Field | Value |
|---|---|
| Prepared by | Ravi Teja (Ravi S), independent system tester |
| Date | 10 November 2026 |
| Base commit | `4988037` on `main` |
| Working branch | **`qa/defect-fixes-final-testing`**: all changes are **uncommitted**; nothing is pushed or merged |
| Full history | `TESTING_IMPLEMENTATION.md` (sections 1–15) |

> [!IMPORTANT]
> **Acceptance readiness: ready for the evaluator's acceptance run once the developers approve and commit the fixes. Not accepted, and not 100% complete.**
>
> - **Regression run 4 (fixed build):** <span style="color:#00704A">**50 / 50 PASS**</span>, with 0 FAIL and 0 UNVERIFIED.
> - **Defects:** all 4 fixed and retested PASS. <span style="color:#C55A11">**DEVELOPER REVIEW:**</span> the developers must review and commit the fixes before the defects can be Closed.
> - <span style="color:#c00000">**PENDING:**</span>
>   - a rerun on the committed SHA;
>   - lint of the changed files (tools not installed);
>   - the evaluator's acceptance run;
>   - approvals and the `[FILL LATER]` fields.

---

## 1. Work completed

1. **Inspection:** repository, Git status, remote branches, frontend, API and trainer code. `origin/fix/show-model-testing` is older than `main` and lacks Sprint 2, so it was not used.
2. **Branch:** `qa/defect-fixes-final-testing`, created from `main` at `4988037`.
3. **Defect fixes:** DEF-01, DEF-02, DEF-03 and DEF-04 fixed (section 3). ST-36 was investigated and found not to be a code defect (section 3.5).
4. **Regression:**
   - developer pytest suite, before and after the fixes;
   - all 50 Playwright system test cases (run 4);
   - a fresh-server ST-50 retest (run 5).
5. **Performance verification:** API response times; ML, DL and QML tuning checks P1–P14; CPU/GPU behaviour (section 5).
6. **Documentation:** every deliverable updated from the actual results, with IEEE 829 structure unchanged and previous finals archived (section 2).
7. **Cleanup:** caches and stale lock files removed; `.gitignore` updated (section 7).
8. **Final validation:** documents and workbooks opened, formulas, evidence paths, Git diff, secrets (section 8).

## 2. Files created and modified

### 2.1 Application and test code (tracked; uncommitted on the branch)

| File | Reason | +/− lines |
|---|---|---|
| `static/js/config.js` | DEF-01 and DEF-02 | +40 / −5 |
| `static/css/styles.css` | DEF-03 | +2 / −0 |
| `core/trainers.py` | DEF-04 | +65 / −11 |
| `tests/test_trainers.py` | DEF-04: 3 new developer tests | +45 / −1 |
| `.gitignore` | Cleanup rules (section 7) | +11 / −0 |

`app.py`, `config.py`, `tools/`, the other `core/` modules, `static/index.html` and the other `static/js/` modules were **not changed**.

### 2.2 Testing documents (untracked, in `testing-docs/final/` unless stated)

| File | Action |
|---|---|
| `STP_Full_Stack_Prediction_Platform_v2.2.docx` | **Created** from v2.1 with 28 targeted edits: version 2.2, revision row, SRS 1.2 references, 8.8 retest note, 8.10 run 4 result, sections 11, 12 and 16 statuses, Appendix A title and T36–T39 |
| `SRS_Full_Stack_Prediction_Platform_v1.2.docx` | **Created** from v1.1: REQ-MS-4 restated on SVM training rows (`[DEV CONFIRM]`), S25, revision 1.2, reference to STP 2.2 |
| `System_Test_Cases.xlsx` | **Modified:** revision 1.3; Actual Result and PASS/FAIL from run 4; cover rows corrected |
| `Test_Automation_Mapping.xlsx` | **Modified:** status from run 4 (50 passing); stale subtitle corrected |
| `Test_Execution_Record_Final.xlsx` | **Rebuilt** from run 4; retest columns; Run History covers runs 1–5 and both developer-suite cycles |
| `Defect_Log.xlsx` | **Modified:** DEF-01 to DEF-04 set to Fixed, with fix description, retest date and result PASS; stale subtitle corrected |
| `Test_Summary_Report_Final.docx` | **Rebuilt** as version 1.1 for run 4, with the same IEEE 829 sections 1–8 and Appendix A |
| `Developer_Review.docx` | **Rebuilt** as version 1.1, with new section 3A "Fixes Applied for Developer Review" and an accept/reject row |
| `Manual_Verification_Guide.md`, `tests/README.md` | **Modified:** no manual check open; ST-36 timing note |
| `evidence/developer-pytest-*-after-fixes.txt` | **Created** |
| `tests/runs/20261110-225957-4988037-modified/`, `tests/runs/20261110-231135-4988037-modified/` | **Created** by the test runs |
| `archive/` | **Created:** STP v2.1, SRS v1.1, the run 2 TSR, Developer Review v1.0 and the run 2 execution record, moved or copied here unchanged |
| `TESTING_IMPLEMENTATION.md` (root) | **Modified:** status box, sections 13–14 rewritten, new section 15, appendices A and B |
| `FINAL_COMPLETION_REPORT.md` (root) | **Created** (this file) |
| `SRS.docx`, `STP.docx` (repository root) | **Created:** submission copies of SRS v1.2 and STP v2.2, byte-identical to the versioned files in `testing-docs/final/` (MD5 `b0b1f7a20273…`, `0fa32b295a53…`). Edit the versioned files and re-copy, so the two stay identical |

**Preserved unchanged:**
- run folders 1–3 and `runs/dev-check/`;
- the root PDFs (MD5 prefixes `ce3efb2bbedd` and `ba847c370873`);
- the drafts in `testing-docs/` (`bbd844b4e5b8` and `a8c3af3207a8`);
- `.env` and the test data.

## 3. Application defects fixed

| Defect | Severity | Root cause | Fix | Retest |
|---|---|---|---|---|
| **DEF-01** (ST-33) | Major | `onStartTraining` disabled Start training and nothing re-evaluated it when returning to the configure screen; `#opt-balance` and `#opt-test-ratio` had no listener | `updateTrainButton()` runs whenever the configure screen is shown (`onStateChange`); change listeners added | 🟢 PASS (run 4) |
| **DEF-02** (ST-34) | Minor | The warning compared **all rows** with 20,000; the backend subsamples **training rows** (after the test split and the 20% validation hold-out) | The warning uses the same training-row rule (sklearn rounding) and updates when the test split changes | 🟢 PASS: 25,000 rows: no warning, no subsampling; 40,000 rows: warning shown, 20,000 of 25,600 rows used |
| **DEF-03** (ST-49) | Minor | At ≤ 768 px, only `.winner-name` reserved space for the 50 px stamp | `.winner-kicker { display: block; padding-right: 52px; }` in the same media rule | 🟢 PASS: overlap 0 px² (was 691 px²) in Chrome, Edge and Chromium at 375 px |
| **DEF-04** (ST-50) | Major | `POST /api/train` built the trainers in the request, importing TensorFlow and PennyLane before returning 202 | `DeferredTrainer` in `core/trainers.py` builds the DNN/VQC trainer on first use in the job thread | 🟢 PASS: 20 ms (run 4), **18 ms on a fresh server** (run 5); was 1,043 ms and 5,641 ms |

### 3.5 ST-36: investigated, not changed

- **Server side:** the server sets the `Testing <model> (i of n)` stage correctly; 0.1 s polling saw it.
- **UI side:** the UI renders the latest poll every 2 s, as COM-2 requires, so a testing step shorter than 2 s can be skipped. The label was not drawn in runs 1 and 2 but was drawn in run 4.

This is a requirement conflict between REQ-JB-5 and COM-2, not a code defect. The decision is left to the developers (Q9).

## 4. Tests executed and actual results

| Execution | Build | Result | Evidence |
|---|---|---|---|
| Developer pytest, not slow (before) | `4988037` | 231 passed, 0 failed; coverage 96% | `testing-docs/final/evidence/developer-pytest-not-slow.txt` |
| Developer pytest, slow (before) | `4988037` | 1 passed | `testing-docs/final/evidence/developer-pytest-slow.txt` |
| Run 1, all 50 | `4988037` | 48 passed, 1 failed (ST-33), 1 skipped | `tests/runs/20261110-122004-4988037/` |
| Run 2, all 50 | `4988037` | 47 PASS, 🔴 2 FAIL (ST-33, ST-50), 🔴 1 UNVERIFIED (ST-36) | `tests/runs/20261110-132648-4988037/` |
| Run 3, ST-50 on a fresh server | `4988037` | 🔴 FAIL, 5,641 ms | `tests/runs/20261110-133444-4988037/` |
| **Developer pytest, not slow (after)** | fixed | 🟢 **234 passed** (231 + 3 new), 0 failed; coverage 96% (1,165 statements, 45 missed) | `testing-docs/final/evidence/developer-pytest-not-slow-after-fixes.txt` |
| **Developer pytest, slow (after)** | fixed | 🟢 **1 passed** | `testing-docs/final/evidence/developer-pytest-slow-after-fixes.txt` |
| **Run 4, all 50 (final)** | fixed | 🟢 **50 PASS**, 0 FAIL, 0 UNVERIFIED (10 min 54 s) | `tests/runs/20261110-225957-4988037-modified/` |
| **Run 5, ST-50 on a fresh server** | fixed | 🟢 PASS, train 202 = 18 ms | `tests/runs/20261110-231135-4988037-modified/` |

Run 4 covered every area the request listed: API, UI, upload, configuration, training, cancellation, results, CSV export and performance (feature groups A–K).

## 5. Performance measurements

**Environment:**
- AMD Ryzen 5 7535HS, 15.3 GB RAM, NVIDIA RTX 3050 Laptop GPU (driver 566.07), Windows 11 Home build 26300.
- Python 3.11.9; app on `127.0.0.1:5055` with `COMPUTE_DEVICE=cpu`.
- Chrome 157.0.8103.39 (it updated itself before run 4), Edge 157.0.4341.42, Chromium 158.0.8127.18.
- Playwright 1.65.0, pytest 9.2.0.

| Endpoint (`tools/measure_times.py`) | SRS target | Run 2 (before) | Run 4 | Run 5 (fresh) | Met |
|---|---|---|---|---|---|
| `POST /api/upload` median (200 / 10k / 50k rows) | < 2 s | 47 / 62 / 111 ms | 66 / 83 / 171 ms | 36 / 43 / 135 ms | 🟢 |
| `POST /api/configure` median | < 500 ms | 15 / 32 / 51 ms | 8 / 19 / 77 ms | 7 / 24 / 59 ms | 🟢 |
| `POST /api/train` → 202 | < 200 ms | 🔴 1,043 ms | 20 ms | 18 ms | 🟢 |
| `GET /api/status` median (max) | < 200 ms | 15 (88) ms | 6 (164) ms | 9 (198) ms | 🟢 |
| `GET /api/results` | < 500 ms | 16 ms | 24 ms | 4 ms | 🟢 |

| Tuning or hardware check (STP 8.11) | Run 4 measurement | Met |
|---|---|---|
| P1 Split reproducibility | Seed 42 → `ba7ba92f3866e20f` twice; seed 7 differs | 🟢 |
| P2 Advanced-panel limits | All boundary values behave as specified | 🟢 |
| P3 DNN epoch limit 5 | 5 epochs run, not stopped early, 12,289 parameters | 🟢 |
| P4 QML settings | Qubits 2, epochs 3, 128 training rows, `default.qubit` | 🟢 |
| P5 QML row cap | 2,000 training rows; 8,000 test rows | 🟢 |
| P6 Threshold modes | Auto: LR 0.328, XGBoost 0.574; fixed 0.500 | 🟢 |
| P7 Class balancing | "balanced", then "none", without a workaround | 🟢 (was 🔴 before DEF-01) |
| P8 Winner ranking | DNN, PR-AUC 0.935 | 🟢 |
| P9 SVM subsampling | 20,000 of 25,600 training rows on 40,000 rows | 🟢 |
| P10 Run times | Default QML 47.90 s (≤ 180 s); 8 models 17 s (< 120 s) | 🟢 |
| P11 API times | See the table above | 🟢 |
| P12 CPU/GPU | `auto`: XGBoost "GPU (CUDA)", LR and DNN on CPU; `cpu`: all CPU | 🟢 |
| P13 Quality baseline | Identical to run 2 (8 models on Banking) | Recorded |
| P14 Better than random `[PROPOSED]` | Lowest ROC-AUC 0.675 (VQC) > 0.5 | 🟢 |

**Note:** the whole laptop was about twice as slow in run 4 as in run 2 (for example, the default QML run took 47.90 s against 25.28 s). Every target is still met, but the timings between those runs are not a like-for-like comparison.

## 6. Remaining failures and pending items

| Item | Status |
|---|---|
| Failing system test cases | **None** in run 4 |
| Review and commit of the 5 changed tracked files | 🟠 **DEVELOPER REVIEW** |
| DEF-01 to DEF-04: Fixed → Closed after the commit and a rerun on its SHA | 🟠 **DEVELOPER REVIEW** / 🟡 PENDING |
| Q1–Q9 requirement questions (SEC-6 timeouts, REQ-MS-4 wording, new requirement IDs, PERF-8, `deviations.md` item 8, README screenshot, owners, Q9 stage label) | 🟠 **DEVELOPER REVIEW** |
| <span style="color:#c00000">flake8 / pylint / black on the changed files</span> | 🔴 **NOT RUN**: not installed in the project `.venv`; to be run by CI on the commit |
| <span style="color:#c00000">Runs 4 and 5 used uncommitted changes (folders end in `-modified`)</span> | 🟡 PENDING: rerun on the commit SHA |
| <span style="color:#c00000">Evaluator selection of about 30 cases and the acceptance run</span> | 🟡 PENDING |
| USN, institution, faculty guide, effort hours, signatures and approvals | 🟡 PENDING (`[FILL LATER]`) |
| Hosted (Render) instance test; CI pipeline result | 🟡 NOT EXECUTED: no hosted instance available to the tester; CI is developer-owned |

## 7. Cleanup

| Action | Detail |
|---|---|
| Removed | `__pycache__/` in the root, `core/`, `tests/` and `testing-docs/final/tests/`; two stale Word lock files (`~$P_…v2.1.docx`, `~$S_…v1.1.docx`) |
| `.gitignore` additions | Keep each run's `app-server.log` as evidence (`*.log` was hiding it); ignore `testing-docs/final/tests/runs/dev-check/` (debug runs, kept on disk), `.coverage-*` and `~$*` |
| Not removed | `.env`, `uploads/`, every run folder, test data, root PDFs, drafts, archive |
| Not found | `.pytest_cache`, `*.tmp`, `*.bak` |

## 8. Final validation

| Check | Result |
|---|---|
| Word documents open (STP v2.2, SRS v1.2, TSR, Developer Review) | 🟢 All opened in Microsoft Word; contents tables and fields refreshed; 23 / 20 / 10 / 9 pages; key pages rendered to PDF and checked visually |
| Excel workbooks (5) | 🟢 All opened in Excel and fully recalculated: **0 formula errors**; counts agree (50 PASS, 4 defects Fixed) |
| Evidence paths cited in the Markdown records and the execution record | 🟢 127 checked, 0 missing |
| Git diff | 🟢 Only the 5 files in section 2.1 changed; each change reviewed |
| Secrets and personal files | 🟢 `.env` is git-ignored and unchanged; no keys, tokens or e-mail addresses in the files that would be committed. Two evidence text files contain the local Windows path (`C:\Users\ravit\…`) as environment data |
| Originals | 🟢 Root PDFs and drafts have unchanged MD5 checksums |

## 9. Developer actions required

1. **Review the fixes:** `git diff main` on branch `qa/defect-fixes-final-testing`, with the explanation in `Developer_Review.docx` section 3A. Accept and commit them, or replace them with your own fix.
2. **Run the CI lint and tests** (`.github/workflows/ci.yml`) on that commit.
3. **Answer Q1–Q9** in `Developer_Review.docx`, Q9 in particular.
4. **Name** the frontend owner, the development lead and the project manager.
5. **Sign** the Developer Review response rows.

## 10. Git changes and branch

```
Branch: qa/defect-fixes-final-testing (from main @ 4988037), not committed, not pushed

 .gitignore             | 11 +++++
 core/trainers.py       | 76 +++++++++++++++++-----
 static/css/styles.css  |  2 +
 static/js/config.js    | 45 ++++++++++---
 tests/test_trainers.py | 46 ++++++++++++-
 5 files changed, 163 insertions(+), 17 deletions(-)

Untracked (tester deliverables): testing-docs/, TESTING_IMPLEMENTATION.md,
FINAL_COMPLETION_REPORT.md, SRS.docx, STP.docx, SRS_Full_Stack_Prediction_Platform (1).pdf,
STP_Full_Stack_Prediction_Platform.pdf
```

**No commit, push or merge has been made. These need your permission.**

## 11. Final acceptance readiness

| Criterion (STP 9) | Status |
|---|---|
| All 50 cases executed | 🟢 50 / 50 |
| Pass rate ≥ `[PROPOSED: 90%]` | 🟢 100% (run 4) |
| No UNVERIFIED case | 🟢 None |
| No open Critical or Major defect | 🟠 DEF-01 and DEF-04 Fixed and retested, but not Closed until committed |
| Run on a committed build | 🟡 Pending |
| Evaluator acceptance run and approvals | 🟡 Pending |

**Verdict:** the build is **ready for the evaluator's acceptance run after the developers commit the fixes and the regression is repeated on that commit**. It is <span style="color:#c00000">**not yet accepted**</span>, and the work is **not 100% complete**, because of the items in section 6.
