# Manual Verification Guide (updated after regression run 4)

Final regression run: `testing-docs\04_Test_Execution_Results\20261110-225957-4988037-modified\` (commit 4988037 plus the
fixes for DEF-01 to DEF-04, 10 Nov 2026). **All 50 cases passed; no manual check is open.**

History:
- After run 1, eight cases needed a person (ST-29, ST-30, ST-34, ST-35, ST-36, ST-42, ST-45,
  ST-48). They were then scripted with measurable criteria.
- In run 2, ST-36 stayed UNVERIFIED: the 'Testing <model> (i of n)' label was never drawn on
  screen.
- In run 4, the label was drawn and photographed in the TD-12 confirmation run, so ST-36 passed.

| Case | How it is verified | Run 2 | Run 4 |
|---|---|---|---|
| ST-29 | Epoch label photographed on screen in a TD-12 confirmation run | PASS | PASS |
| ST-30 | Same, for the quantum classifier | PASS | PASS |
| ST-34 | Fully scripted, including the 25,000-row boundary (DEF-02, fixed) | PASS | PASS |
| ST-35 | Fixture restarts the app with `COMPUTE_DEVICE=auto`, then `cpu`; `nvidia-smi` recorded | PASS | PASS |
| ST-36 | Stage labels sampled on screen; TD-12 confirmation run | **UNVERIFIED** | PASS |
| ST-42 | Chart colours, labels, dashed baselines and axes read from Chart.js | PASS | PASS |
| ST-45 | CSV opened in Microsoft Excel by script: 0 formula cells, 0 error cells | PASS | PASS |
| ST-48 | Every focus outline measured: ≥ 2 px and 6.09:1 to 6.64:1 contrast (≥ 3:1 required) | PASS | PASS |

## Optional manual confirmation of ST-36 (timing-dependent)

> **DEVELOPER REVIEW (Q9):** the UI refreshes every 2 s (COM-2), and a model's testing step is
> often shorter than that. Whether the label is drawn therefore depends on timing: it was drawn in
> run 4 but not in runs 1 and 2. The developers decide whether that is acceptable.

To see it yourself:

1. In PowerShell, at the repository root:

   ```powershell
   $env:COMPUTE_DEVICE = "cpu"; .venv\Scripts\python.exe app.py
   ```

2. Open http://127.0.0.1:5000 in Chrome.
3. Upload `testing-docs\03_Test_Data\TD-12_large_40000.csv`, click **Continue to configure →** and choose target **Exited**.
4. Tick **Random Forest**, **SVM** and **Deep Neural Network**, then click **Start training**.
5. Watch the stage label for `Testing <model> (i of 3)`.
6. If you see it, screenshot it as `testing-docs\04_Test_Execution_Results\Manual_Checks\<date>\ST-36_testing-stage.png`.

## People actions still open

- **Developers:**
  - Review and commit the fixes for DEF-01 to DEF-04 (`testing-docs\05_Defect_Management\Developer_Review_v1.1.docx`, Section 3A).
  - Answer Q1 to Q9.
- **Tester:** after the commit, rerun the suite on the commit SHA:

  ```powershell
  testing-docs\02_Test_Automation\run_tests.ps1
  testing-docs\02_Test_Automation\run_tests.ps1 -Select st50   # fresh server
  ```

  Then set each defect to Closed in `testing-docs\05_Defect_Management\Defect_Log.xlsx`.
- **Evaluator:** select about 30 cases and run them with the tester (`testing-docs\04_Test_Execution_Results\Execution_Records\Test_Execution_Record_Final.xlsx`, column *Selected for Acceptance*).
