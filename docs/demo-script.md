# Demo rehearsal script and manual system checklist (SM-90, SM-98)

## Before the demo

1. Open the hosted app 5 minutes early so the free tier wakes up (ASM-2). If it does not
   respond, run the local fallback: `docker compose up`, then open http://localhost:5000.
2. Close other heavy programs. A full run trains 8 models; the quantum model alone takes
   up to a minute on 2,000 rows.
3. Have `data/samples/banking.csv` and `data/samples/three_class.csv` ready to drag in.

## Two-minute script

| Time | Show | Say |
|---|---|---|
| 0:00 | Upload screen | "One page, four steps. Every model is trained and tested on exactly the same split." |
| 0:10 | Click **Banking** under "Try a sample dataset" | "The profile flags CustomerId as an identifier and the missing balances; both are handled automatically." |
| 0:25 | **Continue to configure** | "Exited is suggested as the target: two classes, a target-like name, last column." |
| 0:35 | Point at the QML note, open **Advanced settings** | "Quantum training compresses to 4 features with PCA and caps the rows at 2,000. The defaults work, so a first-time user never needs this panel." |
| 0:45 | **Start training** | "The server answers immediately; training runs in the background. The bar is the real server progress, epoch by epoch for the DNN and the quantum circuit." |
| 1:05 | Results: winner banner and paradigm cards | "Best model overall, and the best of each paradigm." |
| 1:15 | Metrics table, click **ROC-AUC** to sort | "Every score is on the held-out test rows. The best value in each column is highlighted." |
| 1:25 | Charts | "Grouped metrics, ROC curves against the random baseline, training time on a log scale, and accuracy against time: the cost of each paradigm." |
| 1:40 | Confusion matrices, Run metadata | "The split fingerprint proves every model saw the same split. QML trained on 160 rows and kept this share of the variance." |
| 1:50 | **Export CSV** | "Formula characters are escaped so a spreadsheet cannot run injected formulas." |
| 1:55 | Footer | "Results are for comparative study only, not for clinical or credit decisions." |

## Manual system checklist

Tick each box during the rehearsal; note the browser and date.

**Upload (REQ-UP)**
- [ ] Dragging `banking.csv` onto the drop zone shows summary cards, a 10-row preview and the column table.
- [ ] Clicking the drop zone opens the file picker; Enter and Space do the same from the keyboard.
- [ ] Each of the four sample buttons loads its dataset with no file picker.
- [ ] A `.txt` file shows a toast naming the allowed types; nothing is uploaded.
- [ ] A file over 100 MB is refused in the browser before upload.
- [ ] An `.xlsx` file uploads and profiles like a CSV.

**Configure (REQ-TG, REQ-MS)**
- [ ] The suggested target is preselected and starred; any column can be chosen.
- [ ] Choosing `Balance` shows "Unsupported" and blocks training with a reason.
- [ ] `three_class.csv` with target `Segment`: QML is unticked, disabled and asterisked with "Binary targets only".
- [ ] Switching back to a binary target re-enables and re-ticks QML.
- [ ] A dataset with more than 20,000 rows shows the SVM subsampling warning.
- [ ] Entering 12 qubits shows an error in the Advanced panel and disables **Start training**.
- [ ] The class-imbalance warning appears for a target with a ratio above 3:1.

**Train (REQ-JB)**
- [ ] The stage label reads like "Training Random Forest (3 of 8)", and the bar only moves forward.
- [ ] Each model goes pending, then running, then testing, then done in the checklist.
- [ ] **Cancel training** during the DNN stops the job within a few seconds and opens the partial results.
- [ ] Starting a second job in another tab of the same browser while one runs shows "already running".

**Results (REQ-RS)**
- [ ] The winner banner is the largest element and shows model, paradigm, accuracy and F1.
- [ ] Three paradigm cards show each paradigm's best model, accuracy and training time.
- [ ] Every table column sorts on click; the best value per column is highlighted.
- [ ] All four charts render; ROC is replaced by a note for a multi-class target.
- [ ] One confusion matrix per model, with class labels.
- [ ] The metadata panel shows dataset, rows, features, target, task, seed, split fingerprint, QML subsample size and PCA variance.
- [ ] **Export CSV** downloads a file that opens in a spreadsheet without running formulas.

**Accessibility and layout (UI-1 to UI-5)**
- [ ] The whole flow works with the keyboard only; focus outlines are visible.
- [ ] At 375 px wide every multi-column layout becomes one column and wide tables scroll sideways.
- [ ] Every chart and badge pairs colour with a text label.

## Hosted instance check (SM-98)

Run after each deployment to Render:

1. `curl https://<your-app>.onrender.com/api/health` returns `{"status": "ok"}`.
2. `python tools/measure_times.py https://<your-app>.onrender.com`, and add the numbers to
   `docs/performance.md`.
3. Run all 8 models on `banking.csv` in the browser. In the Render dashboard, check that
   memory stays under the plan limit (TensorFlow, PennyLane and XGBoost together need
   about 800 MB to 1 GB, ASM-3) and note the total job time.
4. If the instance restarts for lack of memory, record it here and use the Docker fallback
   for the demo.

| Date | Instance | Memory peak | 8-model job time | Result |
|---|---|---|---|---|
| | | | | |
