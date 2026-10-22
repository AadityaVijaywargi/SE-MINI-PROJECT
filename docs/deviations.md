# Deviations from the submitted documents

Changes against the SRS, the Software Architecture & Design Specification (SAD) and the
Test Plan. They need to be explained in the final demo.

## Sprint 1

| # | Document | What changed | Why |
|---|---|---|---|
| 1 | Jira plan | SM-43 (per-model failure isolation, Sprint 2) was built in Sprint 1, inside `JobRunner.train_one`. | Without it, one model crashing would fail the whole demo run. |
| 2 | Jira plan | SM-55 (best-overall and best-per-paradigm summary, Sprint 2) was built in Sprint 1 (`evaluator.summarise`). | The Sprint 1 winner banner (SM-57) needs to know the best model. |
| 3 | Jira plan | SM-56 (run metadata) is partly built: dataset, rows, target, task, seed, split fingerprint and column lists. QML subsample size and PCA variance are added with QML in Sprint 2. | The results payload in SRS 3.3 always carries a metadata object. |
| 4 | SRS 2.4 | Python 3.10 works locally as well as 3.11. CI runs on 3.11. | Team laptops have 3.10. |
| 5 | SRS 2.4 | Gunicorn is installed only on Linux. Locally, `python app.py` runs Flask's threaded server. | Gunicorn does not run on Windows. |
| 6 | SRS 3.3 | `/api/status` also returns `models[]`, `log[]`, `elapsed` and `error`. `/api/upload` also returns `name`. `/api/configure` also returns `classes`, `imbalance_ratio`, `trainable` and `reason`. | The Train screen checklist and log (REQ-JB-5) and the blocked-target message (REQ-TG-6) need them. No fields were removed. |
| 7 | SRS UI-7 | An extra frontend module, `dom.js`, holds shared DOM helpers. | Every screen builds elements the same way, using `textContent` only, so file contents cannot inject HTML. |
| 8 | SRS REQ-EV-4 | "Best" means highest accuracy, with F1 as the tie-break. | The SRS does not say which metric decides. |
| 9 | Jira plan | SM-15 and SM-16 moved from Aaditya to Anish, and SM-36 from Anish to Aaditya (Oct 21). | Rebalanced work: Aaditya owns the data path through ML training; Anish owns the API layer. |
| 10 | SRS REQ-EV-1, REQ-JB-5, REQ-RS-3 | Each result also carries `test_time` and `test_rows`. The Train screen shows a "Testing X (i of n)" stage and a `testing` checklist status after each model trains, and the master table has a Test time column. | Models were always scored on the held-out test set, but testing ran inside the "Training" stage and was never shown, so it looked as if models were only trained. |
