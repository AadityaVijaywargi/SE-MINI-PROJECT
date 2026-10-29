# Test Cases — ML/DL/QML Training Dashboard (v2.0)

Updated test case document for Deliverables Part 3. It replaces v1.0 (12 Oct 2026, 14 cases)
and brings every case in line with what Sprints 1 and 2 actually built
(`docs/deviations.md`, deviations 1 to 29).

Each test case (TC-nn) traces to a requirement ID in the SRS and names the pytest test(s)
that execute it, so a result can always be reproduced with one command. Results are from
the run recorded in `docs/test-reports/` (branch `main` at `8fb23bf` plus the new system
tests); see the four reports there for the evidence.

**Legend.** *Level*: U = unit, I = integration/API (Flask test client), S = system (end to
end through the API on the preloaded datasets), M = manual (browser, demo checklist in
`docs/demo-script.md`), CI = GitHub Actions. *Change*: what differs from v1.0.

## Preloaded test data

The five bundled datasets in `data/samples/` (Test Plan 8.6) are the test data for every
level. The four domain datasets are the **Try a sample dataset** buttons on the Upload
screen (`/api/samples`); each has one deliberate defect.

| Dataset | Domain | Rows | Target (expected suggestion) | Task | Deliberate defect |
|---|---|---|---|---|---|
| `banking.csv` | Financial (bank churn) | 200 | `Exited` | binary | missing `Balance` and `CreditScore`; `CustomerId` identifier |
| `insurance.csv` | Financial (insurance fraud) | 200 | `Fraud` | binary | high-cardinality `Country`; rare positive class |
| `healthcare.csv` | Healthcare | 200 | `Diagnosis` | binary | text labels (`Positive`/`Negative`); `PatientId` identifier |
| `telecom.csv` | Telecom churn | 200 | `Churn` | binary | text labels (`Yes`/`No`); high-cardinality `City` |
| `three_class.csv` | Retail segments | 200 | `Segment` | 3-class | multi-class path (QML and ROC not applicable) |

## 1. Upload & Validation

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-01 | Valid CSV / XLSX upload | Upload `banking.csv` (drag-and-drop in the UI; `POST /api/upload` in tests). Repeat with the same data as `.xlsx`. | 201 with `dataset_id`, `name`, profile, column roles, target suggestions and a 10-row preview; the `.xlsx` profiles like the CSV. | REQ-UP-1, REQ-UP-5..8 | I, M | `test_api::test_upload_returns_profile_suggestions_and_preview`, `test_api::test_xlsx_upload_is_parsed_like_csv`, `test_profiling::*` | PASS | Updated: XLSX added (SM-97); response fields per deviation 6. Drag-and-drop stays a manual check. |
| TC-02 | Reject oversized file | Upload a file larger than `MAX_UPLOAD_MB` (100 MB, set lower in the test). | 413 with a JSON error the toast shows; nothing is stored. | REQ-UP-2, SEC-1 | I | `test_api::test_upload_rejects_oversized_file`, `test_security::test_sec1_size_limit_is_enforced` | PASS | Unchanged. |
| TC-15 | Reject bad, malformed and hostile files | Upload a `.txt`, an `.exe` and a look-alike (`data.csv.exe`); upload a malformed CSV; upload with filename `../../etc/passwd`. | 400 with an actionable message naming the allowed types; malformed file deleted; stored name is always a UUID, never the client name. | REQ-UP-3, REQ-UP-4, SEC-1 | U, I | `test_api::test_upload_rejects_bad_extension`, `test_api::test_upload_rejects_executables_and_lookalikes`, `test_api::test_upload_rejects_malformed_file_and_deletes_it`, `test_storage::*`, `test_security::test_sec1_client_filename_never_reaches_the_path` | PASS | New (split out of the v1.0 catalogue). |
| TC-16 | Preloaded sample datasets | `GET /api/samples`, then `POST /api/samples/<name>` for each of banking, healthcare, telecom, insurance. Try `three_class`, `..` and `config.py`. | Four samples listed with their targets; each loads like an upload (201, new UUID, profile); anything else is 404 and nothing is written. | REQ-UP-9 | U, I | `test_api::test_samples_are_listed`, `test_api::test_sample_loads_like_an_upload`, `test_api::test_unknown_sample_is_404`, `test_api::test_sample_path_traversal_never_matches_a_route`, `test_storage::test_each_sample_loads_under_a_new_uuid`, `test_system::test_preloaded_samples_match_the_bundled_files` | PASS | New: sample buttons were added in Sprint 2 (deviation 20). |

## 2. Target & Task Analysis

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-03 | Target scoring / ranking | Profile each preloaded dataset. | Candidates (2 to 20 distinct values) ranked with binary +50, keyword +40 (whole word), last column +20, non-numeric +10. Top suggestion: `Exited`, `Diagnosis`, `Churn`, `Fraud`, `Segment`. | REQ-TG-1, REQ-TG-2 | U | `test_targets::test_scoring_rules`, `test_targets::test_keyword_must_be_a_whole_word`, `test_targets::test_domain_samples_suggest_their_target_first`, `test_targets::test_three_class_suggests_segment` | PASS | Updated: checked on all five preloaded datasets, not one. |
| TC-04 | Block unsupported target | Choose a continuous column (`Balance`), a single-class column, and a column where one class has a single row; call `/api/configure`, then `/api/train`. | `task: unsupported`, `trainable: false` with a plain-language `reason`; `/api/train` refuses with 400. | REQ-TG-4, REQ-TG-6 | U, I | `test_targets::test_regression_target_blocked`, `test_targets::test_single_class_blocked`, `test_targets::test_class_with_one_row_blocked`, `test_api::test_train_refuses_unsupported_target` | PASS | Updated: single-class and one-row-class cases added; `reason` field per deviation 6. |
| TC-17 | Task type and imbalance warning | Configure `banking` / `Exited` and `three_class` / `Segment`; configure a target with a class ratio above 3:1. | `binary` and `multiclass` with class list and distribution; imbalance ratio and warning shown above 3:1. | REQ-TG-3..5 | U, I | `test_targets::test_binary_task`, `test_targets::test_multiclass_task`, `test_targets::test_imbalance_warning`, `test_api::test_configure` | PASS | New. |

## 3. Preprocessing & Split

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-05 | Imputation, encoding and scaling | Preprocess every preloaded dataset (missing numerics, text categories, identifiers, an all-missing column, an unseen test category). | Numerics median-imputed then standardised; categoricals mode-imputed then one-hot (or ordinal for high cardinality); identifier, constant and all-missing columns dropped; output is a dense finite matrix with no NaN. | REQ-PP-1..6 | U | `test_preprocessing::test_dense_finite_matrix_for_every_domain`, `test_preprocessing::test_identifier_and_constant_dropped_target_kept`, `test_preprocessing::test_all_missing_column_dropped`, `test_preprocessing::test_unseen_category_in_test_does_not_break`, `test_preprocessing::test_string_labels_encoded` | PASS | Updated: run on all four domain datasets. |
| TC-06 | No data leakage | Inspect the fitted transformer and the validation rows during a training run. | Transformer fitted on training rows only, then applied to the test rows; DNN validation rows and tuned-threshold validation rows come from the training rows only; preprocessing runs once per job. | REQ-PP-7, REQ-PP-8 | U | `test_preprocessing::test_scaler_fitted_on_training_rows_only`, `test_preprocessing::test_validation_rows_are_split_from_training_rows`, `test_preprocessing::test_preprocessing_runs_once_per_job`, `test_dnn::test_validation_split_comes_from_training_rows_only` | PASS | Updated: covers the validation rows added in Sprint 2 (deviations 16 and the rare-class fix). |
| TC-18 | Stratified, reproducible split | Split with seed 42 twice, then with seed 7; run two jobs with the same seed via the API. | Same seed gives the same split and the same split fingerprint; another seed differs; class shares kept in train and test. | REQ-PP-9, REQ-PP-10, R4 | U, I | `test_preprocessing::test_split_is_stratified_and_sized`, `test_preprocessing::test_same_seed_same_split_other_seed_differs`, `test_api::test_same_seed_gives_same_split_fingerprint` | PASS | New. |

## 4. Model Training

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-07 | Train all ML models | Start a job with LR, DT, RF, SVM, NB and XGBoost on each preloaded dataset. | All six train with the configured seed and identical predictions on a rerun; class weights balance rare classes; XGBoost uses CUDA only when a GPU is usable. | REQ-TR-1, REQ-TR-2 | U, S | `test_trainers::test_each_model_trains_and_predicts`, `test_trainers::test_same_seed_same_predictions`, `test_trainers::test_balanced_models_find_a_rare_class`, `test_gpu::*`, `test_system::test_preloaded_dataset_end_to_end` | PASS | Updated: class weights (rare-class fix) and optional GPU (deviation 29). |
| TC-19 | Train the DNN | Train the Keras DNN on a binary and a 3-class target. | Architecture per SRS (sigmoid / softmax output); early stopping restores the best epoch; progress reported every epoch; cancel stops at the next epoch. | REQ-TR-3, REQ-TR-4 | U | `test_dnn::*` | PASS | New: DNN built in Sprint 2. |
| TC-20 | Train the quantum classifier (VQC) | Train the 4-qubit VQC on 2,000+ rows; rerun with the same seed; try a 3-class target. | PCA to the qubit count, angles scaled to [0, π]; training rows capped at 2,000 and the cap and PCA variance recorded; tested on the full shared test set; same seed gives same predictions; multi-class refused; default run beats chance. | REQ-TR-5..7 | U (one slow) | `test_qml::*` (`test_default_vqc_learns_better_than_chance` is `slow`), `test_jobs::test_qml_metadata_recorded_when_vqc_runs` | PASS | New: QML built in Sprint 2; cap applies to training rows only (deviation 15). |
| TC-08 | Per-model failure isolation | Force SVM to fail with an invalid hyperparameter (`C = -1`) in a two-model job. | Job completes; SVM reported `failed` with its reason; Naive Bayes finishes and is chosen best. | REQ-TR-8 | U, I | `test_api::test_forced_model_failure_is_reported_and_others_finish`, `test_jobs::test_one_failing_model_does_not_stop_the_job` | PASS | Unchanged. |
| TC-21 | SVM subsampling above 20,000 rows | Train SVM on more than `SVM_MAX_ROWS` rows. | SVM trains on a stratified sample of 20,000 rows; other models use every row. | REQ-MS-3, PERF, R8 | U | `test_trainers::test_svm_is_subsampled_above_the_row_cap`, `test_trainers::test_svm_not_subsampled_below_the_cap`, `test_trainers::test_other_models_are_never_subsampled` | PASS | New. |

## 5. Job & API

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-09 | Asynchronous job creation | `POST /api/train` with a valid config. | 202 with a 32-character `job_id` in under 2 s, before training finishes; bad models or settings give 400. | REQ-JB-1, PERF-1 | I | `test_api::test_train_returns_202_immediately`, `test_api::test_train_rejects_unknown_model`, `test_api::test_train_requires_a_model`, `test_api::test_train_validates_settings` | PASS | Updated: input validation added. |
| TC-10 | Unhandled exception in a job | Force an unexpected exception inside a job. | Job `failed`, message stored, full traceback logged server-side. | REQ-JB-8 | U | `test_jobs::test_unhandled_job_error_marks_failed_and_logs_traceback` | PASS | Unchanged. |
| TC-22 | Progress, cancel and partial results | Poll `/api/status` during a run; `POST /api/cancel/<id>` while a model trains. | Progress only moves forward and moves inside DNN/QML epochs; cancel sets `cancelled`, stops at the next epoch or model, keeps finished results, and `/api/results` returns them. | REQ-JB-3..6 | U, I | `test_jobs::test_progress_moves_inside_a_long_model`, `test_jobs::test_cancel_stops_before_the_next_model`, `test_jobs::test_cancelled_job_keeps_finished_results`, `test_api::test_cancel_stops_the_job_and_keeps_finished_results` | PASS | New: cancel built in Sprint 2 (deviations 17 and 19). |
| TC-23 | One job per session; job eviction; unknown job | Start a second job from the same browser while one runs; wait past `JOB_TTL_SECONDS`; ask for an unknown job id. | Second job 409 (`pp_session` cookie); finished jobs evicted after the TTL; unknown ids 404 on status, results, CSV and cancel. | REQ-JB-7, SEC-6 | U, I | `test_api::test_second_job_in_the_same_session_is_refused`, `test_jobs::test_one_active_job_per_session`, `test_jobs::test_store_evicts_finished_jobs_after_ttl`, `test_api::test_unknown_job_is_404` | PASS | New (deviation 18). |

## 6. Evaluation & Results

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-11 | Core metrics for every model | Complete a job with all eight models on each preloaded dataset. | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, train time, test time, test rows, threshold and confusion matrix for every model; weighted averages and no ROC for multi-class; ROC downsampled to about 100 points. | REQ-EV-1..3 | U, S | `test_evaluator::*`, `test_trainers::test_each_model_result_has_every_metric_key`, `test_system::test_preloaded_dataset_end_to_end`, `test_system::test_three_class_dataset_end_to_end` | PASS | Updated: PR-AUC, test time/rows and threshold added (deviation 10 and the rare-class fix). |
| TC-24 | Best-model summary and rare-class threshold | Summarise a binary run with a rare positive class; run once with a fixed 0.5 threshold. | Binary runs ranked by PR-AUC then F1 (multi-class by F1 then accuracy); best overall and best per paradigm reported; decision threshold tuned on validation rows, and a fixed threshold is reported as `fixed`. | REQ-EV-4, REQ-EV-5 | U, I | `test_evaluator::test_summary_best_overall_and_per_paradigm`, `test_evaluator::test_binary_summary_ranks_by_pr_auc_not_accuracy`, `test_evaluator::test_threshold_is_tuned_on_validation_rows`, `test_api::test_fixed_threshold_and_unbalanced_run_are_reported` | PASS | New: replaces "best = highest accuracy" (deviation 8) after the rare-class fix. |
| TC-12 | CSV export injection safety | Export results where a model name or value starts with `=`, `+`, `-` or `@` via `GET /api/results/<id>/csv`. | Leading character escaped; numbers kept as numbers; finished models listed first; downloaded as `<dataset>_model_comparison.csv`. | REQ-RS-9, SEC-2 | U, I | `test_export::*`, `test_api::test_csv_export_downloads_the_metrics_table` | PASS | Updated: export now built on the server (deviation 21). |
| TC-25 | End-to-end run on every preloaded dataset | For banking, insurance, healthcare and telecom: load the sample, configure the suggested target, train all eight models, poll, read results, export CSV. For `three_class`: upload, configure `Segment`, train the six ML models and the DNN. | Every model finishes; all metrics in [0, 1]; one split fingerprint and one test-row count for all models; each confusion matrix sums to the test rows; CSV has one row per model; ROC present for binary and hidden for 3-class. | REQ-UP-9, REQ-JB-1..3, REQ-EV-1, REQ-RS-5, REQ-RS-9, BR-2 | S | `test_system::test_preloaded_dataset_end_to_end[banking/healthcare/insurance/telecom]`, `test_system::test_three_class_dataset_end_to_end` | PASS | New: system validation for Part 3. |
| TC-26 | Results dashboard | Open the Results screen after a full run on `banking.csv` and on `three_class.csv`. | Winner banner, three paradigm cards, sortable table with best values highlighted, four charts (ROC replaced by a note for multi-class), one confusion matrix per model, run-metadata panel. | REQ-RS-1..8 | M | Manual checklist in `docs/demo-script.md` | Manual | New (manual, as in Test Plan feature M). |

## 7. Security, Deployment & Non-functional

| TC ID | Test Case | Steps | Expected Result | Req | Level | Automated by | Result | Change |
|---|---|---|---|---|---|---|---|---|
| TC-13 | Generic errors in production | With `APP_ENV=production`, force a server exception and an unexpected job error; then a data problem. | Client sees a generic message; full detail and traceback logged server-side; debug off; data problems (for example "no usable feature columns") still shown in full; unknown routes answer JSON. | SEC-5 | U, I | `test_security::test_sec5_unexpected_errors_are_generic_and_logged`, `test_security::test_sec5_unknown_routes_answer_json`, `test_jobs::test_production_hides_unexpected_job_errors`, `test_jobs::test_production_still_shows_data_problems` | PASS | Updated: data problems stay visible (deviation 23). |
| TC-27 | CORS, secrets, upload cleanup, non-root | Check CORS headers in production and development; check `.env` is ignored and `.env.example` exists; leave an upload past `UPLOAD_TTL_SECONDS` and fill the folder past its cap; check the Dockerfile user. | Production echoes only `CORS_ORIGINS` (none when empty), development only localhost; no `.env` committed; old uploads deleted and folder kept under `UPLOAD_DIR_MAX_MB`; container runs as uid 1000. | SEC-3, SEC-4, SEC-7, SEC-8 | U, CI | `test_security::test_sec3_*`, `test_security::test_sec4_env_file_is_ignored_and_example_exists`, `test_storage::test_cleanup_*`, `test_security::test_sec8_dockerfile_runs_as_non_root`; CI job `docker-smoke-test` checks `id -u` = 1000 | PASS | New (Test Plan 5.1; deviations 24 and 25). |
| TC-14 | Auto-deploy on merge | Merge a PR into `main`. | Render redeploys via `render.yaml` (`autoDeploy: true`, branch `main`); `/api/health` passes. CI builds the image, boots it and calls `/api/health` within 40 s. | OTH-2, OTH-3, PERF-6 | CI | `test_api::test_health`; CI job `docker-smoke-test` | Health: PASS. Deploy: not run here (needs the hosted instance) | Updated: health check is now automated; the Render deploy itself stays a hosted check. |

## 8. Traceability summary

| Area | Test cases | Requirement group |
|---|---|---|
| Upload & Validation | TC-01, TC-02, TC-15, TC-16 | REQ-UP, SEC-1 |
| Target & Task Analysis | TC-03, TC-04, TC-17 | REQ-TG |
| Preprocessing & Split | TC-05, TC-06, TC-18 | REQ-PP |
| Model Training | TC-07, TC-08, TC-19, TC-20, TC-21 | REQ-TR, REQ-MS |
| Job & API | TC-09, TC-10, TC-22, TC-23 | REQ-JB, PERF, SEC-6 |
| Evaluation & Results | TC-11, TC-12, TC-24, TC-25, TC-26 | REQ-EV, REQ-RS, SEC-2, BR-2 |
| Security, Deployment & Non-functional | TC-13, TC-14, TC-27 | SEC, OTH, PERF |

**Total test cases: 27** (14 from v1.0, all kept with updated steps and expectations, plus
13 new). TC IDs from v1.0 are unchanged so earlier references still hold; new cases are
numbered from TC-15.

## 9. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 12 Oct 2026 | 14 representative cases (TC-01 to TC-14). |
| 2.0 | 29 Oct 2026 | Part 3 update. Every case now names its pytest tests and records a result. TC-01, 03, 04, 05, 06, 07, 09, 11, 12, 13 and 14 updated for Sprint 1 and 2 changes. Added TC-15 to TC-27: hostile uploads, preloaded samples, task/imbalance, reproducible split, DNN, QML, SVM cap, cancel and partial results, one job per session, rare-class ranking and thresholds, end-to-end runs on every preloaded dataset, the results dashboard and the remaining SEC checks. |

## 10. Sign-off

| Field | Value |
|---|---|
| Tester | Aaditya Vijaywargi |
| Date executed | 29 Oct 2026 |
| Environment | Local (Windows 11, Python 3.10, pinned requirements); CI on GitHub Actions for TC-14 and TC-27 container checks |
| Pass / Fail summary | 25 of 27 cases pass in the automated run. TC-26 is manual (demo checklist). TC-14: health check passes; the Render deploy needs the hosted instance. |
