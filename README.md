# Full-Stack Prediction Platform: ML vs DL vs QML

Upload a tabular dataset, pick the column to predict, and train eight classifiers from
three paradigms on exactly the same train/test split: six classical machine learning
models, one deep neural network and one variational quantum classifier. The dashboard
ranks every model on the same held-out test rows and shows what each paradigm costs in
accuracy and time.

> **Results are for comparative study only. Do not use them for clinical, credit or
> underwriting decisions.**

## Quick start

### Docker (recommended)

```bash
docker compose up --build
```

Open http://localhost:5000, then click a sample dataset or drop in your own CSV or Excel file.

### Virtualenv

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
python app.py
```

Open http://127.0.0.1:5000. Settings come from environment variables or a `.env` file;
`.env.example` lists them all with their defaults.

In production (Linux) the app runs under Gunicorn with one worker and four threads, because
the job store is held in memory: `gunicorn -c gunicorn.conf.py`. The Docker image does this
already.

### NVIDIA GPU (optional)

With an NVIDIA GPU and its driver, XGBoost trains on the GPU automatically, on Windows
too. The DNN also uses the GPU on Linux or WSL2 once the GPU build of TensorFlow is in
place of the CPU one:

```bash
pip uninstall -y tensorflow-cpu
pip install -r requirements-gpu.txt
```

Each model's details show `Device: GPU (CUDA)` or `CPU`. Set `COMPUTE_DEVICE=cpu` to keep
everything on the CPU. The scikit-learn models have no GPU support, and the quantum
simulator stays on the CPU because 2 to 8 qubits are too small to gain from a GPU.

## What it does

1. **Upload:** drag and drop a `.csv` or `.xlsx` file (up to 100 MB), or click one of four
   sample datasets. The server stores it under a random id, profiles every column and shows
   a preview.
2. **Configure:** the most likely target column is preselected. The task is labelled
   binary, multi-class or unsupported, and an imbalance warning appears above 3:1. All
   eight models are ticked. The Advanced panel sets the test split, seed, DNN epochs,
   QML epochs and qubit count.
3. **Train:** the job runs in the background. The progress bar, stage label, model
   checklist and log show real server progress, epoch by epoch for the DNN and the
   quantum model. The job can be cancelled at any time.
4. **Results:** a winner banner, the best model of each paradigm, a sortable metrics table,
   grouped metric bars, ROC curves, training time on a log scale, accuracy against time,
   one confusion matrix per model, the run metadata and a CSV export.

### The models

| Paradigm | Models | Notes |
|---|---|---|
| ML | Logistic Regression, Decision Tree, Random Forest (200 trees), SVM (RBF), Gaussian Naive Bayes, XGBoost | SVM trains on a stratified sample of 20,000 rows when the training set is larger. |
| DL | Dense network: 128 (BatchNorm, Dropout 0.3), 64 (Dropout 0.2), 32, output | Keras on TensorFlow-CPU, Adam 1e-3, early stopping (patience 10, best weights) on a 20% validation split of the training rows. |
| QML | Variational quantum classifier | PCA to 4 features, scaling to [0, pi], angle encoding, 2 strongly entangling layers, <Z> on qubit 0. PennyLane simulator, binary targets only, at most 2,000 training rows. |

Every model is tested on the same held-out rows. Binary targets use binary averaging and
get ROC curves; multi-class targets use weighted averaging. The split fingerprint in the
run metadata is a hash of the test labels, so two runs can be checked for the same split.

## Architecture

```
Browser (static/, Chart.js)  --HTTP/JSON-->  Flask API (app.py, Gunicorn 1 worker x 4 threads)
                                               |-- DatasetStore     core/storage.py      UUID uploads, samples, cleanup
                                               |-- profiling        core/profiling.py    column roles, preview
                                               |-- target analysis  core/targets.py      suggestions, task type
                                               |-- CSV export       core/export.py       formula escaping
                                               '-- JobRunner thread core/jobs.py         progress, cancel, TTL
                                                      |-- Preprocessor core/preprocessing.py  once per job
                                                      |-- Trainers     core/trainers.py       6 ML models
                                                      |                core/dnn.py            Keras DNN
                                                      |                core/qml.py            PennyLane VQC
                                                      '-- Evaluator    core/evaluator.py      same metrics for all
```

All trainers implement the `Trainer` interface in `core/base.py`: `fit(split, report)`
returns a model with `predict` and `predict_proba`, so the evaluator has one code path for
every paradigm. TensorFlow and PennyLane are imported only when a DL or QML model trains.

## API reference

Errors are always JSON: `{error: "message"}`.

| Endpoint | Method | Returns |
|---|---|---|
| `/api/health` | GET | `{status: "ok"}` |
| `/api/samples` | GET | `[{name, label, target}]` for the four sample datasets |
| `/api/samples/<name>` | POST | 201, same body as `/api/upload` |
| `/api/upload` | POST (multipart `file`) | 201 `{dataset_id, name, profile, suggested_targets, preview}` |
| `/api/configure` | POST `{dataset_id, target}` | `{task, n_classes, classes, distribution, imbalance_ratio, warning, trainable, reason}` |
| `/api/train` | POST `{dataset_id, target, models[], seed?, test_ratio?, dnn_epochs?, qml_epochs?, qubits?}` | 202 `{job_id}`; 409 if this browser already has a job running |
| `/api/status/<job_id>` | GET | `{status, progress, stage, models[], log[], elapsed, error, cancel_requested}`; 404 if unknown or expired |
| `/api/cancel/<job_id>` | POST | 202 with the status body; 409 if the job has already finished |
| `/api/results/<job_id>` | GET | `{status, results[], summary, metadata}` |
| `/api/results/<job_id>/csv` | GET | the metrics table as a CSV download |

Model keys: `logistic_regression`, `decision_tree`, `random_forest`, `svm`, `naive_bayes`,
`xgboost`, `dnn`, `vqc`. Job status is `queued`, `running`, `complete`, `failed` or
`cancelled`. Each result holds `model`, `paradigm`, `status`, `accuracy`, `precision`,
`recall`, `f1`, `roc_auc`, `roc_curve {fpr, tpr}` (at most 100 points, binary only),
`train_time`, `test_time`, `test_rows`, `confusion_matrix` and `details` (for example epochs
run, qubits, training rows). `metadata` holds the dataset, rows, features, target, task,
seed, test ratio, split fingerprint, column lists, `qml_subsampled_to` and
`pca_variance_retained`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `production` restricts CORS and hides unexpected job errors |
| `CORS_ORIGINS` | empty | comma-separated origins allowed to call the API cross-site |
| `PORT` | `5000` | listen port |
| `UPLOAD_DIR` | `uploads/` | where uploads are stored |
| `MAX_UPLOAD_MB` | `100` | upload size limit |
| `UPLOAD_TTL_SECONDS` / `UPLOAD_DIR_MAX_MB` | `3600` / `500` | uploads are deleted after an hour, oldest first above the cap |
| `JOB_TTL_SECONDS` | `3600` | finished jobs are forgotten after this |
| `SVM_MAX_ROWS` | `20000` | SVM subsampling threshold |
| `DNN_EPOCHS` | `100` | default DNN epoch limit (early stopping usually ends sooner) |
| `QML_MAX_ROWS`, `QML_EPOCHS`, `QML_QUBITS` | `2000`, `20`, `4` | quantum model defaults |
| `QML_DEVICE` | `default.qubit` | PennyLane simulator; `lightning.qubit` also works but is about 5x slower for 4 qubits; `lightning.gpu` needs Linux and `requirements-gpu.txt` |
| `COMPUTE_DEVICE` | `auto` | `auto` trains XGBoost and the DNN on an NVIDIA GPU when CUDA is usable; `cpu` never uses the GPU |

## Tests, lint and CI

```bash
pytest -m "not slow"
```

```bash
pytest -m slow
```

```bash
black --check . && flake8 . && bandit -c pyproject.toml -r .
```

```bash
pylint app.py config.py gunicorn.conf.py core tools tests
```

There are 199 tests: 198 fast ones plus one slow test, the full default QML run. Coverage of
`core/` and `app.py` is 97%. GitHub Actions (`.github/workflows/ci.yml`) runs black,
flake8, bandit and pytest with coverage on every push and pull request. It then builds the
Docker image, boots it, waits for `/api/health` and checks that the container runs as uid
1000. Static-analysis reports from Sprint 1 are in `reports/`.

## Deployment

`render.yaml` deploys the Docker image to Render on every merge to `main`, with health
check path `/api/health`. The image is a multi-stage `python:3.11-slim` build that runs as
a non-root user (uid 1000) with a Docker health check. Open the hosted app about 5 minutes
before a demo: the free tier sleeps when idle. `docs/demo-script.md` has the rehearsal
script, the manual checklist and the hosted-instance check.

## Sample datasets

Generated by `tools/make_samples.py` (deterministic, seed 42). Each has 200 rows and one
deliberate defect.

| File | Domain | Target | Defect |
|---|---|---|---|
| `banking.csv` | Banking churn | Exited | missing values in Balance and CreditScore; CustomerId is an ID |
| `healthcare.csv` | Diabetes screening | Diagnosis | string labels (Positive / Negative) |
| `telecom.csv` | Telecom churn | Churn | City has 60 categories (ordinal-encoded) |
| `insurance.csv` | Claim fraud | Fraud | Country is constant (dropped) |
| `three_class.csv` | Customer segments | Segment | three classes, for multi-class runs (not a sample button) |

The data is synthetic: no real customers or patients.

## Known limitations

- **QML is compressed and subsampled.** The quantum model sees only 4 PCA components
  (about 55 to 70% of the variance on the samples) and at most 2,000 training rows, so it
  is not a like-for-like comparison of raw capacity. The metadata panel always reports the
  subsample size and variance retained. No quantum advantage or disadvantage is claimed.
- **QML is binary only.** It is disabled for multi-class targets.
- **Simulator only.** No real quantum hardware.
- **Tabular classification only.** Regression targets (more than 20 classes), images, text
  and time series are rejected.
- **Single worker.** Jobs live in memory, so Gunicorn must run one worker, and jobs are lost
  on restart. Celery and Redis are the documented upgrade path (ASM-5).
- **Free-tier hosting.** Render's free tier sleeps after about 15 minutes idle (30 to 60 s
  to wake). Its 512 MB memory may not fit TensorFlow, PennyLane and XGBoost together;
  `docker compose up` is the fallback.
- **One job per browser** at a time. Datasets are deleted an hour after upload.

## Project layout

```
app.py                  Flask app factory and routes
config.py               settings (environment variables and .env override them)
gunicorn.conf.py        production server settings
core/                   domain logic, no Flask imports
static/                 index.html, css/, js/ (ES modules, no build step)
data/samples/           four domain datasets + one three-class dataset
tools/                  make_samples.py, measure_times.py
tests/                  pytest suite, one file per core module plus API and security tests
docs/                   deviations.md, performance.md, demo-script.md
Dockerfile, docker-compose.yml, render.yaml, .env.example
```

## Branches

- `main`: the released version; only merged from `dev` via pull request.
- `dev`: the integration branch; feature branches merge here via pull request.
- `feature/SM-<story>-<short-name>`: one branch per Jira story, opened by the assignee.

## Team

| Member | Responsibilities |
|---|---|
| Aaditya Vijaywargi | Data path through ML training: upload profiling, preprocessing, the six ML models and SVM subsampling, evaluation (ROC, summary, metadata), API tests, documentation |
| Anish Kumar | API layer and DevOps: upload validation, DNN, cancel and job expiry, CORS, logging, cleanup, Docker, CI, Render |
| Arya Devang | Quantum model (PCA, VQC, row cap, metadata), QML note, security tests, demo script, performance targets |
| Ravi S | Frontend: sample buttons, Advanced panel, warnings, charts, confusion matrices, metadata panel, CSV export, accessibility |

Differences from the submitted SRS, design and test documents are listed in
`docs/deviations.md`.
