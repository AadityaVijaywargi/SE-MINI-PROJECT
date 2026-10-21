# Full-Stack Prediction Platform: ML vs DL vs QML

Upload a tabular dataset, pick the column to predict, and train several classifiers on
exactly the same train/test split. The results dashboard ranks every model on the same
test rows.

**Sprint 1 (Oct 19 to 23):** upload, profiling, target analysis, shared preprocessing,
six classical ML models with live progress, and the results table. The deep neural
network, the quantum classifier, charts, CSV export and Docker come in Sprint 2.

> Results are for comparative study only. Do not use them for clinical, credit or
> underwriting decisions.

## Quick start

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
python app.py
```

Open http://127.0.0.1:5000 and upload one of the files in `data/samples/`.

In production (Linux) the app runs under Gunicorn with one worker and four threads, because
the job store is held in memory:

```bash
gunicorn -c gunicorn.conf.py
```

## Tests and lint

```bash
pytest
black --check .
flake8 .
bandit -c pyproject.toml -r .
pylint app.py config.py gunicorn.conf.py core tools tests
```

GitHub Actions runs pytest, black, flake8 and bandit on every push and pull request
(`.github/workflows/ci.yml`). Static-analysis reports from before and after the lint fixes
are in `reports/before/` and `reports/after/`.

## How it works

```
Browser (static/)  --HTTP/JSON-->  Flask API (app.py)
                                      |-- DatasetStore      core/storage.py       uploads under UUID names
                                      |-- profiling         core/profiling.py     column roles, preview
                                      |-- target analysis   core/targets.py       suggestions, task type
                                      '-- JobRunner thread  core/jobs.py
                                             |-- Preprocessor  core/preprocessing.py  runs once per job
                                             |-- Trainers      core/trainers.py       6 ML models
                                             '-- Evaluator     core/evaluator.py      same metrics for all
```

| Endpoint | Method | Returns |
|---|---|---|
| `/api/health` | GET | `{status: "ok"}` |
| `/api/upload` | POST (multipart `file`) | 201 `{dataset_id, name, profile, suggested_targets, preview}` |
| `/api/configure` | POST `{dataset_id, target}` | `{task, n_classes, classes, distribution, imbalance_ratio, warning, trainable, reason}` |
| `/api/train` | POST `{dataset_id, target, models[], seed?, test_ratio?}` | 202 `{job_id}` |
| `/api/status/<job_id>` | GET | `{status, progress, stage, models[], log[], elapsed, error}`; 404 if unknown |
| `/api/results/<job_id>` | GET | `{results[], summary, metadata}` |

Errors are always JSON: `{error: "message"}`.

## Project layout

```
app.py                  Flask app factory and routes
config.py               settings (environment variables override them)
gunicorn.conf.py        production server settings
core/                   domain logic, no Flask imports
static/                 index.html, css/, js/ (ES modules, no build step)
data/samples/           four domain datasets + one three-class dataset
tools/make_samples.py   regenerates data/samples/
tests/                  pytest suite, one file per core module plus test_api.py
docs/deviations.md      differences from the submitted SRS / SAD / Test Plan
```

## Branches

- `main`: the released version; only merged from `dev` via pull request.
- `dev`: the integration branch; feature branches merge here via pull request.
- `feature/SM-<story>-<short-name>`: one branch per Jira story, opened by the assignee.

## Sample datasets

Each sample has 200 rows and one deliberate defect:

| File | Target | Defect |
|---|---|---|
| `banking.csv` | Exited | missing values in Balance and CreditScore; CustomerId is an ID |
| `healthcare.csv` | Diagnosis | string labels (Positive / Negative) |
| `telecom.csv` | Churn | City has 60 categories (ordinal-encoded) |
| `insurance.csv` | Fraud | Country is constant (dropped) |
| `three_class.csv` | Segment | three classes, for multi-class runs |
