# Test reports (Deliverables Part 3)

All four reports come from one recorded pytest run on `main` (application code at
`8fb23bf`), with every test included: unit, integration/API, the slow QML test and the
system runs on the preloaded datasets. Result: **238 of 238 tests passed.**

| Report | File | Headline |
|---|---|---|
| Unit Test Report | [unit-test-report.md](unit-test-report.md) | 232 unit and integration/API tests, all passed |
| System Validation Report | [system-validation-report.md](system-validation-report.md) | banking, insurance, healthcare, telecom and three_class end to end, 6 of 6 passed |
| Code Coverage Report | [code-coverage-report.md](code-coverage-report.md) | 96.2% of statements; `core/` 96.7% against the 80% target |
| Branch Coverage Report | [branch-coverage-report.md](branch-coverage-report.md) | 90.0% of branch outcomes; 95.1% line + branch combined |

The test cases these reports execute are in [../test-cases.md](../test-cases.md) (v2.0).

Supporting files:

- `coverage-html/index.html`: browsable coverage, every line and branch highlighted.
- `raw/junit.xml`: every test and its result (JUnit, xunit1, with the system runs' metrics).
- `raw/coverage.json`, `raw/coverage.xml` (Cobertura): line and branch data.
- `raw/pytest-output.txt`: the console output of the run.

## Regenerate

```bash
pytest -o junit_family=xunit1 --junitxml=docs/test-reports/raw/junit.xml --cov=core --cov=app --cov-branch --cov-report=json:docs/test-reports/raw/coverage.json --cov-report=xml:docs/test-reports/raw/coverage.xml --cov-report=html:docs/test-reports/coverage-html
```

```bash
python tools/test_reports.py
```

Delete the `.gitignore` coverage.py writes into `coverage-html/` before committing it.
