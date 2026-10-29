"""System validation (Test Plan 8.1, Section 9): every preloaded dataset run end to end through
the API, the same way the browser drives it: load the sample, configure the suggested target,
train all eight models, poll, read the results and export the CSV.

Each test records its per-model metrics with ``record_property``, so they land in the JUnit XML
that ``tools/test_reports.py`` turns into the System Validation Report."""

import csv
import io
import json

import pytest

from tests.conftest import DOMAIN_SAMPLES, SAMPLES_DIR
from tests.test_api import ALL_ML, ALL_MODELS, FAST_OPTIONS, train, upload, wait_for

pytestmark = pytest.mark.system

METRICS = ("accuracy", "precision", "recall", "f1")


def check_results(body, n_models):
    """Checks shared by every dataset: one result per model, all on the same held-out rows."""
    results = body["results"]
    metadata = body["metadata"]
    assert len(results) == n_models
    assert all(result["status"] == "done" for result in results), results
    # BR-2: one split for every model, proven by the fingerprint and the test row count.
    assert metadata["split_fingerprint"]
    assert {result["test_rows"] for result in results} == {metadata["test_rows"]}
    for result in results:
        for metric in METRICS:
            assert 0 <= result[metric] <= 1, (result["model"], metric)
        assert result["train_time"] >= 0 and result["test_time"] >= 0
        assert sum(map(sum, result["confusion_matrix"])) == metadata["test_rows"]
    assert body["summary"]["best_overall"] in {result["model"] for result in results}


def check_csv(client, job_id, n_models):
    """REQ-RS-9: the export has a header and one row per model."""
    response = client.get(f"/api/results/{job_id}/csv")
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
    assert len(rows) == n_models + 1


def record(record_property, dataset, body):
    """Store the run in the JUnit XML for the System Validation Report."""
    record_property(
        "system_run",
        json.dumps(
            {
                "dataset": dataset,
                "metadata": body["metadata"],
                "summary": body["summary"],
                "results": [
                    {
                        key: result.get(key)
                        for key in (
                            "model",
                            "paradigm",
                            *METRICS,
                            "roc_auc",
                            "pr_auc",
                            "train_time",
                            "test_time",
                        )
                    }
                    for result in body["results"]
                ],
            }
        ),
    )


@pytest.mark.parametrize("name", sorted(DOMAIN_SAMPLES))
def test_preloaded_dataset_end_to_end(client, record_property, name):
    # SV-01 to SV-04: banking, healthcare, insurance and telecom, all eight models.
    loaded = client.post(f"/api/samples/{name}")
    assert loaded.status_code == 201
    dataset = loaded.get_json()
    target = DOMAIN_SAMPLES[name]
    assert dataset["suggested_targets"][0]["column"] == target

    task = client.post(
        "/api/configure", json={"dataset_id": dataset["dataset_id"], "target": target}
    ).get_json()
    assert task["task"] == "binary" and task["trainable"]

    response = train(client, dataset["dataset_id"], ALL_MODELS, target=target, **FAST_OPTIONS)
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    status = wait_for(client, job_id, timeout=300)
    assert status["status"] == "complete", status["error"]
    assert status["progress"] == 100

    body = client.get(f"/api/results/{job_id}").get_json()
    check_results(body, len(ALL_MODELS))
    assert [result["paradigm"] for result in body["results"]] == ["ML"] * 6 + ["DL", "QML"]
    assert all(result["roc_auc"] is not None for result in body["results"])
    assert body["metadata"]["dataset"] == f"{name}.csv"
    assert body["metadata"]["target"] == target
    assert {"best_ml", "best_dl", "best_qml"} <= set(body["summary"])
    check_csv(client, job_id, len(ALL_MODELS))
    record(record_property, f"{name}.csv", body)


def test_three_class_dataset_end_to_end(client, record_property):
    # SV-05: the multi-class path. QML is binary only, so it is left out, and ROC is hidden.
    dataset = upload(client, "three_class.csv").get_json()
    assert dataset["suggested_targets"][0]["column"] == "Segment"

    task = client.post(
        "/api/configure", json={"dataset_id": dataset["dataset_id"], "target": "Segment"}
    ).get_json()
    assert task["task"] == "multiclass" and task["n_classes"] == 3

    models = [*ALL_ML, "dnn"]
    job_id = train(client, dataset["dataset_id"], models, target="Segment", dnn_epochs=5)
    job_id = job_id.get_json()["job_id"]
    status = wait_for(client, job_id, timeout=300)
    assert status["status"] == "complete", status["error"]

    body = client.get(f"/api/results/{job_id}").get_json()
    check_results(body, len(models))
    assert body["metadata"]["task"] == "multiclass"
    assert all(result["roc_curve"] is None for result in body["results"])
    assert all(len(result["confusion_matrix"]) == 3 for result in body["results"])
    check_csv(client, job_id, len(models))
    record(record_property, "three_class.csv", body)


def test_preloaded_samples_match_the_bundled_files(client):
    # SV-06: the sample buttons serve exactly the committed files, nothing else.
    listed = client.get("/api/samples").get_json()
    assert {sample["name"] for sample in listed} == set(DOMAIN_SAMPLES)
    for sample in listed:
        assert (SAMPLES_DIR / f"{sample['name']}.csv").exists()
        assert sample["target"] == DOMAIN_SAMPLES[sample["name"]]
