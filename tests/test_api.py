"""REST API tests (SM-88, SM-89, SM-97): upload, samples, configure, train, status, cancel,
results and CSV export, through the Flask test client."""

import csv
import io
import time

import pandas as pd
import pytest
from sklearn.svm import SVC

from app import create_app
from core.trainers import ML_MODELS
from tests.conftest import SAMPLES_DIR

ALL_ML = ["logistic_regression", "decision_tree", "random_forest", "svm", "naive_bayes", "xgboost"]
ALL_MODELS = [*ALL_ML, "dnn", "vqc", "qsvm"]
# Small DL and QML settings so the full nine-model flow stays quick.
FAST_OPTIONS = {"dnn_epochs": 5, "qml_epochs": 1, "qubits": 2}


def upload(client, name="banking.csv", content=None):
    """POST a sample file (or ``content``) to /api/upload."""
    data = content if content is not None else (SAMPLES_DIR / name).read_bytes()
    return client.post("/api/upload", data={"file": (io.BytesIO(data), name)})


def dataset_id_of(client, name="banking.csv"):
    """Upload a sample and return its dataset id."""
    return upload(client, name).get_json()["dataset_id"]


def train(client, dataset_id, models, target="Exited", **extra):
    """POST /api/train."""
    return client.post(
        "/api/train", json={"dataset_id": dataset_id, "target": target, "models": models, **extra}
    )


def wait_for(client, job_id, timeout=120):
    """Poll /api/status until the job finishes; return the last status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/status/{job_id}").get_json()
        if body["status"] in ("complete", "failed", "cancelled"):
            return body
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}


def test_upload_returns_profile_suggestions_and_preview(client):
    response = upload(client)
    body = response.get_json()
    assert response.status_code == 201
    assert len(body["dataset_id"]) == 32
    assert body["profile"]["rows"] == 200
    assert body["suggested_targets"][0]["column"] == "Exited"
    assert len(body["preview"]["rows"]) == 10


def test_xlsx_upload_is_parsed_like_csv(client):
    frame = pd.read_csv(SAMPLES_DIR / "healthcare.csv")
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    response = upload(client, "healthcare.xlsx", buffer.getvalue())
    body = response.get_json()
    assert response.status_code == 201, body
    assert body["name"] == "healthcare.xlsx"
    assert body["profile"]["rows"] == 200
    assert body["profile"]["columns"] == frame.shape[1]
    assert body["suggested_targets"][0]["column"] == "Diagnosis"


def test_upload_rejects_bad_extension(client):
    response = upload(client, "notes.txt", b"a,b\n1,2\n")
    assert response.status_code == 400
    assert ".csv" in response.get_json()["error"]


@pytest.mark.parametrize("name", ["setup.exe", "data.csv.exe", "payload.xlsm"])
def test_upload_rejects_executables_and_lookalikes(client, name):
    response = upload(client, name, b"MZ\x90\x00 not a table")
    assert response.status_code == 400


def test_upload_without_file(client):
    assert client.post("/api/upload").status_code == 400


def test_upload_rejects_oversized_file(tmp_path):
    app = create_app(
        {"UPLOAD_DIR": tmp_path / "uploads", "MAX_CONTENT_LENGTH": 1024, "TESTING": True}
    )
    response = upload(app.test_client(), "big.csv", b"a,b\n" + b"1,2\n" * 1000)
    assert response.status_code == 413
    assert "larger than" in response.get_json()["error"]


def test_upload_rejects_malformed_file_and_deletes_it(client, tmp_path):
    response = upload(client, "broken.csv", b"\x00\xff\x00 not a table")
    assert response.status_code == 400
    assert response.get_json()["error"]
    assert not list((tmp_path / "uploads").glob("*"))


def test_samples_are_listed(client):
    body = client.get("/api/samples").get_json()
    assert [sample["name"] for sample in body] == ["banking", "healthcare", "telecom", "insurance"]
    assert {sample["target"] for sample in body} == {"Exited", "Diagnosis", "Churn", "Fraud"}


def test_sample_loads_like_an_upload(client, tmp_path):
    response = client.post("/api/samples/telecom")
    body = response.get_json()
    assert response.status_code == 201
    assert body["name"] == "telecom.csv"
    assert body["suggested_targets"][0]["column"] == "Churn"
    assert (tmp_path / "uploads" / f"{body['dataset_id']}.csv").exists()


@pytest.mark.parametrize("name", ["unknown", "three_class", "..", "config.py"])
def test_unknown_sample_is_404(client, name):
    assert client.post(f"/api/samples/{name}").status_code == 404


def test_sample_path_traversal_never_matches_a_route(client, tmp_path):
    # Encoded slashes are decoded before routing, so no sample route matches at all.
    assert client.post("/api/samples/..%2F..%2Fconfig").status_code in (404, 405)
    assert not (tmp_path / "uploads").exists()


def test_configure(client):
    body = client.post(
        "/api/configure", json={"dataset_id": dataset_id_of(client), "target": "Exited"}
    )
    assert body.get_json()["task"] == "binary"


def test_configure_unknown_dataset_is_404(client):
    response = client.post("/api/configure", json={"dataset_id": "0" * 32, "target": "x"})
    assert response.status_code == 404


def test_unknown_job_is_404(client):
    for path in ("/api/status/x", "/api/results/x", "/api/results/x/csv"):
        assert client.get(path).status_code == 404
    assert client.post("/api/cancel/x").status_code == 404


def test_train_returns_202_immediately(client):
    dataset_id = dataset_id_of(client)
    start = time.perf_counter()
    response = train(client, dataset_id, ["random_forest"])
    assert response.status_code == 202
    assert time.perf_counter() - start < 2
    assert len(response.get_json()["job_id"]) == 32
    wait_for(client, response.get_json()["job_id"])


def test_train_refuses_unsupported_target(client):
    response = train(client, dataset_id_of(client), ALL_ML, target="Balance")
    assert response.status_code == 400
    assert "regression" in response.get_json()["error"]


def test_train_rejects_unknown_model(client):
    response = train(client, dataset_id_of(client), ["nope"])
    assert response.status_code == 400
    assert "nope" in response.get_json()["error"]


def test_train_requires_a_model(client):
    assert train(client, dataset_id_of(client), []).status_code == 400


def test_train_refuses_qml_on_multiclass_target(client):
    response = train(client, dataset_id_of(client, "three_class.csv"), ["vqc"], target="Segment")
    assert response.status_code == 400
    assert "binary targets only" in response.get_json()["error"]


@pytest.mark.parametrize(
    "extra",
    [
        {"seed": -1},
        {"seed": "42"},
        {"seed": True},
        {"test_ratio": 0.5},
        {"dnn_epochs": 0},
        {"dnn_epochs": 501},
        {"qml_epochs": 2.5},
        {"qubits": 1},
        {"qubits": 9},
        {"threshold": 0},
        {"threshold": 1.5},
        {"threshold": "high"},
        {"threshold": True},
        {"balance_classes": "yes"},
    ],
)
def test_train_validates_settings(client, extra):
    response = train(client, dataset_id_of(client), ["naive_bayes"], **extra)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_second_job_in_the_same_session_is_refused(client):
    dataset_id = dataset_id_of(client)
    first = train(client, dataset_id, ["dnn"], dnn_epochs=300)
    assert first.status_code == 202
    second = train(client, dataset_id, ["naive_bayes"])
    assert second.status_code == 409
    assert "already running" in second.get_json()["error"]

    other_browser = client.application.test_client()  # its own cookie jar, so a new session
    assert train(other_browser, dataset_id, ["naive_bayes"]).status_code == 202

    client.post(f"/api/cancel/{first.get_json()['job_id']}")
    wait_for(client, first.get_json()["job_id"])
    assert train(client, dataset_id, ["naive_bayes"]).status_code == 202


def test_cancel_stops_the_job_and_keeps_finished_results(client):
    dataset_id = dataset_id_of(client)
    job_id = train(client, dataset_id, ["naive_bayes", "dnn"], dnn_epochs=300).get_json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        status = client.get(f"/api/status/{job_id}").get_json()
        if status["models"][1]["status"] == "running":
            break
        time.sleep(0.1)
    response = client.post(f"/api/cancel/{job_id}")
    assert response.status_code == 202
    assert response.get_json()["cancel_requested"] is True

    status = wait_for(client, job_id)
    assert status["status"] == "cancelled"
    assert [model["status"] for model in status["models"]] == ["done", "cancelled"]
    body = client.get(f"/api/results/{job_id}").get_json()
    assert body["status"] == "cancelled"
    assert [result["key"] for result in body["results"]] == ["naive_bayes"]
    assert client.post(f"/api/cancel/{job_id}").status_code == 409


def test_forced_model_failure_is_reported_and_others_finish(client, monkeypatch):
    # TC-08: an invalid hyperparameter makes SVM raise during fit.
    monkeypatch.setitem(ML_MODELS, "svm", ("SVM", lambda seed: SVC(C=-1.0)))
    job_id = train(client, dataset_id_of(client), ["svm", "naive_bayes"]).get_json()["job_id"]
    status = wait_for(client, job_id)
    assert status["status"] == "complete"
    svm, bayes = status["models"]
    assert svm["status"] == "failed" and "C" in svm["error"]
    assert bayes["status"] == "done"
    body = client.get(f"/api/results/{job_id}").get_json()
    assert body["results"][0]["status"] == "failed"
    assert body["summary"]["best_overall"] == "Naive Bayes"


def test_full_flow(client):
    dataset_id = dataset_id_of(client)
    response = train(client, dataset_id, ALL_ML)
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]

    status = wait_for(client, job_id)
    assert status["status"] == "complete", status["error"]
    assert status["progress"] == 100
    assert all(model["status"] == "done" for model in status["models"])
    assert sum("testing on" in line for line in status["log"]) == 6
    assert sum("tested in" in line for line in status["log"]) == 6

    body = client.get(f"/api/results/{job_id}").get_json()
    assert len(body["results"]) == 6
    assert body["summary"]["best_overall"] in {result["model"] for result in body["results"]}
    assert body["metadata"]["dataset"] == "banking.csv"
    assert body["metadata"]["split_fingerprint"]
    test_rows = body["metadata"]["test_rows"]
    assert all(result["test_rows"] == test_rows for result in body["results"])
    assert all(result["test_time"] >= 0 for result in body["results"])
    assert all(len(result["roc_curve"]["fpr"]) <= 100 for result in body["results"])


def test_full_flow_all_nine_models(client):
    job_id = train(client, dataset_id_of(client), ALL_MODELS, **FAST_OPTIONS).get_json()["job_id"]
    status = wait_for(client, job_id, timeout=300)
    assert status["status"] == "complete", status["error"]
    assert [model["status"] for model in status["models"]] == ["done"] * 9

    body = client.get(f"/api/results/{job_id}").get_json()
    assert [result["paradigm"] for result in body["results"]] == ["ML"] * 6 + ["DL", "QML", "QML"]
    assert body["summary"]["best_dl"] == "Deep Neural Network"
    assert body["summary"]["best_qml"] in {"Variational Quantum Classifier", "Quantum Kernel SVM"}
    metadata = body["metadata"]
    assert metadata["qml_subsampled_to"] == metadata["train_rows"]
    assert 0 < metadata["pca_variance_retained"] <= 1
    assert metadata["qubits"] == 2
    # BR-2: every paradigm was scored on the same held-out rows.
    assert {result["test_rows"] for result in body["results"]} == {metadata["test_rows"]}


def test_full_flow_multiclass(client):
    dataset_id = dataset_id_of(client, "three_class.csv")
    response = train(client, dataset_id, [*ALL_ML, "dnn"], target="Segment", dnn_epochs=5)
    status = wait_for(client, response.get_json()["job_id"])
    assert status["status"] == "complete", status["error"]

    body = client.get(f"/api/results/{response.get_json()['job_id']}").get_json()
    assert body["metadata"]["task"] == "multiclass"
    assert all(result["status"] == "done" for result in body["results"])
    assert all(result["roc_auc"] is None for result in body["results"])
    assert all(result["roc_curve"] is None for result in body["results"])
    assert all(len(result["confusion_matrix"]) == 3 for result in body["results"])


def test_same_seed_gives_same_split_fingerprint(client):
    dataset_id = dataset_id_of(client)
    fingerprints = []
    for _ in range(2):
        job_id = train(client, dataset_id, ["naive_bayes"], seed=7).get_json()["job_id"]
        wait_for(client, job_id)
        fingerprints.append(
            client.get(f"/api/results/{job_id}").get_json()["metadata"]["split_fingerprint"]
        )
    assert fingerprints[0] == fingerprints[1]


def test_csv_export_downloads_the_metrics_table(client):
    job_id = train(client, dataset_id_of(client), ["naive_bayes", "svm"]).get_json()["job_id"]
    wait_for(client, job_id)
    response = client.get(f"/api/results/{job_id}/csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert 'filename="banking_model_comparison.csv"' in response.headers["Content-Disposition"]
    rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
    assert rows[0][:4] == ["Model", "Paradigm", "Status", "Accuracy"]
    assert {row[0] for row in rows[1:]} == {"SVM", "Naive Bayes"}


def test_fixed_threshold_and_unbalanced_run_are_reported(client):
    response = train(
        client, dataset_id_of(client), ["naive_bayes"], threshold=0.3, balance_classes=False
    )
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    assert wait_for(client, job_id)["status"] == "complete"
    body = client.get(f"/api/results/{job_id}").get_json()
    result = body["results"][0]
    assert result["threshold"] == 0.3 and result["threshold_source"] == "fixed"
    assert body["metadata"]["threshold_mode"] == 0.3
    assert body["metadata"]["balance_classes"] is False
    assert result["details"]["Class weights"] == "none"
