"""REST API smoke tests: upload, configure, train, status and results."""

import io
import time

from app import create_app
from tests.conftest import SAMPLES_DIR

ALL_ML = ["logistic_regression", "decision_tree", "random_forest", "svm", "naive_bayes", "xgboost"]


def upload(client, name="banking.csv", content=None):
    """POST a sample file (or ``content``) to /api/upload."""
    data = content if content is not None else (SAMPLES_DIR / name).read_bytes()
    return client.post("/api/upload", data={"file": (io.BytesIO(data), name)})


def wait_for(client, job_id, timeout=60):
    """Poll /api/status until the job completes or fails; return the last status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/status/{job_id}").get_json()
        if body["status"] in ("complete", "failed"):
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


def test_upload_rejects_bad_extension(client):
    response = upload(client, "notes.txt", b"a,b\n1,2\n")
    assert response.status_code == 400
    assert ".csv" in response.get_json()["error"]


def test_upload_without_file(client):
    assert client.post("/api/upload").status_code == 400


def test_upload_rejects_oversized_file(tmp_path):
    app = create_app({"UPLOAD_DIR": tmp_path / "uploads", "MAX_CONTENT_LENGTH": 1024})
    response = upload(app.test_client(), "big.csv", b"a,b\n" + b"1,2\n" * 1000)
    assert response.status_code == 413
    assert "larger than" in response.get_json()["error"]


def test_upload_rejects_malformed_file_and_deletes_it(client, tmp_path):
    response = upload(client, "broken.csv", b"\x00\xff\x00 not a table")
    assert response.status_code == 400
    assert response.get_json()["error"]
    assert not list((tmp_path / "uploads").glob("*"))


def test_configure(client):
    dataset_id = upload(client).get_json()["dataset_id"]
    body = client.post("/api/configure", json={"dataset_id": dataset_id, "target": "Exited"})
    assert body.get_json()["task"] == "binary"


def test_unknown_job_is_404(client):
    assert client.get("/api/status/does-not-exist").status_code == 404
    assert client.get("/api/results/does-not-exist").status_code == 404


def test_train_refuses_unsupported_target(client):
    dataset_id = upload(client).get_json()["dataset_id"]
    response = client.post(
        "/api/train", json={"dataset_id": dataset_id, "target": "Balance", "models": ALL_ML}
    )
    assert response.status_code == 400
    assert "regression" in response.get_json()["error"]


def test_train_rejects_unknown_model(client):
    dataset_id = upload(client).get_json()["dataset_id"]
    response = client.post(
        "/api/train", json={"dataset_id": dataset_id, "target": "Exited", "models": ["dnn"]}
    )
    assert response.status_code == 400


def test_full_flow(client):
    dataset_id = upload(client).get_json()["dataset_id"]
    response = client.post(
        "/api/train", json={"dataset_id": dataset_id, "target": "Exited", "models": ALL_ML}
    )
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


def test_full_flow_multiclass(client):
    dataset_id = upload(client, "three_class.csv").get_json()["dataset_id"]
    response = client.post(
        "/api/train", json={"dataset_id": dataset_id, "target": "Segment", "models": ALL_ML}
    )
    status = wait_for(client, response.get_json()["job_id"])
    assert status["status"] == "complete", status["error"]

    body = client.get(f"/api/results/{response.get_json()['job_id']}").get_json()
    assert body["metadata"]["task"] == "multiclass"
    assert all(result["status"] == "done" for result in body["results"])
    assert all(result["roc_auc"] is None for result in body["results"])
    assert all(len(result["confusion_matrix"]) == 3 for result in body["results"])
