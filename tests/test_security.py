"""Security validation, SEC-1 to SEC-8 (SM-96, TC-13).

SEC-6 (one job per session) and SEC-7 (upload cleanup) are also covered in test_api.py,
test_jobs.py and test_storage.py; this file checks each requirement at least once.
"""

import io
from pathlib import Path

import pytest

import app as app_module
import config
from app import create_app
from core.trainers import build_trainers
from tests.conftest import SAMPLES_DIR

ROOT = Path(__file__).resolve().parent.parent


def make_client(tmp_path, **overrides):
    """A test client for a fresh app."""
    return create_app(
        {"UPLOAD_DIR": tmp_path / "uploads", "TESTING": True, **overrides}
    ).test_client()


# SEC-1: uploads go through an allowlist, sanitised names, UUID paths and a size limit.
@pytest.mark.parametrize("name", ["../../evil.csv", "..\\..\\evil.csv", "/etc/passwd.csv"])
def test_sec1_client_filename_never_reaches_the_path(tmp_path, name):
    client = make_client(tmp_path)
    data = (SAMPLES_DIR / "banking.csv").read_bytes()
    body = client.post("/api/upload", data={"file": (io.BytesIO(data), name)}).get_json()
    stored = list((tmp_path / "uploads").iterdir())
    assert [path.name for path in stored] == [f"{body['dataset_id']}.csv"]
    assert not (tmp_path / "evil.csv").exists()
    assert "/" not in body["name"] and "\\" not in body["name"]


def test_sec1_size_limit_is_enforced(tmp_path):
    client = make_client(tmp_path, MAX_CONTENT_LENGTH=2048)
    response = client.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n" * 2000), "x.csv")})
    assert response.status_code == 413


def test_sec1_dataset_ids_must_be_uuids(tmp_path):
    client = make_client(tmp_path)
    for dataset_id in ("../uploads", "banking", "0" * 31, None, 42):
        response = client.post("/api/configure", json={"dataset_id": dataset_id, "target": "x"})
        assert response.status_code == 404


# SEC-2 is covered by tests/test_export.py.


# SEC-3: CORS origins are restricted in production.
def cors_client(tmp_path, monkeypatch, production, origins):
    """A client built with the given production flag and CORS origins."""
    monkeypatch.setattr(config, "PRODUCTION", production)
    monkeypatch.setattr(config, "CORS_ORIGINS", origins)
    return make_client(tmp_path)


def allowed_origin(client, origin):
    """The Access-Control-Allow-Origin header the API returns for ``origin``."""
    return client.get("/api/health", headers={"Origin": origin}).headers.get(
        "Access-Control-Allow-Origin"
    )


def test_sec3_production_without_origins_allows_no_cross_origin(tmp_path, monkeypatch):
    client = cors_client(tmp_path, monkeypatch, True, [])
    assert allowed_origin(client, "https://evil.example") is None
    assert allowed_origin(client, "http://localhost:3000") is None


def test_sec3_production_allows_only_listed_origins(tmp_path, monkeypatch):
    client = cors_client(tmp_path, monkeypatch, True, ["https://ui.example"])
    assert allowed_origin(client, "https://ui.example") == "https://ui.example"
    assert allowed_origin(client, "https://evil.example") is None


def test_sec3_development_allows_localhost_only(tmp_path, monkeypatch):
    client = cors_client(tmp_path, monkeypatch, False, [])
    assert allowed_origin(client, "http://localhost:5173") == "http://localhost:5173"
    assert allowed_origin(client, "https://evil.example") is None


# SEC-4: .env is never committed; .env.example documents every variable with dummy values.
def test_sec4_env_file_is_ignored_and_example_exists():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("APP_ENV", "CORS_ORIGINS", "UPLOAD_DIR", "JOB_TTL_SECONDS", "QML_MAX_ROWS"):
        assert f"{name}=" in example


# SEC-5: debug off, generic client errors, details logged server-side (TC-13).
def test_sec5_unexpected_errors_are_generic_and_logged(tmp_path, monkeypatch, caplog):
    def explode(_frame):
        raise RuntimeError("database password is hunter2 at /srv/app/core")

    monkeypatch.setattr(app_module, "suggest_targets", explode)
    client = make_client(tmp_path)
    data = (SAMPLES_DIR / "banking.csv").read_bytes()
    response = client.post("/api/upload", data={"file": (io.BytesIO(data), "banking.csv")})
    assert response.status_code == 500
    assert response.get_json() == {"error": "Something went wrong on the server. Please try again."}
    assert "hunter2" not in response.get_data(as_text=True)
    assert any("hunter2" in record.getMessage() for record in caplog.records)
    assert client.application.debug is False


def test_sec5_unknown_routes_answer_json(tmp_path):
    response = make_client(tmp_path).get("/api/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()


# SEC-6: bounded resources: QML row cap, SVM row cap and one job per session.
def test_sec6_row_caps_are_configured():
    _, vqc = build_trainers(["svm", "vqc"], seed=0)
    assert vqc.max_rows == config.QML_MAX_ROWS == 2000
    assert build_trainers(["svm"], seed=0)[0].max_rows == config.SVM_MAX_ROWS == 20000


# SEC-7 is covered by tests/test_storage.py (TTL and size cap).


# SEC-8: the container runs as a non-root user.
def test_sec8_dockerfile_runs_as_non_root():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    user_lines = [line.split()[1] for line in dockerfile.splitlines() if line.startswith("USER ")]
    assert user_lines and user_lines[-1] not in ("root", "0")
    assert "1000" in dockerfile
