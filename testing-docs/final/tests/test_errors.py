"""Group I: security, invalid input and error handling (ST-46, ST-47)."""

import time

import pytest
from playwright.sync_api import expect

from pages import App, sample

LEAK_MARKERS = ("Traceback", 'File "', "C:\\", "/home/", "site-packages")


def finished_job(api):
    dataset = api.post("/api/samples/banking").json()["dataset_id"]
    response = api.post("/api/train", data={"dataset_id": dataset, "target": "Exited", "models": ["logistic_regression"]})
    assert response.status == 202, response.text()
    job = response.json()["job_id"]
    for _ in range(240):
        if api.get(f"/api/status/{job}").json()["status"] == "complete":
            return job
        time.sleep(0.5)
    raise AssertionError("helper job did not complete")


@pytest.mark.st("ST-46")
def test_st46_invalid_api_requests_return_safe_json_errors(page, evidence):
    api = page.request
    job = finished_job(api)
    checks = [
        ("GET", "/api/status/00000000000000000000000000000000", None, 404, "Unknown or expired job."),
        ("POST", "/api/samples/..%2F..%2Fconfig", None, (404, 405), None),
        ("GET", "/api/does-not-exist", None, 404, None),
        ("POST", "/api/configure", {"dataset_id": "abc", "target": "Exited"}, 404, "Unknown or expired dataset. Upload the file again."),
        ("POST", f"/api/cancel/{job}", {}, 409, "The job has already finished (complete)."),
    ]
    log = []
    for method, path, body, status, message in checks:
        response = api.fetch(path, method=method, data=body) if body is not None else api.fetch(path, method=method)
        raw = response.text()
        log.append(f"{method} {path} -> {response.status} {raw.strip()}")
        allowed = status if isinstance(status, tuple) else (status,)
        assert response.status in allowed, f"{method} {path}: {response.status}"
        payload = response.json()
        assert isinstance(payload.get("error"), str) and payload["error"], f"{path}: no JSON error body"
        if message:
            assert payload["error"] == message, f"{path}: {payload['error']!r}"
        assert not any(marker in raw for marker in LEAK_MARKERS), f"{path} leaks internals: {raw}"
    (evidence.folder).mkdir(parents=True, exist_ok=True)
    (evidence.folder / "api-responses.txt").write_text("\n".join(log), encoding="utf-8")
    evidence.note("\n".join(log))
    evidence.note("Same rules are also asserted by developer pytest tests (see Test_Automation_Mapping.xlsx); this run is system-level evidence against the live server.")


@pytest.mark.st("ST-47")
def test_st47_messages_when_server_unreachable_or_dataset_expired(page, app_server, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()

    # Step 4 of the case: the server restarts, so the dataset held in memory is gone.
    app_server.restart()
    page.click("#start-training")
    expired = app.wait_toast("Upload the file again")
    evidence.shot(page, "expired-dataset", full_page=False)
    assert expired == "Unknown or expired dataset. Upload the file again."

    # Steps 2 and 3: the server is stopped.
    app_server.stop()
    try:
        app.clear_toasts()
        page.select_option("#target-select", "Geography")
        unreachable = app.wait_toast("Cannot reach the server")
        evidence.shot(page, "server-unreachable", full_page=False)
        assert unreachable == "Cannot reach the server. Check that it is running and try again."
        page.click("#back-to-upload")
        app.clear_toasts()
        app.choose_file(sample("banking.csv"))
        failed = app.wait_toast("Upload failed")
        evidence.shot(page, "upload-failed", full_page=False)
        assert failed == "Upload failed: cannot reach the server."
    finally:
        app_server.start()

    app.clear_toasts()
    app.upload(sample("banking.csv"))
    expect(page.locator("#dataset-name")).to_have_text("banking.csv")
    evidence.shot(page, "upload-works-again", full_page=False)
    evidence.note("Order run: expired dataset (step 4) first, then server stopped (steps 2 and 3), so each message comes from a real server state.")
