"""Shared fixtures for the black-box system tests (STP FSPP-MTP02.0, section 8.10).

* Starts its own copy of the application from the project's .venv on 127.0.0.1:5055 with
  COMPUTE_DEVICE=cpu and uploads redirected to the run folder. The application code and
  the project's uploads/ folder are not touched.
* Refuses any base URL that is not on this machine.
* Gives every test a fresh browser context (its own pp_session cookie).
* Saves step screenshots, attaches them to the HTML report, and writes results.csv with
  the outcome pytest actually recorded for each ST case.
"""

import csv
import json
import os
import socket
import subprocess  # nosec B404 - fixed local application and test commands only
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import APP_PYTHON, BASE_URL, LOCAL_HOSTS, PORT, REPO, RUN_DIR, SHOTS  # noqa: E402


def assert_local(url):
    host = urlparse(url).hostname
    if host not in LOCAL_HOSTS:
        raise pytest.UsageError(f"Refusing to test a non-local URL: {url}")


def _healthy(url, timeout=1.0):
    try:
        with urllib.request.urlopen(
            url + "/api/health", timeout=timeout
        ) as response:  # nosec B310 (local URL)
            return json.loads(response.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001 - any failure means "not up yet"
        return False


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


class AppServer:
    """The application under test, started from the repository with test settings."""

    def __init__(self, compute_device="cpu"):
        self.url = BASE_URL
        self.compute_device = compute_device
        self.process = None
        self.log_path = RUN_DIR / "app-server.log"

    def start(self):
        assert_local(self.url)
        if not _port_free(PORT):
            raise RuntimeError(f"Port {PORT} is already in use; set E2E_PORT to a free port.")
        env = dict(os.environ)
        env.update(
            PORT=str(PORT),
            APP_ENV="development",
            COMPUTE_DEVICE=self.compute_device,
            UPLOAD_DIR=str(RUN_DIR / "uploads"),
            PYTHONUNBUFFERED="1",
        )
        log = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed in stop()
        log.write(
            f"\n===== start {datetime.now():%H:%M:%S} COMPUTE_DEVICE={self.compute_device} =====\n"
        )
        log.flush()
        self._log = log
        self.process = subprocess.Popen(  # nosec B603 - fixed executable and app.py path
            [str(APP_PYTHON), "app.py"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Application exited early; see {self.log_path}")
            if _healthy(self.url):
                return self
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(
            f"Application did not answer /api/health within 90 s; see {self.log_path}"
        )

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if getattr(self, "_log", None):
            self._log.close()
            self._log = None
        deadline = time.time() + 15
        while not _port_free(PORT) and time.time() < deadline:
            time.sleep(0.3)

    def restart(self, compute_device=None):
        self.stop()
        if compute_device:
            self.compute_device = compute_device
        return self.start()


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def app_server():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    server = AppServer(os.environ.get("E2E_COMPUTE_DEVICE", "cpu")).start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def base_url(app_server):  # overrides pytest-base-url so every page.goto("/") is local
    assert_local(app_server.url)
    return app_server.url


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args, base_url):
    return {
        **browser_context_args,
        "base_url": base_url,
        "viewport": {"width": 1366, "height": 768},
        "accept_downloads": True,
    }


@pytest.fixture
def evidence(request):
    """Step screenshots and manual-review notes for the current test."""
    return Evidence(request)


class Evidence:
    def __init__(self, request):
        self.request = request
        self.node = request.node
        self.folder = SHOTS / request.node.name
        self.count = 0
        self.node.stash.setdefault(EVIDENCE_KEY, {"shots": [], "notes": [], "manual": []})

    def shot(self, page, label, full_page=True):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.count += 1
        path = self.folder / f"{self.count:02d}-{label}.png"
        page.screenshot(path=str(path), full_page=full_page)
        self.node.stash[EVIDENCE_KEY]["shots"].append(path)
        return path

    def note(self, text):
        self.node.stash[EVIDENCE_KEY]["notes"].append(text)

    def manual(self, text):
        """Something a person must still check; reported, never counted as passed."""
        self.node.stash[EVIDENCE_KEY]["manual"].append(text)


EVIDENCE_KEY = pytest.StashKey[dict]()
RESULTS = []


def pytest_configure(config):
    config.addinivalue_line("markers", "st(id): system test case ID from System_Test_Cases.xlsx")
    config.addinivalue_line("markers", "slow: long-running case (large dataset or full model set)")
    config.addinivalue_line("markers", "partial: partially automatable case; manual review remains")
    config.addinivalue_line("markers", "manual: manual-only case; never executed by the script")
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _st_id(item):
    marker = item.get_closest_marker("st")
    return marker.args[0] if marker else ""


def _report_message(report, call):
    if report.failed and call.excinfo is not None:
        return call.excinfo.exconly().splitlines()[0][:500]
    if report.skipped:
        longrepr = report.longrepr
        return longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    return ""


def _add_html_extras(item, report, data):
    try:
        from pytest_html import extras
    except ImportError:  # pragma: no cover
        return
    html_dir = (
        Path(item.config.option.htmlpath).resolve().parent
        if getattr(item.config.option, "htmlpath", None)
        else RUN_DIR
    )
    items = list(getattr(report, "extras", []))
    for text in data["notes"]:
        items.append(extras.text(text, name="Note"))
    for text in data["manual"]:
        items.append(extras.text(text, name="Manual review required"))
    for path in data["shots"]:
        rel = os.path.relpath(path, html_dir).replace("\\", "/")
        items.append(extras.image(rel, name=path.stem))
    report.extras = items


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    data = item.stash.get(EVIDENCE_KEY, {"shots": [], "notes": [], "manual": []})
    if report.when == "call" or (report.when == "setup" and report.skipped):
        _add_html_extras(item, report, data)
        RESULTS.append(
            {
                "test_case_id": _st_id(item),
                "test": item.nodeid,
                "pytest_outcome": report.outcome,
                "duration_s": round(report.duration, 1),
                "message": _report_message(report, call).replace("\n", " "),
                "manual_review_pending": "; ".join(data["manual"]),
                "screenshots": len(data["shots"]),
            }
        )
    elif report.when == "setup" and report.failed:
        RESULTS.append(
            {
                "test_case_id": _st_id(item),
                "test": item.nodeid,
                "pytest_outcome": "error",
                "duration_s": round(report.duration, 1),
                "message": (call.excinfo.exconly().splitlines()[0][:500] if call.excinfo else ""),
                "manual_review_pending": "",
                "screenshots": 0,
            }
        )


def pytest_sessionfinish(session, exitstatus):
    if not RESULTS:
        return
    path = RUN_DIR / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(RESULTS, key=lambda row: row["test_case_id"]))


def pytest_html_report_title(report):
    report.title = "System Test Execution Report - Full-Stack Prediction Platform"
