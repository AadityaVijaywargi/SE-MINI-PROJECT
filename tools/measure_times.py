"""Measure API response times against a running server (SM-68, SM-98).

    python tools/measure_times.py                      # local server on port 5000
    python tools/measure_times.py https://<app>.onrender.com

Uploads the banking sample and two synthetic files five times each, calls /api/configure
after each upload, then runs one four-model job and polls /api/status until it finishes.
Prints median and maximum times; docs/performance.md records the results.
"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "banking.csv"
REPEATS = 5


class Client:
    """Minimal JSON and multipart client that times every call."""

    def __init__(self, base):
        if not base.startswith(("http://", "https://")):
            raise SystemExit("The base URL must start with http:// or https://")
        self.base = base.rstrip("/")
        self.cookie = None

    def call(self, method, path, body=None, upload=None):
        """Return ``(milliseconds, parsed JSON body)`` for one request."""
        headers, data = {}, None
        if upload:
            boundary = uuid.uuid4().hex
            name, content = upload
            data = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    f'filename="{name}"\r\nContent-Type: text/csv\r\n\r\n'
                ).encode()
                + content
                + f"\r\n--{boundary}--\r\n".encode()
            )
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.cookie:
            headers["Cookie"] = self.cookie
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers=headers
        )
        start = time.perf_counter()
        # The scheme is checked in __init__, so only http(s) URLs are opened.
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
            payload = json.loads(response.read())
            self.cookie = (response.headers.get("Set-Cookie") or self.cookie or "").split(";")[0]
        return (time.perf_counter() - start) * 1000, payload


def summary(values):
    """'median 12 ms, max 20 ms'."""
    return f"median {statistics.median(values):.0f} ms, max {max(values):.0f} ms"


def synthetic(rows, seed=0):
    """A CSV with numeric, categorical and binary target columns."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.integers(0, 50, rows),
            "c": rng.choice(["x", "y", "z"], rows),
            "d": rng.normal(size=rows),
            "target": rng.integers(0, 2, rows),
        }
    )
    return frame.to_csv(index=False).encode()


def measure_uploads(client):
    """Upload and configure timings per file size."""
    files = [("banking.csv (200 rows)", SAMPLE.read_bytes(), "Exited")]
    files += [(f"synthetic {rows:,} rows", synthetic(rows), "target") for rows in (10_000, 50_000)]
    for label, content, target in files:
        uploads, configures = [], []
        for _ in range(REPEATS):
            elapsed, body = client.call("POST", "/api/upload", upload=("perf.csv", content))
            uploads.append(elapsed)
            elapsed, _ = client.call(
                "POST", "/api/configure", {"dataset_id": body["dataset_id"], "target": target}
            )
            configures.append(elapsed)
        size = len(content) / 1e6
        print(
            f"{label} ({size:.2f} MB): upload {summary(uploads)}; configure {summary(configures)}"
        )


def measure_job(client):
    """Train, status-poll and results timings for one job."""
    _, body = client.call("POST", "/api/upload", upload=("perf.csv", synthetic(10_000)))
    request = {
        "dataset_id": body["dataset_id"],
        "target": "target",
        "models": ["random_forest", "svm", "dnn", "vqc"],
        "qml_epochs": 3,
    }
    train_ms, job = client.call("POST", "/api/train", request)
    polls = []
    while True:
        elapsed, status = client.call("GET", f"/api/status/{job['job_id']}")
        polls.append(elapsed)
        if status["status"] in ("complete", "failed", "cancelled"):
            break
        time.sleep(0.5)
    results_ms, results = client.call("GET", f"/api/results/{job['job_id']}")
    print(
        f"train 202: {train_ms:.0f} ms; status while training: {summary(polls)} over "
        f"{len(polls)} polls; results: {results_ms:.0f} ms; job took {status['elapsed']} s"
    )
    times = ", ".join(
        f"{result['key']} {result['train_time']:.1f}s"
        for result in results["results"]
        if result["status"] == "done"
    )
    print(f"per-model training time: {times}")


def main():
    """Parse the base URL and run both measurements."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:5000")
    client = Client(parser.parse_args().base)
    measure_uploads(client)
    measure_job(client)


if __name__ == "__main__":
    main()
