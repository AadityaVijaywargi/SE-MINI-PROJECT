"""Group K: performance (ST-50). Reuses the project's own tools/measure_times.py."""

import re
import subprocess

import pytest

from settings import APP_PYTHON, REPO

UPLOAD = re.compile(
    r"^(?P<label>.+?) \((?P<mb>[\d.]+) MB\): upload median (?P<up>\d+) ms, max (?P<upmax>\d+) ms; "
    r"configure median (?P<cf>\d+) ms, max (?P<cfmax>\d+) ms$"
)
JOB = re.compile(
    r"^train 202: (?P<train>\d+) ms; status while training: median (?P<st>\d+) ms, max (?P<stmax>\d+) ms "
    r"over (?P<polls>\d+) polls; results: (?P<res>\d+) ms"
)


@pytest.mark.slow
@pytest.mark.st("ST-50")
def test_st50_api_response_times(base_url, evidence):
    run = subprocess.run(  # noqa: S603 - fixed command, local URL only
        [str(APP_PYTHON), "tools/measure_times.py", base_url],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    output = run.stdout + run.stderr
    evidence.folder.mkdir(parents=True, exist_ok=True)
    (evidence.folder / "measure_times-output.txt").write_text(output, encoding="utf-8")
    evidence.note(output.strip())
    assert run.returncode == 0, f"measure_times.py exited with {run.returncode}"

    uploads = [m.groupdict() for m in map(UPLOAD.match, output.splitlines()) if m]
    job = next((m.groupdict() for m in map(JOB.match, output.splitlines()) if m), None)
    assert len(uploads) == 3 and job, "could not read the tool's output"
    failures = []
    for u in uploads:
        if float(u["mb"]) <= 10 and int(u["up"]) >= 2000:
            failures.append(f"upload {u['label']}: median {u['up']} ms (target < 2000 ms)")
        if int(u["cf"]) >= 500:
            failures.append(f"configure {u['label']}: median {u['cf']} ms (target < 500 ms)")
    if int(job["train"]) >= 200:
        failures.append(f"train 202: {job['train']} ms (target < 200 ms)")
    if int(job["st"]) >= 200:
        failures.append(f"status while training: median {job['st']} ms (target < 200 ms)")
    if int(job["res"]) >= 500:
        failures.append(f"results: {job['res']} ms (target < 500 ms)")
    evidence.note("Pass rule: medians of 5 calls for upload/configure and of all polls for status, as in docs/performance.md; maxima are recorded above.")
    assert not failures, failures
