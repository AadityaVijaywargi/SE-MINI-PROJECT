"""Job runner and job store (SM-43 to SM-51, SM-56, SM-78)."""

# A Job starts with summary, metadata and error set to None and fills them in when it runs;
# pylint cannot follow that, so it flags every lookup on a finished job.
# pylint: disable=unsubscriptable-object,unsupported-membership-test

import logging
import threading
import time

import pytest

import config
from core import jobs as jobs_module
from core.jobs import CANCELLED, COMPLETE, FAILED, Job, JobRunner, JobStore
from core.trainers import Trainer, build_trainers
from tests.conftest import load_sample


class BrokenTrainer(Trainer):
    """Always raises while training, like a model with an invalid hyperparameter (TC-08)."""

    key = "broken"
    name = "Broken"
    paradigm = "ML"

    def fit(self, split, report):
        raise ValueError("invalid hyperparameter C=-1")


class SlowTrainer(Trainer):
    """Reports progress for ``steps`` ticks, sleeping ``delay`` seconds between them."""

    paradigm = "DL"

    def __init__(self, key, steps=50, delay=0.02):
        super().__init__()
        self.key = key
        self.name = key.title()
        self.steps = steps
        self.delay = delay
        self.started = threading.Event()

    def fit(self, split, report):
        self.started.set()
        for step in range(self.steps):
            time.sleep(self.delay)
            report((step + 1) / self.steps, f"step {step + 1}")
        return build_trainers(["naive_bayes"], 0)[0].fit(split, report)


def settings(**extra):
    """Run settings for the banking sample."""
    return {
        "target": "Exited",
        "seed": 42,
        "test_ratio": 0.2,
        "dataset_name": "banking.csv",
        **extra,
    }


def run_job(trainers, frame=None, **extra):
    """Run a job synchronously and return it."""
    job = Job(trainers)
    JobRunner(
        job, load_sample("banking") if frame is None else frame, trainers, settings(**extra)
    ).run()
    return job


def test_one_failing_model_does_not_stop_the_job():
    trainers = [BrokenTrainer(), *build_trainers(["naive_bayes"], 42)]
    job = run_job(trainers)
    assert job.status == COMPLETE
    failed, done = job.results[0], job.results[1]
    assert failed == {
        "key": "broken",
        "model": "Broken",
        "paradigm": "ML",
        "status": "failed",
        "error": "invalid hyperparameter C=-1",
    }
    assert done["status"] == "done"
    assert job.summary["best_overall"] == "Naive Bayes"
    assert job.models[0]["status"] == "failed"
    assert job.models[0]["error"] == "invalid hyperparameter C=-1"


def test_unhandled_job_error_marks_failed_and_logs_traceback(caplog):
    with caplog.at_level(logging.ERROR, logger="core.jobs"):
        job = run_job(build_trainers(["naive_bayes"], 42), target="NotAColumn")
    assert job.status == FAILED
    assert "NotAColumn" in job.error
    assert job.finished is not None
    assert any(record.exc_info for record in caplog.records)


def test_production_hides_unexpected_job_errors(monkeypatch):
    monkeypatch.setattr(config, "PRODUCTION", True)
    runner = JobRunner(Job([]), None, [], settings())
    assert runner.client_error(KeyError("/srv/app/secret")) == jobs_module.GENERIC_JOB_ERROR


def test_production_still_shows_data_problems(monkeypatch):
    monkeypatch.setattr(config, "PRODUCTION", True)
    job = run_job(build_trainers(["naive_bayes"], 42), target="NotAColumn")
    assert "NotAColumn" in job.error


def test_metadata_always_has_qml_fields():
    job = run_job(build_trainers(["naive_bayes"], 42), options={"dnn_epochs": 100})
    assert job.metadata["qml_subsampled_to"] is None
    assert job.metadata["pca_variance_retained"] is None
    assert job.metadata["dnn_epochs"] == 100
    for key in ("dataset", "rows", "features", "target", "task", "seed", "split_fingerprint"):
        assert job.metadata[key] is not None


def test_qml_metadata_recorded_when_vqc_runs():
    trainers = build_trainers(["vqc"], 42, {"qml_epochs": 1, "qubits": 2})
    job = run_job(trainers)
    assert job.results[0]["status"] == "done"
    assert job.metadata["qml_subsampled_to"] == job.metadata["train_rows"]
    assert 0 < job.metadata["pca_variance_retained"] <= 1


def test_progress_moves_inside_a_long_model():
    trainer = SlowTrainer("slow", steps=20, delay=0.01)
    job = Job([trainer])
    seen = set()
    runner = JobRunner(job, load_sample("banking"), [trainer], settings())
    thread = runner.start()
    while thread.is_alive():
        seen.add(job.progress)
        time.sleep(0.005)
    assert job.status == COMPLETE
    assert len({value for value in seen if 50 < value < 100}) >= 3
    assert all(0 <= value <= 100 for value in seen)


def test_cancel_stops_before_the_next_model():
    slow = SlowTrainer("slow", steps=1000, delay=0.005)
    later = build_trainers(["naive_bayes"], 42)[0]
    job = Job([slow, later])
    thread = JobRunner(job, load_sample("banking"), [slow, later], settings()).start()
    assert slow.started.wait(10)
    job.request_cancel()
    thread.join(10)
    assert job.status == CANCELLED
    assert [entry["status"] for entry in job.models] == ["cancelled", "cancelled"]
    assert not job.results
    assert "cancelled by the user" in job.log[-1]


def test_cancelled_job_keeps_finished_results():
    first = build_trainers(["naive_bayes"], 42)[0]
    slow = SlowTrainer("slow", steps=1000, delay=0.005)
    job = Job([first, slow])
    thread = JobRunner(job, load_sample("banking"), [first, slow], settings()).start()
    assert slow.started.wait(10)
    job.request_cancel()
    thread.join(10)
    assert job.status == CANCELLED
    assert [result["key"] for result in job.results] == ["naive_bayes"]
    assert job.summary["best_overall"] == "Naive Bayes"
    assert job.metadata["test_rows"] == 40


def test_store_evicts_finished_jobs_after_ttl():
    store = JobStore(ttl=60)
    old, fresh, running = Job([]), Job([]), Job([])
    for job in (old, fresh, running):
        store.add(job)
    old.finished = time.time() - 61
    fresh.finished = time.time()
    running.started = time.time() - 10_000
    assert store.get(old.job_id) is None
    assert store.get(fresh.job_id) is fresh
    assert store.get(running.job_id) is running


def test_one_active_job_per_session():
    store = JobStore()
    job = Job([], session_id="a" * 32)
    store.add(job)
    assert store.active_for("a" * 32) is job
    assert store.active_for("b" * 32) is None
    assert store.active_for(None) is None
    job.status = COMPLETE
    assert store.active_for("a" * 32) is None


@pytest.mark.parametrize("status", [COMPLETE, FAILED, CANCELLED])
def test_finished_statuses(status):
    job = Job([])
    job.finish(status, "Done")
    assert job.is_finished
    assert job.status_payload()["status"] == status


def test_status_elapsed_stops_when_the_job_finishes():
    job = Job([])
    job.started = time.time() - 5
    job.finish(COMPLETE, "Complete")
    first = job.status_payload()["elapsed"]
    time.sleep(0.05)
    assert job.status_payload()["elapsed"] == first
