"""Asynchronous training jobs (SM-44 to SM-47, SM-51).

Implements REQ-JB-1 to REQ-JB-5 and REQ-JB-8: a job runs in a background thread, reports
real progress after each stage, and a crash marks it failed instead of killing the server.
"""

import logging
import threading
import time
import uuid

from core.evaluator import evaluate, failed_result, summarise
from core.preprocessing import Preprocessor

logger = logging.getLogger(__name__)

QUEUED, RUNNING, COMPLETE, FAILED = "queued", "running", "complete", "failed"


class Job:  # pylint: disable=too-many-instance-attributes
    """State of one training run (REQ-JB-2). Updated only by its JobRunner thread.

    A plain state holder, as in the SAD class diagram: every attribute is part of the
    status or results payload.
    """

    def __init__(self, trainers):
        self.job_id = uuid.uuid4().hex
        self.status = QUEUED
        self.progress = 0
        self.stage = "Queued"
        self.models = [
            {"key": t.key, "model": t.name, "paradigm": t.paradigm, "status": "pending"}
            for t in trainers
        ]
        self.log = []
        self.results = []
        self.summary = None
        self.metadata = None
        self.error = None
        self.started = time.time()
        self._lock = threading.Lock()

    def update(self, stage, progress=None, message=None):
        """Set the stage label, optionally the progress percent, and append to the log."""
        with self._lock:
            self.stage = stage
            if progress is not None:
                self.progress = int(progress)
            self.log.append(message or stage)

    def set_progress(self, progress):
        """Set the progress percent (0 to 100) without changing the stage or log."""
        with self._lock:
            self.progress = int(progress)

    def set_model_status(self, key, status, error=None):
        """Mark one model in the checklist as running, done or failed."""
        with self._lock:
            for entry in self.models:
                if entry["key"] == key:
                    entry["status"] = status
                    if error:
                        entry["error"] = error

    def status_payload(self):
        """Response body for GET /api/status/<job_id>."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "stage": self.stage,
                "models": [dict(entry) for entry in self.models],
                "log": list(self.log),
                "elapsed": round(time.time() - self.started, 1),
                "error": self.error,
            }

    def results_payload(self):
        """Response body for GET /api/results/<job_id>."""
        with self._lock:
            return {
                "results": list(self.results),
                "summary": self.summary,
                "metadata": self.metadata,
            }


class JobStore:
    """In-memory job dictionary (single Gunicorn worker, CON-2)."""

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def add(self, job):
        """Register a job so status polls can find it."""
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id):
        """Return the job for ``job_id`` or None if it is unknown."""
        with self._lock:
            return self._jobs.get(job_id)


class JobRunner:
    """Runs preprocessing once, then each trainer and the evaluator, for one job."""

    def __init__(self, job, frame, trainers, settings):
        """``settings`` holds target, seed, test_ratio and dataset_name for this run."""
        self.job = job
        self.frame = frame
        self.trainers = trainers
        self.settings = settings

    def start(self):
        """Run the job in a daemon thread and return immediately (REQ-JB-1)."""
        thread = threading.Thread(target=self.run, name=f"job-{self.job.job_id}", daemon=True)
        thread.start()
        return thread

    def run(self):
        """Execute the whole job. Never raises: failures end in status 'failed' (REQ-JB-8)."""
        job = self.job
        steps = 1 + len(self.trainers)
        try:
            job.status = RUNNING
            job.update("Preprocessing data", 0)
            preprocessor = Preprocessor(self.settings["seed"], self.settings["test_ratio"])
            split = preprocessor.fit_split(self.frame, self.settings["target"])
            job.update(
                "Preprocessing done",
                100 / steps,
                f"Preprocessing done: {split.x_train.shape[0]} train rows, "
                f"{split.x_test.shape[0]} test rows, {split.x_train.shape[1]} features",
            )

            for index, trainer in enumerate(self.trainers, start=1):
                self.train_one(split, trainer, index)
                job.set_progress(100 * (index + 1) / steps)

            job.summary = summarise(job.results)
            job.metadata = {
                "dataset": self.settings.get("dataset_name"),
                "rows": int(len(self.frame)),
                "target": self.settings["target"],
                "task": "binary" if split.n_classes == 2 else "multiclass",
                "seed": self.settings["seed"],
                "test_ratio": self.settings["test_ratio"],
                **split.metadata(),
            }
            job.status = COMPLETE
            job.update("Complete", 100)
        # REQ-JB-8 requires catching everything so the job, not the server, fails.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Job %s failed", job.job_id)
            job.error = str(exc) or exc.__class__.__name__
            job.status = FAILED
            job.update("Failed", message=f"Job failed: {job.error}")

    def train_one(self, split, trainer, index):
        """Train and evaluate a single model. One model failing does not stop the job."""
        job = self.job
        job.set_model_status(trainer.key, "running")
        job.update(f"Training {trainer.name} ({index} of {len(self.trainers)})")
        try:
            model, train_time = trainer.train(split)
            result = evaluate(model, split, trainer, train_time)
        # REQ-TR-8 requires one model's failure, whatever it is, not to stop the job.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Model %s failed in job %s", trainer.key, job.job_id)
            reason = str(exc) or exc.__class__.__name__
            job.results.append(failed_result(trainer, reason))
            job.set_model_status(trainer.key, "failed", reason)
            job.update(job.stage, message=f"{trainer.name} failed: {reason}")
            return
        job.results.append(result)
        job.set_model_status(trainer.key, "done")
        job.update(
            job.stage,
            message=f"{trainer.name} done in {train_time:.2f}s, accuracy "
            f"{result['accuracy']:.3f}",
        )
