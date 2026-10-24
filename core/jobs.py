"""Asynchronous training jobs (SM-44 to SM-51, SM-56, SM-78).

Implements REQ-JB-1 to REQ-JB-8: a job runs in a background thread, reports real progress
after each stage (and inside long models), can be cancelled, is forgotten JOB_TTL_SECONDS
after it finishes, and a crash marks it failed instead of killing the server.
"""

import logging
import threading
import time
import uuid

import config
from core.evaluator import evaluate, failed_result, summarise
from core.preprocessing import Preprocessor, PreprocessingError

logger = logging.getLogger(__name__)

QUEUED, RUNNING, COMPLETE, FAILED, CANCELLED = (
    "queued",
    "running",
    "complete",
    "failed",
    "cancelled",
)
FINISHED = (COMPLETE, FAILED, CANCELLED)
GENERIC_JOB_ERROR = "Training failed on the server. Please try again or choose other models."


class JobCancelled(Exception):
    """Raised inside a job when the user has asked it to stop (REQ-JB-6)."""


class Job:  # pylint: disable=too-many-instance-attributes
    """State of one training run (REQ-JB-2). Updated only by its JobRunner thread.

    A plain state holder, as in the SAD class diagram: every attribute is part of the
    status or results payload, or of the store's eviction and per-session rules.
    """

    def __init__(self, trainers, session_id=None):
        self.job_id = uuid.uuid4().hex
        self.session_id = session_id
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
        self.finished = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_finished(self):
        """True once the job is complete, failed or cancelled."""
        return self.status in FINISHED

    @property
    def cancel_requested(self):
        """True once ``request_cancel`` has been called."""
        return self._cancel.is_set()

    def request_cancel(self):
        """Ask the job to stop at the next check (REQ-JB-6)."""
        self._cancel.set()

    def update(self, stage, progress=None, message=None):
        """Set the stage label, optionally the progress percent, and append to the log."""
        with self._lock:
            self.stage = stage
            if progress is not None:
                self.progress = int(progress)
            self.log.append(message or stage)

    def set_stage(self, stage, progress):
        """Set the stage label and progress percent without writing to the log."""
        with self._lock:
            self.stage = stage
            self.progress = int(progress)

    def set_progress(self, progress):
        """Set the progress percent (0 to 100) without changing the stage or log."""
        with self._lock:
            self.progress = int(progress)

    def set_model_status(self, key, status, error=None):
        """Mark one model in the checklist as running, testing, done, failed or cancelled."""
        with self._lock:
            for entry in self.models:
                if entry["key"] == key:
                    entry["status"] = status
                    if error:
                        entry["error"] = error

    def finish(self, status, stage, message=None):
        """Move to a final status and record when, for TTL eviction (REQ-JB-7)."""
        self.status = status
        self.finished = time.time()
        self.update(stage, 100 if status == COMPLETE else None, message)

    def status_payload(self):
        """Response body for GET /api/status/<job_id>."""
        with self._lock:
            end = self.finished or time.time()
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "stage": self.stage,
                "models": [dict(entry) for entry in self.models],
                "log": list(self.log),
                "elapsed": round(end - self.started, 1),
                "error": self.error,
                "cancel_requested": self.cancel_requested,
            }

    def results_payload(self):
        """Response body for GET /api/results/<job_id>."""
        with self._lock:
            return {
                "status": self.status,
                "results": list(self.results),
                "summary": self.summary,
                "metadata": self.metadata,
            }


class JobStore:
    """In-memory job dictionary (single Gunicorn worker, CON-2).

    Finished jobs older than ``ttl`` seconds are evicted on every access, so a poll for
    them returns 404 (REQ-JB-7, SRS B.2).
    """

    def __init__(self, ttl=config.JOB_TTL_SECONDS):
        self.ttl = ttl
        self._jobs = {}
        self._lock = threading.Lock()

    def _evict(self):
        """Drop finished jobs past their TTL. The caller holds the lock."""
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished is not None and now - job.finished > self.ttl
        ]
        for job_id in expired:
            del self._jobs[job_id]
            logger.info("Job %s evicted after %ss", job_id, self.ttl)

    def add(self, job):
        """Register a job so status polls can find it."""
        with self._lock:
            self._evict()
            self._jobs[job.job_id] = job

    def get(self, job_id):
        """Return the job for ``job_id`` or None if it is unknown or expired."""
        with self._lock:
            self._evict()
            return self._jobs.get(job_id)

    def active_for(self, session_id):
        """The session's queued or running job, if any (REQ-JB-7, SEC-6)."""
        if session_id is None:
            return None
        with self._lock:
            for job in self._jobs.values():
                if job.session_id == session_id and not job.is_finished:
                    return job
        return None


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

    def check_cancel(self):
        """Raise JobCancelled if the user asked the job to stop (REQ-JB-6)."""
        if self.job.cancel_requested:
            raise JobCancelled()

    def run(self):
        """Execute the whole job. Never raises: failures end in status 'failed' (REQ-JB-8)."""
        job = self.job
        steps = 1 + len(self.trainers)
        split = None
        logger.info(
            "Job %s started: dataset %s, target %s, models %s",
            job.job_id,
            self.settings.get("dataset_name"),
            self.settings["target"],
            ", ".join(trainer.key for trainer in self.trainers),
        )
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
                self.check_cancel()
                self.train_one(split, trainer, index, steps)
                job.set_progress(100 * (index + 1) / steps)

            self.finalise(split)
            job.finish(COMPLETE, "Complete")
        except JobCancelled:
            done = sum(1 for result in job.results if result.get("status") == "done")
            for entry in job.models:
                if entry["status"] in ("pending", "running", "testing"):
                    job.set_model_status(entry["key"], "cancelled")
            if split is not None:
                self.finalise(split)
            job.finish(
                CANCELLED,
                "Cancelled",
                f"Job cancelled by the user; {done} of {len(self.trainers)} models finished.",
            )
        # REQ-JB-8 requires catching everything so the job, not the server, fails.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Job %s failed", job.job_id)
            job.error = self.client_error(exc)
            job.finish(FAILED, "Failed", f"Job failed: {job.error}")
        logger.info(
            "Job %s finished: %s in %.1fs", job.job_id, job.status, time.time() - job.started
        )

    @staticmethod
    def client_error(exc):
        """The message shown to the user. In production only data problems are shown in
        full; anything unexpected gets a generic message (SEC-5)."""
        message = str(exc) or exc.__class__.__name__
        if config.PRODUCTION and not isinstance(exc, PreprocessingError):
            return GENERIC_JOB_ERROR
        return message

    def finalise(self, split):
        """Summary and run metadata (REQ-EV-4, REQ-EV-5), also for a cancelled job."""
        job = self.job
        job.summary = summarise(job.results)
        qml = {"qml_subsampled_to": None, "pca_variance_retained": None}
        done = {result["key"] for result in job.results if result.get("status") == "done"}
        for trainer in self.trainers:
            if trainer.key in done and hasattr(trainer, "run_metadata"):
                qml.update(trainer.run_metadata())
        job.metadata = {
            "dataset": self.settings.get("dataset_name"),
            "rows": int(len(self.frame)),
            "target": self.settings["target"],
            "task": "binary" if split.n_classes == 2 else "multiclass",
            "seed": self.settings["seed"],
            "test_ratio": self.settings["test_ratio"],
            "models_requested": len(self.trainers),
            **self.settings.get("options", {}),
            **split.metadata(),
            **qml,
        }

    def progress_reporter(self, trainer, index, steps, position):
        """Callback a trainer calls inside a long model: moves the bar between this model's
        step and the next, and stops the model if the job was cancelled."""
        job = self.job

        def report(fraction, message=None):
            self.check_cancel()
            fraction = min(max(float(fraction), 0.0), 1.0)
            stage = f"Training {trainer.name} {position}"
            job.set_stage(
                f"{stage}: {message}" if message else stage, 100 * (index + fraction) / steps
            )

        return report

    def train_one(self, split, trainer, index, steps):
        """Train a single model, then test it on the held-out rows. One model failing does
        not stop the job (REQ-TR-8); a cancel does."""
        job = self.job
        position = f"({index} of {len(self.trainers)})"
        job.set_model_status(trainer.key, "running")
        job.update(f"Training {trainer.name} {position}")
        try:
            model, train_time = trainer.train(
                split, self.progress_reporter(trainer, index, steps, position)
            )
            job.set_model_status(trainer.key, "testing")
            job.update(
                f"Testing {trainer.name} {position}",
                message=f"{trainer.name} trained in {train_time:.2f}s, testing on "
                f"{split.x_test.shape[0]} held-out rows",
            )
            result = evaluate(model, split, trainer, train_time)
        except JobCancelled:
            raise
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
        logger.info(
            "Job %s: %s done in %.2fs, accuracy %.3f",
            job.job_id,
            trainer.key,
            train_time,
            result["accuracy"],
        )
        job.update(
            job.stage,
            message=f"{trainer.name} tested in {result['test_time']:.2f}s: test accuracy "
            f"{result['accuracy']:.3f}, F1 {result['f1']:.3f}",
        )
