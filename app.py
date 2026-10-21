"""Flask REST API and static frontend (SRS 3.3).

Endpoints: /api/health, /api/upload, /api/configure, /api/train, /api/status/<job_id>,
/api/results/<job_id>. Every response, including errors, is JSON.
"""

import logging

from flask import Blueprint, Flask, current_app, jsonify, request
from werkzeug.exceptions import HTTPException

import config
from core.jobs import Job, JobRunner, JobStore
from core.profiling import preview, profile_dataframe
from core.storage import DatasetStore, UploadError
from core.targets import analyse_task, suggest_targets
from core.trainers import build_trainers

api = Blueprint("api", __name__, url_prefix="/api")


class RequestError(Exception):
    """A request the client must fix. Turned into ``{error: message}`` with ``status``."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def error(message, status):
    """JSON error body ``{error: message}`` with the given HTTP status."""
    return jsonify({"error": message}), status


def datasets():
    """The app's DatasetStore."""
    return current_app.extensions["datasets"]


def jobs():
    """The app's JobStore."""
    return current_app.extensions["jobs"]


def parse_settings(body):
    """Validate seed and test ratio from a /api/train body; returns a settings dict."""
    seed = body.get("seed", config.DEFAULT_SEED)
    test_ratio = body.get("test_ratio", config.DEFAULT_TEST_RATIO)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RequestError("Seed must be a non-negative whole number.")
    if test_ratio not in config.ALLOWED_TEST_RATIOS:
        raise RequestError("Test split must be 0.1, 0.2 or 0.3.")
    return {"seed": seed, "test_ratio": float(test_ratio)}


def load_target(body):
    """Load the dataset named in a JSON body and check its target column.

    Returns ``(dataset, frame, target)``; raises RequestError (404 or 400) otherwise.
    """
    try:
        dataset, frame = datasets().load(body.get("dataset_id"))
    except KeyError as exc:
        raise RequestError("Unknown dataset. Upload the file again.", 404) from exc
    target = body.get("target")
    if target not in frame.columns:
        raise RequestError(f"Column '{target}' is not in the dataset.")
    return dataset, frame, target


def find_job(job_id):
    """Return the job for ``job_id`` or raise a 404 RequestError (REQ-JB-4)."""
    job = jobs().get(job_id)
    if job is None:
        raise RequestError("Unknown job.", 404)
    return job


@api.get("/health")
def health():
    """Liveness check for the container and CI (PERF-6)."""
    return jsonify({"status": "ok"})


@api.post("/upload")
def upload():
    """Store, parse and profile an uploaded file (REQ-UP-2 to REQ-UP-7)."""
    file = request.files.get("file")
    if file is None or not file.filename:
        raise RequestError("Choose a .csv or .xlsx file to upload.")
    try:
        dataset, frame = datasets().save(file)
    except UploadError as exc:
        raise RequestError(str(exc)) from exc
    body = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "profile": profile_dataframe(frame),
        "suggested_targets": suggest_targets(frame),
        "preview": preview(frame),
    }
    return jsonify(body), 201


@api.post("/configure")
def configure():
    """Analyse the chosen target column (REQ-TG-4)."""
    _, frame, target = load_target(request.get_json(silent=True) or {})
    return jsonify(analyse_task(frame[target]))


@api.post("/train")
def train():
    """Create a training job and return its id immediately (REQ-JB-1, REQ-TG-6)."""
    body = request.get_json(silent=True) or {}
    dataset, frame, target = load_target(body)
    analysis = analyse_task(frame[target])
    if not analysis["trainable"]:
        raise RequestError(analysis["reason"])

    models = body.get("models") or []
    if not isinstance(models, list) or not models:
        raise RequestError("Select at least one model to train.")
    settings = parse_settings(body)
    try:
        trainers = build_trainers(models, settings["seed"])
    except ValueError as exc:
        raise RequestError(str(exc)) from exc

    settings["dataset_name"] = dataset.name
    job = Job(trainers)
    jobs().add(job)
    JobRunner(job, frame, target, trainers, settings).start()
    return jsonify({"job_id": job.job_id}), 202


@api.get("/status/<job_id>")
def status(job_id):
    """Current state of a job; 404 if unknown (REQ-JB-2, REQ-JB-4)."""
    return jsonify(find_job(job_id).status_payload())


@api.get("/results/<job_id>")
def results(job_id):
    """Results, summary and metadata of a job (SRS 3.3)."""
    return jsonify(find_job(job_id).results_payload())


@api.errorhandler(RequestError)
def request_error(exc):
    """Client errors raised by the route helpers."""
    return error(exc.message, exc.status)


def register_error_handlers(app):
    """JSON bodies for HTTP errors; a generic message for anything unexpected (SEC-5)."""

    @app.errorhandler(413)
    def too_large(_exc):
        return error(f"The file is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error(exc.description or exc.name, exc.code)

    @app.errorhandler(Exception)
    def unhandled(exc):
        app.logger.exception("Unhandled error: %s", exc)
        return error("Something went wrong on the server. Please try again.", 500)


def create_app(overrides=None):
    """Application factory. ``overrides`` replaces config values, e.g. in tests."""
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config.update(
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        UPLOAD_DIR=config.UPLOAD_DIR,
    )
    if overrides:
        app.config.update(overrides)

    app.extensions["datasets"] = DatasetStore(app.config["UPLOAD_DIR"])
    app.extensions["jobs"] = JobStore()
    app.register_blueprint(api)
    app.add_url_rule("/", "index", lambda: app.send_static_file("index.html"))
    register_error_handlers(app)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=config.PORT, debug=False, threaded=True)
