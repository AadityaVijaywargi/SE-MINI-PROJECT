"""Flask REST API and static frontend (SRS 3.3).

Endpoints: /api/health, /api/samples, /api/samples/<name>, /api/upload, /api/configure,
/api/train, /api/status/<job_id>, /api/cancel/<job_id>, /api/results/<job_id> and
/api/results/<job_id>/csv. Every response except the CSV download, including errors, is
JSON.
"""

import logging
import re
import threading
import uuid

from flask import Blueprint, Flask, Response, current_app, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import config
from core.evaluator import THRESHOLD_AUTO
from core.export import results_csv
from core.jobs import Job, JobRunner, JobStore
from core.profiling import preview, profile_dataframe
from core.storage import SAMPLE_DATASETS, DatasetStore, UploadError
from core.targets import analyse_task, suggest_targets
from core.trainers import QML_MODELS, build_trainers, default_options

api = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)

SESSION_COOKIE = "pp_session"
_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
CLEANUP_INTERVAL_SECONDS = 600
# REQ-MS-5: Advanced-panel settings, with their inclusive limits.
OPTION_LIMITS = {
    "dnn_epochs": (1, config.DNN_MAX_EPOCHS, "DNN epochs"),
    "qml_epochs": (1, config.QML_MAX_EPOCHS, "QML epochs"),
    "qubits": (config.QML_MIN_QUBITS, config.QML_MAX_QUBITS, "Qubit count"),
}


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


def whole_number(value):
    """True for an int that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def parse_threshold(value):
    """``"auto"`` (tune on the validation rows) or a fixed probability for binary targets."""
    if value == THRESHOLD_AUTO:
        return value
    low, high = config.MIN_THRESHOLD, config.MAX_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise RequestError(f'Decision threshold must be "auto" or a number from {low} to {high}.')
    return float(value)


def parse_settings(body):
    """Validate seed, test ratio and decision threshold from a /api/train body; returns a
    settings dict."""
    seed = body.get("seed", config.DEFAULT_SEED)
    test_ratio = body.get("test_ratio", config.DEFAULT_TEST_RATIO)
    if not whole_number(seed) or seed < 0:
        raise RequestError("Seed must be a non-negative whole number.")
    if test_ratio not in config.ALLOWED_TEST_RATIOS:
        raise RequestError("Test split must be 0.1, 0.2 or 0.3.")
    threshold = parse_threshold(body.get("threshold", THRESHOLD_AUTO))
    return {"seed": seed, "test_ratio": float(test_ratio), "threshold": threshold}


def parse_options(body):
    """Validate the DNN epochs, QML epochs, qubit count (REQ-MS-5) and class balancing."""
    options = default_options()
    for key, (low, high, label) in OPTION_LIMITS.items():
        value = body.get(key, options[key])
        if not whole_number(value) or not low <= value <= high:
            raise RequestError(f"{label} must be a whole number from {low} to {high}.")
        options[key] = value
    balance = body.get("balance_classes", options["balance_classes"])
    if not isinstance(balance, bool):
        raise RequestError("Balance classes must be true or false.")
    options["balance_classes"] = balance
    return options


def load_target(body):
    """Load the dataset named in a JSON body and check its target column.

    Returns ``(dataset, frame, target)``; raises RequestError (404 or 400) otherwise.
    """
    try:
        dataset, frame = datasets().load(body.get("dataset_id"))
    except KeyError as exc:
        raise RequestError("Unknown or expired dataset. Upload the file again.", 404) from exc
    target = body.get("target")
    if target not in frame.columns:
        raise RequestError(f"Column '{target}' is not in the dataset.")
    return dataset, frame, target


def find_job(job_id):
    """Return the job for ``job_id`` or raise a 404 RequestError (REQ-JB-4)."""
    job = jobs().get(job_id)
    if job is None:
        raise RequestError("Unknown or expired job.", 404)
    return job


def session_id():
    """The browser's session id from its cookie, or a new one (REQ-JB-7)."""
    value = request.cookies.get(SESSION_COOKIE, "")
    return value if _SESSION_ID.match(value) else uuid.uuid4().hex


def dataset_body(dataset, frame):
    """Response body for an uploaded or sample dataset (REQ-UP-5 to REQ-UP-7)."""
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "profile": profile_dataframe(frame),
        "suggested_targets": suggest_targets(frame),
        "preview": preview(frame),
    }


@api.get("/health")
def health():
    """Liveness check for the container and CI (PERF-6)."""
    return jsonify({"status": "ok"})


@api.get("/samples")
def samples():
    """The bundled sample datasets offered as buttons (REQ-UP-9)."""
    return jsonify(
        [
            {"name": name, "label": sample["label"], "target": sample["target"]}
            for name, sample in SAMPLE_DATASETS.items()
        ]
    )


@api.post("/samples/<name>")
def load_sample(name):
    """Load a bundled sample exactly as if it had been uploaded (REQ-UP-9)."""
    if name not in SAMPLE_DATASETS:
        raise RequestError("Unknown sample dataset.", 404)
    dataset, frame = datasets().save_sample(name)
    return jsonify(dataset_body(dataset, frame)), 201


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
    return jsonify(dataset_body(dataset, frame)), 201


@api.post("/configure")
def configure():
    """Analyse the chosen target column (REQ-TG-4)."""
    _, frame, target = load_target(request.get_json(silent=True) or {})
    return jsonify(analyse_task(frame[target]))


@api.post("/train")
def train():
    """Create a training job and return its id immediately (REQ-JB-1, REQ-TG-6).

    One queued or running job per browser session (REQ-JB-7, SEC-6); a second one is
    refused with 409.
    """
    body = request.get_json(silent=True) or {}
    dataset, frame, target = load_target(body)
    analysis = analyse_task(frame[target])
    if not analysis["trainable"]:
        raise RequestError(analysis["reason"])

    models = body.get("models") or []
    if not isinstance(models, list) or not models:
        raise RequestError("Select at least one model to train.")
    if analysis["task"] != "binary" and any(key in QML_MODELS for key in models):
        raise RequestError(
            "The quantum classifier supports binary targets only. Untick it or choose a "
            "target with two classes."
        )
    settings = parse_settings(body)
    options = parse_options(body)
    try:
        trainers = build_trainers(models, settings["seed"], options)
    except ValueError as exc:
        raise RequestError(str(exc)) from exc

    session = session_id()
    if jobs().active_for(session) is not None:
        raise RequestError(
            "A training job is already running in this browser. Wait for it to finish or "
            "cancel it first.",
            409,
        )
    settings.update(target=target, dataset_name=dataset.name, options=options)
    job = Job(trainers, session_id=session)
    jobs().add(job)
    JobRunner(job, frame, trainers, settings).start()
    response = jsonify({"job_id": job.job_id})
    response.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="Lax")
    return response, 202


@api.get("/status/<job_id>")
def job_status(job_id):
    """Current state of a job; 404 if unknown (REQ-JB-2, REQ-JB-4)."""
    return jsonify(find_job(job_id).status_payload())


@api.post("/cancel/<job_id>")
def cancel_job(job_id):
    """Ask a job to stop; it stops before the next model or epoch (REQ-JB-6)."""
    job = find_job(job_id)
    if job.is_finished:
        raise RequestError(f"The job has already finished ({job.status}).", 409)
    job.request_cancel()
    logger.info("Job %s: cancel requested", job_id)
    return jsonify(job.status_payload()), 202


@api.get("/results/<job_id>")
def job_results(job_id):
    """Results, summary and metadata of a job (SRS 3.3)."""
    return jsonify(find_job(job_id).results_payload())


@api.get("/results/<job_id>/csv")
def job_results_csv(job_id):
    """The metrics table as a CSV download, formula characters escaped (REQ-RS-9, SEC-2)."""
    payload = find_job(job_id).results_payload()
    dataset = (payload["metadata"] or {}).get("dataset") or "results"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset.rsplit(".", 1)[0]) or "results"
    return Response(
        results_csv(payload["results"]),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_model_comparison.csv"'},
    )


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


def configure_cors(app):
    """Cross-origin access to /api (COM-3, SEC-3).

    Production allows only the origins in CORS_ORIGINS, and none if it is empty, since the
    UI is served from the same origin. Development also allows localhost on any port.
    """
    origins = list(config.CORS_ORIGINS)
    if not config.PRODUCTION:
        origins.append(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$")
    if origins:
        CORS(app, resources={r"/api/*": {"origins": origins}})


def configure_logging():
    """One log format for the app's own messages (OTH-4). Gunicorn keeps its own handlers."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    root.setLevel(logging.INFO)


def start_cleanup(store, interval=CLEANUP_INTERVAL_SECONDS):
    """Delete expired uploads every ``interval`` seconds in a daemon thread (SEC-7)."""
    stop = threading.Event()

    def loop():
        while not stop.wait(interval):
            try:
                store.cleanup()
            # A failed cleanup must never stop the timer or the server.
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Upload cleanup failed")

    threading.Thread(target=loop, name="upload-cleanup", daemon=True).start()
    return stop


def create_app(overrides=None):
    """Application factory. ``overrides`` replaces config values, e.g. in tests."""
    configure_logging()
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config.update(
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        UPLOAD_DIR=config.UPLOAD_DIR,
        UPLOAD_TTL_SECONDS=config.UPLOAD_TTL_SECONDS,
        UPLOAD_DIR_MAX_BYTES=config.UPLOAD_DIR_MAX_MB * 1024 * 1024,
        JOB_TTL_SECONDS=config.JOB_TTL_SECONDS,
    )
    if overrides:
        app.config.update(overrides)

    app.extensions["datasets"] = DatasetStore(
        app.config["UPLOAD_DIR"],
        ttl=app.config["UPLOAD_TTL_SECONDS"],
        max_bytes=app.config["UPLOAD_DIR_MAX_BYTES"],
    )
    app.extensions["jobs"] = JobStore(ttl=app.config["JOB_TTL_SECONDS"])
    app.register_blueprint(api)
    app.add_url_rule("/", "index", lambda: app.send_static_file("index.html"))
    register_error_handlers(app)
    configure_cors(app)
    if not app.config.get("TESTING"):
        start_cleanup(app.extensions["datasets"])
    return app


if __name__ == "__main__":
    # SEC-5: never run with the debugger, even locally.
    create_app().run(host="127.0.0.1", port=config.PORT, debug=False, threaded=True)
