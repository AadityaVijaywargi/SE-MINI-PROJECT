"""Flask REST API and static frontend (SRS 3.3).

Endpoints: /api/health, /api/upload, /api/configure, /api/train, /api/status/<job_id>,
/api/results/<job_id>. Every response, including errors, is JSON.
"""

import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import config
from core.jobs import Job, JobRunner, JobStore
from core.profiling import preview, profile_dataframe
from core.storage import DatasetStore, UploadError
from core.targets import analyse_task, suggest_targets
from core.trainers import build_trainers


def error(message, status):
    """JSON error body ``{error: message}`` with the given HTTP status."""
    return jsonify({"error": message}), status


def parse_settings(body):
    """Validate seed and test ratio from a /api/train body; returns a settings dict."""
    seed = body.get("seed", config.DEFAULT_SEED)
    test_ratio = body.get("test_ratio", config.DEFAULT_TEST_RATIO)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a non-negative whole number.")
    if test_ratio not in config.ALLOWED_TEST_RATIOS:
        raise ValueError("Test split must be 0.1, 0.2 or 0.3.")
    return {"seed": seed, "test_ratio": float(test_ratio)}


def create_app(overrides=None):
    """Application factory. ``overrides`` replaces config values, e.g. in tests."""
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config.update(
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        UPLOAD_DIR=config.UPLOAD_DIR,
    )
    if overrides:
        app.config.update(overrides)

    datasets = DatasetStore(app.config["UPLOAD_DIR"])
    jobs = JobStore()
    app.extensions["datasets"] = datasets
    app.extensions["jobs"] = jobs

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return error("Choose a .csv or .xlsx file to upload.", 400)
        try:
            dataset, frame = datasets.save(file)
        except UploadError as exc:
            return error(str(exc), 400)
        body = {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "profile": profile_dataframe(frame),
            "suggested_targets": suggest_targets(frame),
            "preview": preview(frame),
        }
        return jsonify(body), 201

    @app.post("/api/configure")
    def configure():
        body = request.get_json(silent=True) or {}
        try:
            _, frame = datasets.load(body.get("dataset_id"))
        except KeyError:
            return error("Unknown dataset. Upload the file again.", 404)
        target = body.get("target")
        if target not in frame.columns:
            return error(f"Column '{target}' is not in the dataset.", 400)
        return jsonify(analyse_task(frame[target]))

    @app.post("/api/train")
    def train():
        body = request.get_json(silent=True) or {}
        try:
            dataset, frame = datasets.load(body.get("dataset_id"))
        except KeyError:
            return error("Unknown dataset. Upload the file again.", 404)
        target = body.get("target")
        if target not in frame.columns:
            return error(f"Column '{target}' is not in the dataset.", 400)
        analysis = analyse_task(frame[target])
        if not analysis["trainable"]:
            return error(analysis["reason"], 400)

        models = body.get("models") or []
        if not isinstance(models, list) or not models:
            return error("Select at least one model to train.", 400)
        try:
            settings = parse_settings(body)
            trainers = build_trainers(models, settings["seed"])
        except ValueError as exc:
            return error(str(exc), 400)

        settings["dataset_name"] = dataset.name
        job = Job(trainers)
        jobs.add(job)
        JobRunner(job, frame, target, trainers, settings).start()
        return jsonify({"job_id": job.job_id}), 202

    @app.get("/api/status/<job_id>")
    def status(job_id):
        job = jobs.get(job_id)
        if job is None:
            return error("Unknown job.", 404)
        return jsonify(job.status_payload())

    @app.get("/api/results/<job_id>")
    def results(job_id):
        job = jobs.get(job_id)
        if job is None:
            return error("Unknown job.", 404)
        return jsonify(job.results_payload())

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

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=config.PORT, debug=False, threaded=True)
