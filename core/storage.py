"""Upload storage and parsing (SM-15, SM-16, SM-20, SM-72).

Implements REQ-UP-2 to REQ-UP-4, REQ-UP-9 and SEC-7: extension allowlist,
secure_filename, UUID storage names, rejection (with deletion) of files that cannot be
parsed or are too small, the bundled sample datasets, and deletion of old uploads.
"""

import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename

import config

_DATASET_ID = re.compile(r"^[0-9a-f]{32}$")
logger = logging.getLogger(__name__)

# REQ-UP-9: the four domain samples offered as buttons. Only these names can be loaded,
# so a request can never name an arbitrary file.
SAMPLE_DATASETS = {
    "banking": {"file": "banking.csv", "label": "Banking", "target": "Exited"},
    "healthcare": {"file": "healthcare.csv", "label": "Healthcare", "target": "Diagnosis"},
    "telecom": {"file": "telecom.csv", "label": "Telecom", "target": "Churn"},
    "insurance": {"file": "insurance.csv", "label": "Insurance", "target": "Fraud"},
}


class UploadError(ValueError):
    """An uploaded file was rejected. The message is safe to show to the user."""


@dataclass
class Dataset:
    """A stored upload: its generated id, path on disk and the user's original file name."""

    dataset_id: str
    path: Path
    name: str


def check_extension(filename):
    """Return the lower-case extension of ``filename`` or raise UploadError (REQ-UP-2)."""
    safe_name = secure_filename(filename or "")
    extension = Path(safe_name).suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        shown = extension or "no extension"
        raise UploadError(f"Only .csv and .xlsx files are accepted (got {shown}).")
    return extension


def plural(count, word):
    """'1 row', '3 rows'."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def read_table(path):
    """Parse a stored .csv or .xlsx file and enforce the minimum size (REQ-UP-4).

    Returns a DataFrame. Raises UploadError with an actionable message on any failure.
    """
    try:
        if path.suffix == ".xlsx":
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path, encoding_errors="replace")
    except Exception as exc:
        raise UploadError(
            "The file could not be read as a table. Check that it is a valid CSV or Excel "
            "file with a header row."
        ) from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if len(frame) < config.MIN_ROWS:
        raise UploadError(
            f"The file has {plural(len(frame), 'row')}; at least {config.MIN_ROWS} are needed "
            "to train and test a model."
        )
    if frame.shape[1] < config.MIN_COLUMNS:
        raise UploadError(
            f"The file has {plural(frame.shape[1], 'column')}; at least {config.MIN_COLUMNS} "
            "are needed (one or more features plus a target)."
        )
    return frame


class DatasetStore:
    """Keeps uploaded files under UUID names and remembers their original names (REQ-UP-3).

    Files older than ``ttl`` seconds are deleted, and the oldest files go first whenever
    the folder grows past ``max_bytes`` (SEC-7). Cleanup runs before every new upload.
    """

    def __init__(
        self,
        upload_dir,
        ttl=config.UPLOAD_TTL_SECONDS,
        max_bytes=config.UPLOAD_DIR_MAX_MB * 1024 * 1024,
        samples_dir=config.SAMPLES_DIR,
    ):
        self.upload_dir = Path(upload_dir)
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.samples_dir = Path(samples_dir)
        self._datasets = {}
        self._lock = threading.Lock()

    def cleanup(self, now=None):
        """Delete expired uploads, then the oldest ones until under the size cap.

        Returns the number of files deleted.
        """
        if not self.upload_dir.exists():
            return 0
        now = time.time() if now is None else now
        files = sorted(
            (path for path in self.upload_dir.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        doomed = [path for path in files if now - path.stat().st_mtime > self.ttl]
        kept = [path for path in files if path not in doomed]
        total = sum(path.stat().st_size for path in kept)
        while kept and total > self.max_bytes:
            oldest = kept.pop(0)
            total -= oldest.stat().st_size
            doomed.append(oldest)

        for path in doomed:
            path.unlink(missing_ok=True)
            with self._lock:
                self._datasets.pop(path.stem, None)
        if doomed:
            logger.info("Upload cleanup deleted %d file(s)", len(doomed))
        return len(doomed)

    def _register(self, path, dataset_id, name):
        """Parse a stored file and remember it; delete it if it cannot be used."""
        try:
            frame = read_table(path)
        except UploadError:
            path.unlink(missing_ok=True)
            raise
        dataset = Dataset(dataset_id, path, name)
        with self._lock:
            self._datasets[dataset_id] = dataset
        logger.info(
            "Upload received: %s, %d bytes, %d rows, %d columns",
            name,
            path.stat().st_size,
            len(frame),
            frame.shape[1],
        )
        return dataset, frame

    def save(self, file_storage):
        """Validate, store and parse an uploaded file.

        Returns ``(Dataset, DataFrame)``. On any failure the stored file is deleted and
        UploadError is raised, so no orphan file remains (REQ-UP-4).
        """
        extension = check_extension(file_storage.filename)
        self.cleanup()
        dataset_id = uuid.uuid4().hex
        path = self.upload_dir / f"{dataset_id}{extension}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        file_storage.save(path)
        return self._register(path, dataset_id, secure_filename(file_storage.filename))

    def save_sample(self, name):
        """Copy one bundled sample into the upload folder and parse it (REQ-UP-9).

        Raises KeyError for a name that is not in SAMPLE_DATASETS.
        """
        sample = SAMPLE_DATASETS[name]
        self.cleanup()
        dataset_id = uuid.uuid4().hex
        path = self.upload_dir / f"{dataset_id}.csv"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.samples_dir / sample["file"], path)
        return self._register(path, dataset_id, sample["file"])

    def get(self, dataset_id):
        """Return the Dataset for ``dataset_id`` or raise KeyError if it is unknown."""
        if not isinstance(dataset_id, str) or not _DATASET_ID.match(dataset_id):
            raise KeyError(dataset_id)
        with self._lock:
            return self._datasets[dataset_id]

    def load(self, dataset_id):
        """Return ``(Dataset, DataFrame)`` for a previously uploaded dataset.

        Raises KeyError if it is unknown or its file has been cleaned up.
        """
        dataset = self.get(dataset_id)
        if not dataset.path.exists():
            with self._lock:
                self._datasets.pop(dataset_id, None)
            raise KeyError(dataset_id)
        return dataset, read_table(dataset.path)
