"""Upload storage and parsing (SM-15, SM-16).

Implements REQ-UP-2 to REQ-UP-4: extension allowlist, secure_filename, UUID storage
names, and rejection (with deletion) of files that cannot be parsed or are too small.
"""

import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from werkzeug.utils import secure_filename

import config

_DATASET_ID = re.compile(r"^[0-9a-f]{32}$")


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
    """Keeps uploaded files under UUID names and remembers their original names (REQ-UP-3)."""

    def __init__(self, upload_dir):
        self.upload_dir = Path(upload_dir)
        self._datasets = {}
        self._lock = threading.Lock()

    def save(self, file_storage):
        """Validate, store and parse an uploaded file.

        Returns ``(Dataset, DataFrame)``. On any failure the stored file is deleted and
        UploadError is raised, so no orphan file remains (REQ-UP-4).
        """
        extension = check_extension(file_storage.filename)
        dataset_id = uuid.uuid4().hex
        path = self.upload_dir / f"{dataset_id}{extension}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        file_storage.save(path)
        try:
            frame = read_table(path)
        except UploadError:
            path.unlink(missing_ok=True)
            raise

        dataset = Dataset(dataset_id, path, secure_filename(file_storage.filename))
        with self._lock:
            self._datasets[dataset_id] = dataset
        return dataset, frame

    def get(self, dataset_id):
        """Return the Dataset for ``dataset_id`` or raise KeyError if it is unknown."""
        if not isinstance(dataset_id, str) or not _DATASET_ID.match(dataset_id):
            raise KeyError(dataset_id)
        with self._lock:
            return self._datasets[dataset_id]

    def load(self, dataset_id):
        """Return ``(Dataset, DataFrame)`` for a previously uploaded dataset."""
        dataset = self.get(dataset_id)
        return dataset, read_table(dataset.path)
