"""Upload validation, storage, samples and cleanup (SM-15, SM-16, SM-20, SM-72)."""

import io
import os
import time

import pytest
from werkzeug.datastructures import FileStorage

from core.storage import SAMPLE_DATASETS, DatasetStore, UploadError, check_extension


def make_upload(text, filename):
    """A Werkzeug FileStorage holding ``text``, as Flask passes an uploaded file."""
    return FileStorage(stream=io.BytesIO(text.encode()), filename=filename)


def csv_text(rows, columns=2):
    """A numeric CSV with a header row, ``rows`` data rows and ``columns`` columns."""
    header = ",".join(f"c{i}" for i in range(columns))
    body = "\n".join(",".join(str(r + c) for c in range(columns)) for r in range(rows))
    return f"{header}\n{body}\n"


@pytest.mark.parametrize("name", ["data.csv", "DATA.CSV", "report.xlsx"])
def test_allowed_extensions(name):
    assert check_extension(name) in (".csv", ".xlsx")


@pytest.mark.parametrize("name", ["data.txt", "data.csv.exe", "noextension", "", None])
def test_rejected_extensions(name):
    with pytest.raises(UploadError):
        check_extension(name)


def test_saved_under_uuid_not_client_name(tmp_path):
    store = DatasetStore(tmp_path)
    dataset, frame = store.save(make_upload(csv_text(60), "../../etc/passwd.csv"))
    assert dataset.path.parent == tmp_path
    assert dataset.path.name == f"{dataset.dataset_id}.csv"
    assert "passwd" not in dataset.path.name
    assert len(frame) == 60


def test_too_few_rows_rejected_and_deleted(tmp_path):
    store = DatasetStore(tmp_path)
    with pytest.raises(UploadError, match="at least 50"):
        store.save(make_upload(csv_text(10), "small.csv"))
    assert not list(tmp_path.iterdir())


def test_too_few_columns_rejected(tmp_path):
    with pytest.raises(UploadError, match="column"):
        DatasetStore(tmp_path).save(make_upload(csv_text(60, columns=1), "narrow.csv"))


def test_unparseable_file_rejected_and_deleted(tmp_path):
    store = DatasetStore(tmp_path)
    with pytest.raises(UploadError, match="could not be read"):
        store.save(FileStorage(stream=io.BytesIO(b"PK\x03\x04garbage"), filename="bad.xlsx"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dataset_id", ["../secret", "abc", None, "0" * 32])
def test_unknown_or_malformed_id_raises_key_error(tmp_path, dataset_id):
    with pytest.raises(KeyError):
        DatasetStore(tmp_path).get(dataset_id)


def age(path, seconds):
    """Make ``path`` look ``seconds`` old."""
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_cleanup_deletes_uploads_older_than_ttl(tmp_path):
    store = DatasetStore(tmp_path, ttl=3600)
    old, _ = store.save(make_upload(csv_text(60), "old.csv"))
    new, _ = store.save(make_upload(csv_text(60), "new.csv"))
    age(old.path, 3601)
    assert store.cleanup() == 1
    assert not old.path.exists() and new.path.exists()
    with pytest.raises(KeyError):
        store.get(old.dataset_id)
    assert store.get(new.dataset_id) is new


def test_cleanup_runs_before_each_upload(tmp_path):
    store = DatasetStore(tmp_path, ttl=60)
    old, _ = store.save(make_upload(csv_text(60), "old.csv"))
    age(old.path, 61)
    store.save(make_upload(csv_text(60), "new.csv"))
    assert not old.path.exists()


def test_cleanup_keeps_the_folder_under_its_size_cap(tmp_path):
    probe = DatasetStore(tmp_path / "probe").save(make_upload(csv_text(60), "p.csv"))[0]
    size = probe.path.stat().st_size
    store = DatasetStore(tmp_path / "uploads", max_bytes=int(size * 2.5))
    saved = []
    for index, name in enumerate(["a.csv", "b.csv", "c.csv"]):
        dataset, _ = store.save(make_upload(csv_text(60), name))
        age(dataset.path, 100 - index)  # a.csv is the oldest
        saved.append(dataset)
    assert store.cleanup() == 1
    assert not saved[0].path.exists()
    assert saved[1].path.exists() and saved[2].path.exists()


def test_loading_a_cleaned_up_dataset_raises_key_error(tmp_path):
    store = DatasetStore(tmp_path)
    dataset, _ = store.save(make_upload(csv_text(60), "gone.csv"))
    dataset.path.unlink()
    with pytest.raises(KeyError):
        store.load(dataset.dataset_id)


@pytest.mark.parametrize("name", sorted(SAMPLE_DATASETS))
def test_each_sample_loads_under_a_new_uuid(tmp_path, name):
    store = DatasetStore(tmp_path)
    dataset, frame = store.save_sample(name)
    assert dataset.path.parent == tmp_path
    assert dataset.path.name == f"{dataset.dataset_id}.csv"
    assert dataset.name == SAMPLE_DATASETS[name]["file"]
    assert len(frame) == 200
    assert SAMPLE_DATASETS[name]["target"] in frame.columns


def test_unknown_sample_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        DatasetStore(tmp_path).save_sample("../../config")
