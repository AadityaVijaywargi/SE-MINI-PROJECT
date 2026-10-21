"""Upload validation and storage (SM-15, SM-16)."""

import io

import pytest
from werkzeug.datastructures import FileStorage

from core.storage import DatasetStore, UploadError, check_extension


def make_upload(text, filename):
    return FileStorage(stream=io.BytesIO(text.encode()), filename=filename)


def csv_text(rows, columns=2):
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
    assert list(tmp_path.iterdir()) == []


def test_too_few_columns_rejected(tmp_path):
    with pytest.raises(UploadError, match="column"):
        DatasetStore(tmp_path).save(make_upload(csv_text(60, columns=1), "narrow.csv"))


def test_unparseable_file_rejected_and_deleted(tmp_path):
    store = DatasetStore(tmp_path)
    with pytest.raises(UploadError, match="could not be read"):
        store.save(FileStorage(stream=io.BytesIO(b"PK\x03\x04garbage"), filename="bad.xlsx"))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dataset_id", ["../secret", "abc", None, "0" * 32])
def test_unknown_or_malformed_id_raises_key_error(tmp_path, dataset_id):
    with pytest.raises(KeyError):
        DatasetStore(tmp_path).get(dataset_id)
