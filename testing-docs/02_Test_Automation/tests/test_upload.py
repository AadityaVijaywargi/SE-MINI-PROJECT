"""Group A: dataset upload and validation (ST-01 to ST-08)."""

import base64
from pathlib import Path

import pytest
from playwright.sync_api import expect

from pages import App, sample, td

WATCH_PROGRESS = """() => {
  window.__progressSeen = [];
  const box = document.getElementById('upload-progress');
  new MutationObserver(() => {
    if (!box.classList.contains('hidden'))
      window.__progressSeen.push(document.getElementById('upload-progress-text').textContent);
  }).observe(box, {attributes: true, subtree: true, childList: true, characterData: true});
}"""


def upload_requests(page):
    seen = []
    page.on("request", lambda r: seen.append(r.url) if "/api/upload" in r.url else None)
    return seen


@pytest.mark.st("ST-01")
def test_st01_upload_valid_csv_with_file_picker(page, evidence):
    app = App(page).open()
    page.evaluate(WATCH_PROGRESS)
    app.choose_file(sample("banking.csv"))
    expect(page.locator("#profile")).to_be_visible()
    evidence.shot(page, "profile")

    assert (
        page.evaluate("window.__progressSeen").__len__() > 0
    ), "upload progress text never appeared"
    expect(page.locator("#upload-progress")).to_be_hidden()
    expect(page.locator("#dataset-name")).to_have_text("banking.csv")
    cards = app.summary_cards()
    assert cards["Rows"][0] == "200"
    assert cards["Columns"] == ("9", "1 will be dropped")
    assert cards["Missing values"][0] == "45"
    assert cards["Memory"][0], "Memory card is empty"
    expect(page.locator("#preview-table tbody tr")).to_have_count(10)
    heads = [h.text_content().strip() for h in page.locator("#column-table thead th").all()]
    assert heads == ["Column", "Type", "Unique", "Missing", "Role", "Examples"]
    expect(page.locator("#column-table tbody tr")).to_have_count(9)
    assert page.evaluate("document.activeElement.id") == "to-configure"


@pytest.mark.st("ST-02")
def test_st02_upload_valid_csv_by_drag_and_drop(page, evidence):
    app = App(page).open()
    path = Path(sample("healthcare.csv"))
    payload = base64.b64encode(path.read_bytes()).decode()
    dragging = page.evaluate(
        """([b64, name]) => {
          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const dt = new DataTransfer();
          dt.items.add(new File([bytes], name, {type: 'text/csv'}));
          window.__dt = dt;
          const zone = document.getElementById('dropzone');
          zone.dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true, cancelable: true}));
          zone.dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
          return zone.classList.contains('dragging');
        }""",
        [payload, path.name],
    )
    evidence.shot(page, "dragging-over-dropzone", full_page=False)
    assert dragging, "drop zone was not highlighted while a file was dragged over it"
    page.evaluate(
        """() => document.getElementById('dropzone').dispatchEvent(
             new DragEvent('drop', {dataTransfer: window.__dt, bubbles: true, cancelable: true}))"""
    )
    expect(page.locator("#profile")).to_be_visible()
    evidence.shot(page, "profile")
    expect(page.locator("#dataset-name")).to_have_text("healthcare.csv")
    cards = app.summary_cards()
    assert cards["Rows"][0] == "200"
    assert cards["Columns"] == ("7", "1 will be dropped")
    assert cards["Missing values"][0] == "0"
    expect(page.locator("#preview-table tbody tr")).to_have_count(10)
    expect(page.locator("#column-table tbody tr")).to_have_count(7)
    evidence.note("Drag-and-drop is simulated with DOM drag events carrying the real file bytes.")


@pytest.mark.st("ST-03")
def test_st03_upload_valid_excel_file(page, evidence):
    app = App(page).open()
    app.upload(td("TD-03_banking.xlsx"))
    evidence.shot(page, "profile")
    expect(page.locator("#dataset-name")).to_have_text("TD-03_banking.xlsx")
    cards = app.summary_cards()
    assert cards["Rows"][0] == "200"
    assert cards["Columns"][0] == "9"
    assert cards["Missing values"][0] == "45"
    assert app.column_rows()["CustomerId"]["role"] == "identifier"


@pytest.mark.st("ST-04")
def test_st04_reject_disallowed_extensions(page, evidence):
    app = App(page).open()
    sent = upload_requests(page)
    for name in ("TD-01_notes.txt", "TD-02_lookalike.csv.exe"):
        app.clear_toasts()
        app.choose_file(td(name))
        text = app.wait_toast("is not a .csv or .xlsx file")
        evidence.shot(page, f"toast-{name}", full_page=False)
        assert text == f'"{name}" is not a .csv or .xlsx file.'
    page.wait_for_timeout(500)
    assert sent == [], f"unexpected upload requests: {sent}"
    expect(page.locator("#profile")).to_be_hidden()


@pytest.mark.st("ST-05")
def test_st05_reject_file_over_100_mb_in_browser(page, evidence, tmp_path):
    big = tmp_path / "TD-14_oversize_101MB.csv"
    block = Path(sample("banking.csv")).read_bytes()
    header, body = block.split(b"\n", 1)
    with big.open("wb") as handle:
        handle.write(header + b"\n")
        while handle.tell() < 101 * 1024 * 1024:
            handle.write(body)
    evidence.note(f"TD-14 generated at run time: {big.stat().st_size:,} bytes.")
    app = App(page).open()
    sent = upload_requests(page)
    app.choose_file(str(big))
    text = app.wait_toast("the limit is 100 MB")
    evidence.shot(page, "oversize-toast", full_page=False)
    assert text.startswith('"TD-14_oversize_101MB.csv" is ')
    assert text.endswith("; the limit is 100 MB.")
    page.wait_for_timeout(500)
    assert sent == [], "the oversize file was sent to the server"


@pytest.mark.st("ST-06")
def test_st06_reject_empty_and_corrupt_files(page, evidence):
    app = App(page).open()
    sent = upload_requests(page)
    app.choose_file(td("TD-07_empty.csv"))
    assert app.wait_toast("is empty") == '"TD-07_empty.csv" is empty.'
    evidence.shot(page, "empty-toast", full_page=False)
    page.wait_for_timeout(500)
    assert sent == [], "the empty file was sent to the server"
    app.clear_toasts()
    app.choose_file(td("TD-08_corrupt.xlsx"))
    text = app.wait_toast("could not be read as a table")
    evidence.shot(page, "corrupt-toast", full_page=False)
    assert text == (
        "The file could not be read as a table. Check that it is a valid CSV or Excel file "
        "with a header row."
    )
    expect(page.locator("#profile")).to_be_hidden()


@pytest.mark.st("ST-07")
def test_st07_minimum_row_boundary(page, evidence):
    app = App(page).open()
    app.choose_file(td("TD-04_rows_49.csv"))
    text = app.wait_toast("49 rows")
    evidence.shot(page, "49-rows-rejected", full_page=False)
    assert text == "The file has 49 rows; at least 50 are needed to train and test a model."
    expect(page.locator("#profile")).to_be_hidden()
    app.upload(td("TD-05_rows_50.csv"))
    evidence.shot(page, "50-rows-accepted")
    assert app.summary_cards()["Rows"][0] == "50"


@pytest.mark.st("ST-08")
def test_st08_reject_single_column_file(page, evidence):
    app = App(page).open()
    app.choose_file(td("TD-06_one_column.csv"))
    text = app.wait_toast("1 column")
    evidence.shot(page, "one-column-rejected", full_page=False)
    assert (
        text == "The file has 1 column; at least 2 are needed (one or more features plus a target)."
    )
    expect(page.locator("#profile")).to_be_hidden()
