"""Group B: dataset selection and preprocessing (ST-09 to ST-12)."""

import re

import pytest
from playwright.sync_api import expect

from pages import App, text

SAMPLES = [
    ("Banking", "Exited", "banking.csv"),
    ("Healthcare", "Diagnosis", "healthcare.csv"),
    ("Telecom", "Churn", "telecom.csv"),
    ("Insurance", "Fraud", "insurance.csv"),
]

WATCH_LOADING = """() => {
  window.__loading = [];
  const box = document.getElementById('upload-progress');
  new MutationObserver(() => {
    if (!box.classList.contains('hidden')) {
      const buttons = [...document.querySelectorAll('#sample-buttons button')];
      window.__loading.push({text: document.getElementById('upload-progress-text').textContent,
                             disabled: buttons.every(b => b.disabled)});
    }
  }).observe(box, {attributes: true, subtree: true, childList: true, characterData: true});
}"""


@pytest.mark.st("ST-09")
def test_st09_load_each_sample_dataset(page, evidence):
    app = App(page).open()
    labels = [text(b) for b in page.locator("#sample-buttons button").all()]
    assert labels == [f"{label} · {target}" for label, target, _ in SAMPLES]
    evidence.shot(page, "sample-buttons", full_page=False)
    page.evaluate(WATCH_LOADING)
    chooser_opened = []
    page.on("filechooser", lambda c: chooser_opened.append(c))
    for label, _, file in SAMPLES:
        app.load_sample(label)
        expect(page.locator("#dataset-name")).to_have_text(file)
        assert app.summary_cards()["Rows"][0] == "200"
        evidence.shot(page, f"profile-{label.lower()}")
    loading = page.evaluate("window.__loading")
    for label, _, _ in SAMPLES:
        hits = [x for x in loading if x["text"] == f"Loading the {label} sample…"]
        assert hits, f"loading text for {label} was not shown"
        assert all(
            x["disabled"] for x in hits
        ), f"sample buttons were not disabled while loading {label}"
    assert chooser_opened == [], "a file picker opened for a sample button"


@pytest.mark.st("ST-10")
def test_st10_identifier_and_constant_columns_flagged(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    rows = app.column_rows()
    evidence.shot(page, "banking-columns")
    assert rows["CustomerId"]["role"] == "identifier"
    assert "Dropped: looks like an ID" in rows["CustomerId"]["warning"]
    assert app.summary_cards()["Columns"][1] == "1 will be dropped"
    assert rows["Exited"]["role"] == "feature"
    assert all(r["role"] == "feature" for name, r in rows.items() if name != "CustomerId")

    app.load_sample("Insurance")
    rows = app.column_rows()
    evidence.shot(page, "insurance-columns")
    assert rows["Country"]["role"] == "constant"
    assert "Dropped: only one value" in rows["Country"]["warning"]
    assert rows["Fraud"]["role"] == "feature"
    assert all(r["role"] == "feature" for name, r in rows.items() if name != "Country")


@pytest.mark.st("ST-11")
def test_st11_missing_values_counted_and_blanked(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    rows = app.column_rows()
    evidence.shot(page, "missing-values")
    assert rows["Balance"]["missing"] == "30 (15%)"
    assert rows["CreditScore"]["missing"] == "15 (7.5%)"
    assert all(r["missing"] == "0" for n, r in rows.items() if n not in ("Balance", "CreditScore"))
    assert app.summary_cards()["Missing values"][0] == "45"
    heads = [text(h) for h in page.locator("#preview-table thead th").all()]
    for column in ("Balance", "CreditScore"):
        index = heads.index(column) + 1
        cells = page.locator(f"#preview-table tbody tr td:nth-child({index})").all()
        for cell in cells:
            empty = text(cell) == ""
            styled = "missing" in (cell.get_attribute("class") or "")
            assert empty == styled, f"{column}: empty cell not styled as missing (or the reverse)"
    missing_shown = page.locator("#preview-table td.missing").count()
    evidence.note(f"Missing cells among the 10 preview rows: {missing_shown}.")


@pytest.mark.st("ST-12")
def test_st12_preprocessing_reported_in_metadata(page, evidence):
    app = App(page).open()
    app.load_sample("Telecom")
    app.to_configure()
    assert page.locator("#target-select").input_value() == "Churn"
    app.train(["logistic_regression"])
    meta = app.metadata()
    evidence.shot(page, "metadata")
    assert meta["Ordinal columns"] == "City"
    assert set(meta["One-hot columns"].split(", ")) == {
        "Contract",
        "InternetService",
        "PaymentMethod",
    }
    assert set(meta["Numeric columns"].split(", ")) == {"Tenure", "MonthlyCharges"}
    assert meta["Dropped columns"] == "none"
    assert meta["Positive class"] == "Yes"
    match = re.fullmatch(
        r"(\d+) / (\d+) / (\d+) \(test split 20%\)", meta["Train / validation / test rows"]
    )
    assert match, meta["Train / validation / test rows"]
    train, val, test = map(int, match.groups())
    assert train + val + test == 200 and test == 40
    assert meta["Split fingerprint"]
    assert page.locator("#metrics-table tbody tr.failed-row").count() == 0
    evidence.note(f"Train / validation / test rows: {train} / {val} / {test}.")
