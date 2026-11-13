"""Group E: ML, DL and QML training (ST-26 to ST-35)."""

import re
import subprocess
import time

import pytest
from playwright.sync_api import expect

from pages import MODEL_KEYS, App, number, sample, td, text
from settings import TEST_DATA


def td_path(name):
    return TEST_DATA / name

ALL_COLUMNS = ["Model", "Paradigm", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
               "Threshold", "Train time", "Test time"]


def banking(page):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    return app


def split_rows(meta):
    match = re.fullmatch(r"(\d+) / (\d+) / (\d+) \(test split (\d+)%\)", meta["Train / validation / test rows"])
    assert match, meta["Train / validation / test rows"]
    return tuple(int(x) for x in match.groups())


def stage_check(evidence, ui_seen, server, large_seen=None):
    epoch_stages = [s for s in server if "epoch" in s]
    evidence.note(
        f"Server stage labels seen with 0.1 s polling: {len(server)} ({len(epoch_stages)} with an epoch). "
        f"Epoch stage rendered in the UI label on the Banking run: {'yes' if ui_seen else 'no'}"
        + ("" if large_seen is None else f"; on the TD-12 confirmation run: {'yes' if large_seen else 'no'}") + "."
    )
    if not (ui_seen or large_seen):
        evidence.manual("The epoch stage label was not rendered on screen in either run (the UI refreshes every 2 s). Confirm by watching a longer run.")


def label_on_large_data(app, evidence, models, pattern, tag):
    """Train the same models on TD-12 (40,000 rows) so stages last longer than the 2 s UI
    refresh, and screenshot the stage label the first time it matches ``pattern``."""
    page = app.page
    page.click("#new-run")
    app.upload(td("TD-12_large_40000.csv"))
    app.to_configure()
    app.select_target("Exited")
    app.set_models(models)
    app.start()
    seen = False
    deadline = time.time() + 600
    while time.time() < deadline and not app.screen("results").is_visible():
        label = text(page.locator("#stage-label"))
        if not seen and pattern.fullmatch(label):
            seen = True
            evidence.shot(page, f"{tag}-label-on-screen", full_page=False)
        time.sleep(0.2)
    app.wait_results(timeout=600_000)
    return seen


def matrix_total(page, model):
    figure = page.locator("#confusion-grid figure", has=page.locator("figcaption", has_text=model))
    return sum(int(text(td_)) for td_ in figure.locator("tbody td").all())


@pytest.mark.st("ST-26")
def test_st26_train_all_eight_models_binary(page, evidence):
    app = banking(page)
    seconds = app.train(MODEL_KEYS)
    evidence.shot(page, "results")
    elapsed = app.elapsed_seconds()
    heads, rows = app.table()
    assert heads == ALL_COLUMNS
    assert len(rows) == 8
    assert page.locator("#metrics-table tbody tr.failed-row").count() == 0
    assert [r["Paradigm"] for r in rows].count("ML") == 6
    assert [r["Paradigm"] for r in rows].count("DL") == 1
    assert [r["Paradigm"] for r in rows].count("QML") == 1
    for row in rows:
        for column in ALL_COLUMNS[2:]:
            assert number(row[column]) is not None, f"{row['Model']}: {column} is empty"
    evidence.note(f"Elapsed timer {elapsed} s; wall clock from Start training to results {seconds:.1f} s.")
    evidence.note("Scores: " + "; ".join(f"{r['Model']} acc {r['Accuracy']} F1 {r['F1']} ROC-AUC {r['ROC-AUC']} PR-AUC {r['PR-AUC']}" for r in rows))
    assert elapsed < 120, f"run took {elapsed} s (PERF-8 target under 2 minutes)"


@pytest.mark.st("ST-27")
def test_st27_train_seven_models_multiclass(page, evidence):
    app = App(page).open()
    app.upload(sample("three_class.csv"))
    app.to_configure()
    app.train(MODEL_KEYS[:-1])
    evidence.shot(page, "results")
    heads, rows = app.table()
    assert len(rows) == 7
    assert not {"ROC-AUC", "PR-AUC", "Threshold"} & set(heads)
    expect(page.locator("#roc-hidden")).to_be_visible()
    expect(page.locator("#roc-panel")).to_be_hidden()
    expect(page.locator("#pr-panel")).to_be_hidden()
    assert text(page.locator("#winner-banner .winner-kicker")) == "Best model · ranked by F1, then accuracy"
    qml_card = page.locator("#paradigm-cards .card-qml")
    assert "Binary targets only; not run for this target." in text(qml_card)
    first = page.locator("#confusion-grid figure").first
    labels = [text(th) for th in first.locator("thead th").all()][1:]
    assert sorted(labels) == ["Bronze", "Gold", "Silver"]
    expect(first.locator("tbody tr")).to_have_count(3)
    assert app.metadata()["Decision threshold"] == "not used (multi-class: most probable class)"


@pytest.mark.st("ST-28")
def test_st28_seed_reproducibility(page, evidence):
    app = banking(page)
    runs = []
    for seed in (42, 42, 7):
        app.set_option("opt-seed", seed)
        app.train(["logistic_regression"])
        meta = app.metadata()
        runs.append((seed, meta["Split fingerprint"], split_rows(meta)))
        evidence.shot(page, f"metadata-seed-{seed}-run{len(runs)}", full_page=False)
        page.click("#retrain")
        expect(app.screen("configure")).to_be_visible()
    evidence.note("; ".join(f"seed {s}: fingerprint {f}, rows {r}" for s, f, r in runs))
    assert runs[0][1] == runs[1][1], "same seed gave different fingerprints"
    assert runs[0][2] == runs[1][2], "same seed gave different row counts"
    assert runs[2][1] != runs[0][1], "a different seed gave the same fingerprint"


@pytest.mark.st("ST-29")
def test_st29_dnn_epoch_limit_and_early_stopping(page, evidence):
    app = banking(page)
    app.set_option("opt-dnn-epochs", 5)
    app.set_models(["dnn"])
    app.record_training()
    job = app.start_and_get_job()
    server = app.server_stages(job)
    app.wait_results()
    evidence.shot(page, "results")
    pattern = re.compile(r"Training Deep Neural Network \(1 of 1\): epoch \d+ of up to 5.*")
    assert any(pattern.fullmatch(s) for s in server), server
    ui_seen = any(pattern.fullmatch(h["stage"]) for h in app.training_history())
    details = app.model_details("Deep Neural Network")
    meta_epochs = app.metadata()["DNN epochs (max) / QML epochs / qubits"]
    evidence.note(f"DNN details: {details}")
    run = int(details["Epochs run"])
    assert run <= 5
    assert details["Stopped early"] == ("yes" if run < 5 else "no")
    assert "Validation rows" in details and "Device" in details
    assert meta_epochs.startswith("5 /")
    large = None if ui_seen else label_on_large_data(app, evidence, ["dnn"], pattern, "td12-dnn")
    stage_check(evidence, ui_seen, server, large)


@pytest.mark.st("ST-30")
def test_st30_qml_settings_applied_and_reported(page, evidence):
    app = banking(page)
    app.set_option("opt-qubits", 2)
    app.set_option("opt-qml-epochs", 3)
    app.set_models(["vqc"])
    app.record_training()
    job = app.start_and_get_job()
    server = app.server_stages(job)
    app.wait_results()
    evidence.shot(page, "results")
    pattern = re.compile(r"Training Variational Quantum Classifier \(1 of 1\): epoch \d of 3, loss \d+\.\d{3}")
    assert any(pattern.fullmatch(s) for s in server), server
    ui_seen = any(pattern.fullmatch(h["stage"]) for h in app.training_history())
    d = app.model_details("Variational Quantum Classifier")
    evidence.note(f"VQC details: {d}")
    assert (d["Qubits"], d["Layers"], d["Epochs"]) == ("2", "2", "3")
    assert int(d["Training rows"]) <= 2000
    assert "PCA variance retained" in d
    assert d["Simulator"] == "default.qubit" and d["Device"] == "CPU"
    meta = app.metadata()
    assert meta["QML subsampled to"] == d["Training rows"]
    assert re.fullmatch(r"\d+(\.\d)?%", meta["PCA variance retained"])
    assert meta["DNN epochs (max) / QML epochs / qubits"].endswith("/ 3 / 2")
    large = None if ui_seen else label_on_large_data(app, evidence, ["vqc"], pattern, "td12-vqc")
    stage_check(evidence, ui_seen, server, large)


@pytest.mark.slow
@pytest.mark.st("ST-31")
def test_st31_qml_row_cap_and_default_run_time(page, evidence):
    app = App(page).open()
    app.upload(td("TD-12_large_40000.csv"))
    app.to_configure()
    app.select_target("Exited")
    app.train(["vqc"], timeout=600_000)
    evidence.shot(page, "results")
    d = app.model_details("Variational Quantum Classifier")
    meta = app.metadata()
    _, rows = app.table()
    train_time = number(rows[0]["Train time"])
    _, _, test_rows, _ = split_rows(meta)
    evidence.note(f"VQC details: {d}; train time {rows[0]['Train time']}; metadata test rows {test_rows}.")
    assert d["Training rows"] == "2000" and d["Subsampled"] == "yes"
    assert meta["QML subsampled to"] == "2000"
    assert test_rows == 8000
    assert matrix_total(page, "Variational Quantum Classifier") == test_rows
    assert train_time <= 180, f"default QML run took {train_time} s (PERF-4 upper bound 180 s)"


@pytest.mark.st("ST-32")
def test_st32_threshold_modes_reported(page, evidence):
    app = banking(page)
    app.train(["logistic_regression", "xgboost"])
    meta = app.metadata()
    _, rows = app.table()
    evidence.shot(page, "auto-threshold")
    assert meta["Decision threshold"] == "tuned per model on the validation rows (best F1)"
    for row in rows:
        assert 0.01 <= number(row["Threshold"]) <= 0.99, row
    auto = {r["Model"]: r["Threshold"] for r in rows}

    page.click("#retrain")
    app.open_advanced()
    page.select_option("#opt-threshold-mode", "fixed")
    app.set_option("opt-threshold", "0.5")
    app.start()
    app.wait_results()
    meta = app.metadata()
    _, rows = app.table()
    evidence.shot(page, "fixed-threshold")
    assert meta["Decision threshold"] == "fixed at 0.5"
    assert all(r["Threshold"] == "0.500" for r in rows)
    for caption in page.locator("#confusion-grid .cm-counts").all():
        assert text(caption).endswith("threshold 0.500")
    evidence.note(f"Auto thresholds: {auto}.")


@pytest.mark.st("ST-33")
def test_st33_class_balancing_reported(page, evidence):
    app = banking(page)
    app.train(["logistic_regression"])
    evidence.shot(page, "balanced", full_page=False)
    assert app.metadata()["Class weights"] == "balanced"
    assert app.model_details("Logistic Regression")["Class weights"] == "balanced"
    page.click("#retrain")
    app.open_advanced()
    page.locator("#opt-balance").set_checked(False)
    blocked = app.start_disabled()
    page.locator("#start-training").scroll_into_view_if_needed()
    evidence.shot(page, "start-after-change-models", full_page=False)
    if blocked:
        # Workaround so the reporting requirement can still be checked: toggling a model
        # checkbox makes the page re-evaluate the Start training button.
        app.set_models([])
        app.set_models(["logistic_regression"])
    app.start()
    app.wait_results()
    evidence.shot(page, "unbalanced", full_page=False)
    assert app.metadata()["Class weights"] == "none"
    assert app.model_details("Logistic Regression")["Class weights"] == "none"
    if blocked:
        pytest.fail(
            "Step 2 failed: after '← Change models', unticking 'Balance class weights' left "
            "Start training disabled, so the run could not be started as specified. Class "
            "balancing itself was reported correctly after a workaround (toggling a model)."
        )


@pytest.mark.slow
@pytest.mark.st("ST-34")
def test_st34_svm_subsampling_on_large_dataset(page, evidence, tmp_path):
    app = App(page).open()
    app.upload(td("TD-12_large_40000.csv"))
    app.to_configure()
    app.select_target("Exited")
    hint = page.locator("#model-grid .group-ml .hint")
    evidence.shot(page, "svm-warning", full_page=False)
    assert text(hint) == "⚠ 40,000 rows: SVM will train on a stratified sample of 20,000 rows."
    app.train(["svm"], timeout=900_000)
    evidence.shot(page, "results")
    d = app.model_details("SVM")
    evidence.note(f"SVM details: {d}")
    assert d["Training rows"] == "20000"
    match = re.fullmatch(r"from (\d+) rows", d["Subsampled"])
    assert match and int(match.group(1)) > 20000, d

    # Observation for SRS REQ-MS-4 [DEV CONFIRM] (not part of this case's expected result):
    # 25,000 rows exceed the warning threshold, but the SVM training set is only 16,000 rows.
    lines = td_path("TD-12_large_40000.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    boundary = tmp_path / "TD-15_rows_25000.csv"
    boundary.write_text("".join(lines[:25001]), encoding="utf-8")
    page.click("#new-run")
    app.upload(str(boundary))
    app.to_configure()
    app.select_target("Exited")
    warning = text(hint) if hint.count() else ""
    evidence.shot(page, "td15-warning", full_page=False)
    app.train(["svm"], timeout=900_000)
    evidence.shot(page, "td15-results")
    d15 = app.model_details("SVM")
    observation = (
        f"TD-15 (25,000 rows): ML hint {warning!r}; SVM details {d15}. "
        + ("The warning promises subsampling but SVM trained on all rows (no 'Subsampled' detail)."
           if warning and "Subsampled" not in d15 else "Warning and subsampling agree.")
    )
    evidence.note("Observation for SRS REQ-MS-4 [DEV CONFIRM]: " + observation)
    evidence.folder.mkdir(parents=True, exist_ok=True)
    (evidence.folder / "observation-REQ-MS-4.txt").write_text(observation, encoding="utf-8")


@pytest.mark.st("ST-35")
def test_st35_compute_device_reported(page, app_server, evidence):
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except FileNotFoundError:
        smi = "nvidia-smi not found"
    evidence.note(f"nvidia-smi: {smi}")
    models = ("Logistic Regression", "XGBoost", "Deep Neural Network")
    devices = {}
    try:
        for setting in ("auto", "cpu"):
            app_server.restart(compute_device=setting)
            app = banking(page)
            app.train(["logistic_regression", "xgboost", "dnn"])
            page.locator("#confusion-grid").scroll_into_view_if_needed()
            evidence.shot(page, f"devices-{setting}", full_page=False)
            devices[setting] = {m: app.model_details(m).get("Device") for m in models}
    finally:
        if app_server.compute_device != "cpu":
            app_server.restart(compute_device="cpu")
    evidence.note(f"Devices reported: {devices}")
    assert "RTX" in smi, f"precondition: NVIDIA GPU not reported by nvidia-smi ({smi})"
    assert devices["auto"] == {"Logistic Regression": "CPU", "XGBoost": "GPU (CUDA)", "Deep Neural Network": "CPU"}, devices["auto"]
    assert set(devices["cpu"].values()) == {"CPU"}, devices["cpu"]
