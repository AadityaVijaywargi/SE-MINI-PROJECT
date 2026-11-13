"""Group F: training progress and cancellation (ST-36 to ST-39)."""

import re
import statistics
import time

import pytest
from playwright.sync_api import expect

from pages import MODEL_KEYS, App, td, text

ORDER = {"pending": 0, "running": 1, "testing": 2, "done": 3, "failed": 3, "cancelled": 3}
ICONS = {"pending": "○", "running": "◐", "testing": "◑", "done": "✓"}


def banking(page):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    return app


@pytest.mark.st("ST-36")
def test_st36_progress_stage_checklist_log_and_timer(page, evidence):
    app = banking(page)
    polls = []
    page.on("request", lambda r: polls.append(time.time()) if "/api/status/" in r.url else None)
    app.record_training()
    app.start()
    page.wait_for_timeout(3000)
    evidence.shot(page, "training-in-progress", full_page=False)
    app.wait_results()
    history = app.training_history()
    assert history, "no Train screen updates were recorded"

    values = [h["value"] for h in history]
    assert all(b >= a for a, b in zip(values, values[1:])), f"progress went backwards: {values}"
    assert max(values) == 100, f"progress never reached 100: max {max(values)}"

    stages = [h["stage"] for h in history]
    assert any(re.fullmatch(r"Training .+ \(\d of 8\)(: .*)?", s) for s in stages), set(stages)
    assert any(re.search(r"(Deep Neural Network|Variational Quantum Classifier) \(\d of 8\): epoch", s) for s in stages)
    testing_seen = any(re.fullmatch(r"Testing .+ \(\d of 8\)", s) for s in stages)
    preprocessing_seen = "Preprocessing data" in stages

    per_model = {}
    for h in history:
        for i, item in enumerate(h["checklist"]):
            per_model.setdefault(i, []).append(item["status"])
            status = item["status"]
            if status in ICONS:
                assert item["text"].startswith(ICONS[status]), item
    for i, seq in per_model.items():
        ranks = [ORDER[s] for s in seq]
        assert ranks == sorted(ranks), f"model {i} status went backwards: {seq}"
        assert seq[-1] == "done", f"model {i} ended as {seq[-1]}"
    assert any("running" in seq for seq in per_model.values())

    assert history[-1]["log"] >= 8, "log should have a line per tested model"
    seconds = [int(h["elapsed"].rstrip("s")) for h in history]
    assert seconds[-1] > 0 and seconds == sorted(seconds)
    assert page.get_attribute("#progress-bar", "role") == "progressbar"

    gaps = [b - a for a, b in zip(polls, polls[1:])]
    median_gap = statistics.median(gaps) if gaps else 0
    evidence.note(f"{len(polls)} /api/status requests, median gap {median_gap:.2f} s; distinct stages seen: {len(set(stages))}.")
    assert 1.8 <= median_gap <= 3.0, f"status polling interval {median_gap:.2f} s is not about 2 s"
    if not testing_seen or not preprocessing_seen:
        # Confirmation run on TD-12 (40,000 rows): stages last longer than the 2 s refresh.
        page.click("#new-run")
        app.upload(td("TD-12_large_40000.csv"))
        app.to_configure()
        app.select_target("Exited")
        app.set_models(["random_forest", "svm", "dnn"])
        app.start()
        wanted = {"preprocessing": re.compile(r"Preprocessing (data|done)"),
                  "testing": re.compile(r"Testing .+ \(\d of 3\)")}
        large = {key: False for key in wanted}
        deadline = time.time() + 900
        while time.time() < deadline and not app.screen("results").is_visible():
            label = text(page.locator("#stage-label"))
            for key, pattern in wanted.items():
                if not large[key] and pattern.fullmatch(label):
                    large[key] = True
                    evidence.shot(page, f"td12-{key}-label-on-screen", full_page=False)
            time.sleep(0.2)
        app.wait_results(timeout=900_000)
        preprocessing_seen = preprocessing_seen or large["preprocessing"]
        testing_seen = testing_seen or large["testing"]
        evidence.note(f"TD-12 confirmation run (Random Forest, SVM, DNN): stage labels seen on screen {large}.")
    if not testing_seen or not preprocessing_seen:
        evidence.manual(
            "The UI refreshes every 2 s, so short stages can be skipped. Never rendered on screen: "
            + ", ".join(n for n, seen in (("'Preprocessing data'", preprocessing_seen), ("'Testing <model> (i of n)'", testing_seen)) if not seen)
            + ". Confirm by watching a run."
        )


@pytest.mark.st("ST-37")
def test_st37_cancel_during_dnn_keeps_finished_results(page, evidence):
    app = banking(page)
    app.record_training()
    job = app.start_and_get_job()
    app.wait_model_status(job, "dnn", "running")
    clicked = time.time()
    page.click("#cancel-job")
    evidence.shot(page, "cancel-clicked", full_page=False)
    app.wait_toast("Training cancelled. Showing the models that finished.", timeout=30_000)
    app.wait_results()
    stopped_after = time.time() - clicked
    evidence.shot(page, "partial-results")
    history = app.training_history()
    assert any(h["cancel"]["text"] == "Cancelling…" and h["cancel"]["disabled"] for h in history)
    assert any(h["stage"] == "Cancelling: stopping after the current step…" for h in history)
    _, rows = app.table()
    assert sorted(r["Paradigm"] for r in rows) == ["ML"] * 6, rows
    assert text(page.locator("#winner-banner .winner-kicker")).endswith("· run cancelled, partial results")
    for card in ("dl", "qml"):
        assert "Cancelled before it finished." in text(page.locator(f"#paradigm-cards .card-{card}"))
    evidence.note(f"Results opened {stopped_after:.1f} s after Cancel was clicked.")
    assert stopped_after < 15, f"cancel took {stopped_after:.1f} s"


@pytest.mark.st("ST-38")
def test_st38_cancel_before_any_model_finishes(page, evidence):
    app = banking(page)
    app.set_models(["vqc"])
    job = app.start_and_get_job()
    app.wait_model_status(job, "vqc", "running")
    page.click("#cancel-job")
    expect(page.locator("#stage-label")).to_have_text("Cancelled before any model finished.", timeout=30_000)
    evidence.shot(page, "cancelled-before-any-model", full_page=False)
    expect(page.locator("#cancel-job")).to_be_hidden()
    expect(page.locator("#back-to-configure")).to_be_visible()
    expect(app.screen("results")).to_be_hidden()
    page.click("#back-to-configure")
    expect(app.screen("configure")).to_be_visible()
    checked = [k for k in MODEL_KEYS if page.locator(f"#model-{k}").is_checked()]
    assert checked == ["vqc"], checked


@pytest.mark.st("ST-39")
def test_st39_one_training_job_per_browser(page, context, playwright, base_url, evidence):
    app_a = banking(page)
    app_a.start()
    tab_b = context.new_page()
    app_b = App(tab_b).open()
    app_b.load_sample("Healthcare")
    app_b.to_configure()
    tab_b.click("#start-training")
    toast = app_b.wait_toast("already running")
    evidence.shot(tab_b, "tab-b-refused", full_page=False)
    assert toast == "A training job is already running in this browser. Wait for it to finish or cancel it first."
    expect(app_b.screen("configure")).to_be_visible()
    expect(app_a.screen("train")).to_be_visible()

    edge = playwright.chromium.launch(channel="msedge")
    try:
        edge_page = edge.new_context(base_url=base_url).new_page()
        app_e = App(edge_page).open()
        app_e.load_sample("Healthcare")
        app_e.to_configure()
        app_e.set_models(["logistic_regression"])
        app_e.start()
        evidence.shot(edge_page, "edge-started", full_page=False)
        app_e.wait_results()
    finally:
        edge.close()

    app_a.wait_results()
    app_b.clear_toasts()
    app_b.set_models(["logistic_regression"])
    app_b.start()
    app_b.wait_results()
    evidence.shot(tab_b, "tab-b-after-a-finished", full_page=False)
