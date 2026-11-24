"""Group G: results dashboard, metrics and visualisations (ST-40 to ST-43).

ST-40, ST-41 and ST-43 read one completed 8-model run on the Banking sample (module
fixture), as their precondition in System_Test_Cases.xlsx states.
"""

import re

import pytest
from playwright.sync_api import expect

from pages import MODEL_KEYS, App, number, text

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
TIMES = ["Train time", "Test time"]


@pytest.fixture(scope="module")
def completed(browser, browser_context_args):
    context = browser.new_context(**browser_context_args)
    page = context.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    app.train(MODEL_KEYS)
    app.console_errors = errors
    yield app
    context.close()


def best_by(rows, key, tie):
    return max(rows, key=lambda r: (number(r[key]), number(r[tie])))


@pytest.mark.st("ST-40")
def test_st40_winner_banner_and_paradigm_cards(completed, evidence):
    app, page = completed, completed.page
    page.evaluate("window.scrollTo(0, 0)")
    evidence.shot(page, "winner-and-cards", full_page=False)
    banner = page.locator("#winner-banner")
    assert text(banner.locator(".stamp")) == "Best"
    assert text(banner.locator(".winner-kicker")) == "Best model · ranked by PR-AUC, then F1"
    stats = text(banner.locator(".winner-stats"))
    for word in ("PR-AUC", "F1", "precision", "recall", "accuracy", "ROC-AUC", "to train"):
        assert word in stats, f"banner stats miss {word}: {stats}"
    _, rows = app.table()
    winner = best_by(rows, "PR-AUC", "F1")
    name = text(banner.locator(".winner-name"))
    assert name.startswith(winner["Model"]), f"banner {name!r} vs table best {winner['Model']!r}"
    largest = page.evaluate("""() => {
          const sizes = [...document.querySelectorAll('[data-screen="results"] *')]
            .filter(e => e.offsetParent !== null && !e.closest('.visually-hidden'))
            .filter(e => { const r = e.getBoundingClientRect(); return r.width > 1 && r.height > 1; })
            .filter(e => [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()))
            .map(e => parseFloat(getComputedStyle(e).fontSize));
          return {max: Math.max(...sizes), winner: parseFloat(getComputedStyle(document.querySelector('.winner-name')).fontSize)};
        }""")
    assert largest["winner"] >= largest["max"], f"winner name is not the largest text: {largest}"
    for paradigm in ("ML", "DL", "QML"):
        card = page.locator(f"#paradigm-cards .card-{paradigm.lower()}")
        best = best_by([r for r in rows if r["Paradigm"] == paradigm], "PR-AUC", "F1")
        assert text(card.locator(".card-value")) == best["Model"]
        note = text(card.locator(".card-note"))
        assert "PR-AUC" in note and "recall" in note and "to train" in note, note
    evidence.note(f"Winner {winner['Model']} (PR-AUC {winner['PR-AUC']}, F1 {winner['F1']}).")


@pytest.mark.st("ST-41")
def test_st41_metrics_table_sorting_and_highlight(completed, evidence):
    app, page = completed, completed.page
    heads, rows = app.table()
    assert heads == ["Model", "Paradigm", *METRICS[:4], "ROC-AUC", "PR-AUC", "Threshold", *TIMES]
    assert page.get_attribute("#metrics-table thead th:nth-child(8)", "aria-sort") == "descending"
    pr = [number(r["PR-AUC"]) for r in rows]
    assert pr == sorted(pr, reverse=True), "initial order is not PR-AUC descending"
    assert {r["Paradigm"] for r in rows} == {"ML", "DL", "QML"}
    for column in METRICS + TIMES:
        cells = page.locator("#metrics-table tbody td.best")
        index = heads.index(column)
        values = [number(r[column]) for r in rows]
        target = min(values) if column in TIMES else max(values)
        highlighted = [
            number(text(c))
            for c in page.locator(f"#metrics-table tbody tr td:nth-child({index + 1}).best").all()
        ]
        assert highlighted and all(
            v == target for v in highlighted
        ), f"{column}: highlight {highlighted} vs best {target}"
        assert cells.count() > 0
    for column in heads:
        index = heads.index(column) + 1
        button = page.locator(f"#metrics-table thead th:nth-child({index}) button")
        for click in (1, 2):
            button.click()
            th = page.locator(f"#metrics-table thead th:nth-child({index})")
            aria = th.get_attribute("aria-sort")
            _, rows_now = app.table()
            values = [
                r[column] if column in ("Model", "Paradigm") else number(r[column])
                for r in rows_now
            ]
            expected = sorted(values, reverse=(aria == "descending"))
            assert aria in ("ascending", "descending"), f"{column}: aria-sort {aria}"
            assert values == expected, f"{column} click {click}: not sorted {aria}"
            if click == 1:
                first = aria
            else:
                assert aria != first, f"{column}: second click did not reverse"
    evidence.shot(page, "table-after-sorting", full_page=False)


@pytest.mark.partial
@pytest.mark.st("ST-42")
def test_st42_charts_render_and_degrade_without_chartjs(
    completed, browser, browser_context_args, evidence
):
    app, page = completed, completed.page
    page.locator(".chart-grid").scroll_into_view_if_needed()
    evidence.shot(page, "charts")
    assert page.evaluate("typeof window.Chart") == "function", "Chart.js did not load"
    for canvas in ("chart-metrics", "chart-roc", "chart-pr", "chart-time", "chart-tradeoff"):
        box = page.locator(f"#{canvas}").bounding_box()
        assert box and box["width"] > 50 and box["height"] > 50, f"{canvas} not drawn"
        assert page.evaluate(
            f"!!Chart.getChart(document.getElementById('{canvas}'))"
        ), f"{canvas} has no chart"
    assert text(page.locator("#tradeoff-metric")) == "PR-AUC"
    assert (
        page.evaluate("Chart.getChart(document.getElementById('chart-time')).options.scales.x.type")
        == "logarithmic"
    )
    roc_sets = page.evaluate(
        "Chart.getChart(document.getElementById('chart-roc')).data.datasets.length"
    )
    assert roc_sets >= 9, "ROC chart should hold 8 models plus the random baseline"
    assert app.console_errors == [], app.console_errors
    charts = page.evaluate("""() => {
          const c = id => Chart.getChart(document.getElementById(id));
          const ds = id => c(id).data.datasets.map(d => ({label: d.label, border: d.borderColor,
                                                         background: d.backgroundColor, dash: d.borderDash || []}));
          return {metrics: ds('chart-metrics'), metricLabels: c('chart-metrics').data.labels,
                  roc: ds('chart-roc'), pr: ds('chart-pr'),
                  timeAxis: c('chart-time').options.indexAxis, tradeoffType: c('chart-tradeoff').config.type,
                  legends: ['chart-metrics','chart-roc','chart-pr'].map(id => c(id).options.plugins.legend.display !== false)};
        }""")
    colour = {"ML": "rgba(76, 114, 176", "DL": "rgba(201, 79, 109", "QML": "rgba(129, 114, 179"}
    paradigm_of = {r["Model"]: r["Paradigm"] for r in app.table()[1]}
    assert [d["label"] for d in charts["metrics"]] == ["Accuracy", "Precision", "Recall", "F1"]
    for dataset in charts["metrics"]:
        for label, fill in zip(charts["metricLabels"], dataset["background"]):
            model = next(m for m in paradigm_of if str(label).startswith(m))
            assert fill.startswith(colour[paradigm_of[model]]), f"bar colour of {label}: {fill}"
    for name in ("roc", "pr"):
        models = [d for d in charts[name] if not d["label"].startswith("Random baseline")]
        baseline = [d for d in charts[name] if d["label"].startswith("Random baseline")]
        assert (
            len(models) == 8 and len(baseline) == 1
        ), f"{name}: {len(models)} models, {len(baseline)} baselines"
        assert baseline[0]["dash"], f"{name} baseline is not dashed"
        for d in models:
            p = next(p for p in ("QML", "DL", "ML") if f"({p})" in d["label"])
            assert d["border"].startswith(colour[p]), f"{name} {d['label']}: {d['border']}"
    assert charts["timeAxis"] == "y", "training-time bars are not horizontal"
    assert charts["tradeoffType"] == "scatter"
    assert all(charts["legends"]), "a chart legend is hidden"
    evidence.note(
        "Chart checks: bar and curve colours match the paradigm colours (ML #4c72b0, DL #c94f6d, QML #8172b3); every curve label names its paradigm; ROC and PR have one dashed random baseline; time chart is horizontal on a log axis; legends shown."
    )

    context = browser.new_context(**browser_context_args)
    try:
        blocked = context.new_page()
        blocked.route(re.compile(r".*cdn\.jsdelivr\.net.*"), lambda route: route.abort())
        other = App(blocked).open()
        other.load_sample("Banking")
        other.to_configure()
        other.train(["logistic_regression"])
        notice = blocked.locator("#charts-unavailable")
        expect(notice).to_be_visible()
        evidence.shot(blocked, "chartjs-blocked")
        assert text(notice).startswith("Charts could not be drawn because Chart.js did not load.")
        expect(blocked.locator(".chart-grid")).to_be_hidden()
        assert len(other.table()[1]) == 1
    finally:
        context.close()


@pytest.mark.st("ST-43")
def test_st43_confusion_matrices_and_metadata(completed, evidence):
    app, page = completed, completed.page
    page.locator("#confusion-grid").scroll_into_view_if_needed()
    evidence.shot(page, "confusion-and-metadata")
    meta = app.metadata()
    test_rows = int(
        re.search(r"/ (\d+) \(test split", meta["Train / validation / test rows"]).group(1)
    )
    figures = page.locator("#confusion-grid figure").all()
    assert len(figures) == 8
    pattern = re.compile(
        r"Caught \d+ of \d+ 1 rows · TP \d+ · FP \d+ \(false alarms\) · FN \d+ \(missed\) · TN \d+ · threshold \d\.\d{3}"
    )
    for figure in figures:
        caption = text(figure.locator("figcaption"))
        assert re.match(r"(ML|DL|QML) \S", caption), caption
        labels = [text(th) for th in figure.locator("thead th").all()][1:]
        assert labels == ["0", "1"], labels
        total = sum(int(text(c)) for c in figure.locator("tbody td").all())
        assert total == test_rows, f"{caption}: matrix total {total} vs test rows {test_rows}"
        assert pattern.fullmatch(text(figure.locator(".cm-counts"))), text(
            figure.locator(".cm-counts")
        )
    assert text(page.locator("#confusion-note")).startswith("Positive class: 1.")
    expected = {
        "Dataset": "banking.csv",
        "Rows": "200",
        "Target": "Exited (binary)",
        "Positive class": "1",
        "Class weights": "balanced",
        "Best model chosen by": "PR-AUC, then F1",
        "Seed": "42",
        "DNN epochs (max) / QML epochs / qubits": "100 / 20 / 4",
        "Dropped columns": "CustomerId",
        "Decision threshold": "tuned per model on the validation rows (best F1)",
    }
    for key, value in expected.items():
        assert meta.get(key) == value, f"{key}: {meta.get(key)!r}"
    for key in (
        "Split fingerprint",
        "QML subsampled to",
        "Positive rate in test rows",
        "Features after encoding",
    ):
        assert meta.get(key), f"{key} missing"
    assert meta["PCA variance retained"].endswith("%")
    evidence.note(f"Run metadata: {meta}")
