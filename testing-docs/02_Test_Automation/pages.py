"""Page helpers for the application under test. Only element IDs and visible text already
present in static/index.html and the rendered screens are used (black box)."""

import re
import time

from playwright.sync_api import Page, expect

from settings import SAMPLES, TEST_DATA

MODEL_KEYS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "naive_bayes",
    "xgboost",
    "dnn",
    "vqc",
]
TRAIN_TIMEOUT_MS = 240_000


def td(name):
    """Path of a TD-nn test data file."""
    path = TEST_DATA / name
    assert path.exists(), f"Missing test data {path}; run make_test_data.py"
    return str(path)


def sample(name):
    return str(SAMPLES / name)


class App:
    def __init__(self, page: Page):
        self.page = page

    # ------------------------------------------------------------------ navigation
    def open(self):
        self.page.goto("/")
        expect(self.page.locator("#sample-buttons button")).to_have_count(4)
        return self

    def screen(self, name):
        return self.page.locator(f'[data-screen="{name}"]')

    # ------------------------------------------------------------------ upload screen
    def choose_file(self, path):
        with self.page.expect_file_chooser() as chooser:
            self.page.click("#dropzone")
        chooser.value.set_files(path)

    def upload(self, path):
        self.choose_file(path)
        expect(self.page.locator("#profile")).to_be_visible()
        expect(self.page.locator("#upload-progress")).to_be_hidden()

    def load_sample(self, label):
        self.page.locator("#sample-buttons button", has_text=label).click()
        expect(self.page.locator("#profile")).to_be_visible()
        expect(self.page.locator("#upload-progress")).to_be_hidden()

    def summary_cards(self):
        cards = {}
        for card in self.page.locator("#summary-cards .card").all():
            label = card.locator(".card-label").text_content().strip()
            value = card.locator(".card-value").text_content().strip()
            note = card.locator(".card-note")
            cards[label] = (value, note.text_content().strip() if note.count() else "")
        return cards

    def column_rows(self):
        rows = {}
        for row in self.page.locator("#column-table tbody tr").all():
            name = row.locator("th").text_content().strip()
            cells = [c.text_content().strip() for c in row.locator("td").all()]
            warn = row.locator(".warn-icon")
            rows[name] = {
                "type": cells[0],
                "unique": cells[1],
                "missing": cells[2],
                "role": row.locator(".badge[class*=badge-role]").text_content().strip(),
                "warning": warn.get_attribute("title") if warn.count() else "",
            }
        return rows

    # ------------------------------------------------------------------ toasts
    def toasts(self):
        return [
            t.text_content().replace("×", "").strip()
            for t in self.page.locator("#toasts .toast").all()
        ]

    def wait_toast(self, text, timeout=15_000):
        locator = self.page.locator("#toasts .toast", has_text=text)
        expect(locator.first).to_be_visible(timeout=timeout)
        return locator.first.text_content().replace("×", "").strip()

    def clear_toasts(self):
        self.page.evaluate("document.querySelectorAll('#toasts .toast').forEach(t => t.remove())")

    # ------------------------------------------------------------------ configure screen
    def to_configure(self):
        self.page.click("#to-configure")
        expect(self.screen("configure")).to_be_visible()
        self.wait_analysis()

    def wait_analysis(self):
        expect(self.page.locator("#task-analysis .badge")).to_be_visible(timeout=20_000)

    def select_target(self, column):
        self.page.select_option("#target-select", column)
        expect(self.page.locator("#task-analysis .spinner")).to_have_count(0, timeout=20_000)
        self.wait_analysis()

    def task_badge(self):
        return self.page.locator("#task-analysis .badge").text_content().strip()

    def analysis_text(self):
        return self.page.locator("#task-analysis").text_content()

    def set_models(self, keys):
        for key in MODEL_KEYS:
            box = self.page.locator(f"#model-{key}")
            if box.is_disabled():
                continue
            box.set_checked(key in keys)

    def open_advanced(self):
        if not self.page.locator("#advanced").evaluate("d => d.open"):
            self.page.click("#advanced summary")
        expect(self.page.locator("#opt-seed")).to_be_visible()

    def set_option(self, element_id, value):
        self.open_advanced()
        self.page.fill(f"#{element_id}", str(value))

    def advanced_error(self):
        locator = self.page.locator("#advanced-error")
        return locator.text_content().strip() if locator.is_visible() else ""

    def start_disabled(self):
        return self.page.locator("#start-training").is_disabled()

    def blocked_reason(self):
        locator = self.page.locator("#train-blocked")
        return locator.text_content().strip() if locator.is_visible() else ""

    # ------------------------------------------------------------------ training
    def start(self):
        expect(self.page.locator("#start-training")).to_be_enabled()
        self.page.click("#start-training")
        expect(self.screen("train")).to_be_visible(timeout=15_000)

    def wait_results(self, timeout=TRAIN_TIMEOUT_MS):
        expect(self.screen("results")).to_be_visible(timeout=timeout)
        expect(self.page.locator("#winner-banner")).not_to_be_empty()
        # charts are drawn on the next animation frame
        self.page.wait_for_timeout(500)

    def train(self, models, timeout=TRAIN_TIMEOUT_MS):
        self.set_models(models)
        self.start()
        started = time.time()
        self.wait_results(timeout)
        return time.time() - started

    def elapsed_seconds(self):
        value = self.page.locator("#elapsed").text_content().strip()
        return int(value.rstrip("s") or 0)

    def record_training(self):
        """Record every change the Train screen renders (each 2 s poll) in window.__ui."""
        self.page.evaluate("""() => {
              window.__ui = [];
              const q = s => document.querySelector(s);
              const rec = () => window.__ui.push({
                t: Date.now(),
                stage: q('#stage-label').textContent,
                value: Number(q('#progress-bar').getAttribute('aria-valuenow')),
                checklist: [...document.querySelectorAll('#model-checklist li')].map(li => ({
                  status: li.className.replace('check-', ''), text: li.textContent })),
                cancel: {text: q('#cancel-job').textContent, disabled: q('#cancel-job').disabled,
                         hidden: q('#cancel-job').classList.contains('hidden')},
                log: document.querySelectorAll('#job-log li').length,
                elapsed: q('#elapsed').textContent,
              });
              new MutationObserver(rec).observe(q('[data-screen="train"]'),
                {subtree: true, childList: true, characterData: true, attributes: true});
            }""")

    def training_history(self):
        return self.page.evaluate("window.__ui || []")

    def start_and_get_job(self):
        """Click Start training and return the job id from the /api/train response."""
        with self.page.expect_response(lambda r: r.url.endswith("/api/train")) as info:
            self.start()
        response = info.value
        assert response.status == 202, f"/api/train answered {response.status}"
        return response.json()["job_id"]

    def server_stages(self, job_id, timeout_s=600):
        """Every stage label the server reports, polled every 0.1 s until the job ends.
        The UI polls every 2 s (COM-2), so it can skip stages shorter than that."""
        stages = []
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            body = self.page.request.get(f"/api/status/{job_id}").json()
            if not stages or stages[-1] != body["stage"]:
                stages.append(body["stage"])
            if body["status"] in ("complete", "failed", "cancelled"):
                return stages
            time.sleep(0.1)
        raise AssertionError("job did not finish")

    def wait_model_status(self, job_id, key, status, timeout_s=120):
        """Poll /api/status (black-box HTTP, same browser cookies) until a model has a status."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            body = self.page.request.get(f"/api/status/{job_id}").json()
            states = {m["key"]: m["status"] for m in body["models"]}
            if states.get(key) == status:
                return body
            if body["status"] in ("complete", "failed", "cancelled"):
                raise AssertionError(f"job ended ({body['status']}) before {key} was {status}")
            time.sleep(0.1)
        raise AssertionError(f"{key} never reached {status}")

    # ------------------------------------------------------------------ results
    def metadata(self):
        out = {}
        for item in self.page.locator("#run-meta .meta-item").all():
            out[item.locator("dt").text_content().strip()] = (
                item.locator("dd").text_content().strip()
            )
        return out

    def table(self):
        heads = [
            re.sub(r"[▲▼]", "", h.text_content()).strip()
            for h in self.page.locator("#metrics-table thead th").all()
        ]
        rows = []
        for row in self.page.locator("#metrics-table tbody tr:not(.failed-row)").all():
            cells = [row.locator("th").text_content().strip()] + [
                c.text_content().strip() for c in row.locator("td").all()
            ]
            rows.append(dict(zip(heads, cells)))
        return heads, rows

    def model_details(self, model_name):
        card = self.page.locator(
            "#confusion-grid figure", has=self.page.locator("figcaption", has_text=model_name)
        )
        text = (
            card.locator(".cm-details").text_content()
            if card.locator(".cm-details").count()
            else ""
        )
        details = {}
        for part in text.split(" · "):
            if ": " in part:
                key, value = part.split(": ", 1)
                details[key.strip()] = value.strip()
        return details


def text(locator):
    """DOM text of an element with whitespace collapsed (ignores CSS text-transform)."""
    return " ".join((locator.text_content() or "").split())


def number(text):
    """'86.5%' -> 0.865, '0.912' -> 0.912, '120 ms' -> 0.12, '1.20 s' -> 1.2, '—' -> None."""
    text = text.strip()
    if text in ("—", ""):
        return None
    if text.endswith("%"):
        return float(text[:-1]) / 100
    if text.endswith(" ms"):
        return float(text[:-3]) / 1000
    if text.endswith(" s"):
        return float(text[:-2])
    return float(text)
