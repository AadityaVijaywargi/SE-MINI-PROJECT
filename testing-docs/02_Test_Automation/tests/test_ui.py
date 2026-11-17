"""Group J: UI/UX and browser compatibility (ST-48, ST-49)."""

import pytest
from playwright.sync_api import expect

from pages import App, text

FOOTER = "◆ Results are for comparative study only. Do not use them for clinical, credit or underwriting decisions."
FOCUS_STYLE = r"""() => {
  const e = document.activeElement, s = getComputedStyle(e);
  const rgb = c => (c.match(/[\d.]+/g) || []).map(Number);
  const lum = ([r, g, b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
                                return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  let bg = null;
  for (let n = e.parentElement; n; n = n.parentElement) {
    const v = rgb(getComputedStyle(n).backgroundColor);
    if (v.length >= 3 && (v.length < 4 || v[3] > 0.5)) { bg = v; break; }
  }
  bg = bg || [255, 255, 255];
  const oc = rgb(s.outlineColor);
  const l1 = lum(oc), l2 = lum(bg);
  return {id: e.id, tag: e.tagName, label: (e.textContent || '').trim().slice(0, 30),
          outline: s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0,
          width: parseFloat(s.outlineWidth) || 0, colour: s.outlineColor,
          contrast: Math.round(((Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)) * 100) / 100,
          shadow: s.boxShadow !== 'none'};
}"""


def visible_screens(page):
    return page.evaluate("[...document.querySelectorAll('[data-screen]')].filter(s => !s.classList.contains('hidden')).length")


class Keyboard:
    def __init__(self, page):
        self.page = page
        self.focused = []

    def tab_to(self, predicate, limit=150):
        for _ in range(limit):
            self.page.keyboard.press("Tab")
            state = self.page.evaluate(FOCUS_STYLE)
            self.focused.append(state)
            if self.page.evaluate(predicate):
                return state
        raise AssertionError(f"could not reach the element with Tab: {predicate}")


@pytest.mark.partial
@pytest.mark.st("ST-48")
def test_st48_keyboard_only_flow_and_accessibility(page, evidence):
    app = App(page).open()
    keys = Keyboard(page)
    keys.tab_to("document.activeElement.id === 'dropzone'")
    for key in ("Enter", " "):
        with page.expect_file_chooser(timeout=5000):
            page.keyboard.press(key)
    keys.tab_to("document.activeElement.closest('#sample-buttons') && document.activeElement.textContent.includes('Banking')")
    page.keyboard.press("Enter")
    expect(page.locator("#profile")).to_be_visible()
    assert page.evaluate("document.activeElement.id") == "to-configure"
    page.keyboard.press("Enter")
    expect(app.screen("configure")).to_be_visible()
    app.wait_analysis()
    assert visible_screens(page) == 1
    keys.tab_to("document.activeElement.id === 'target-select'")
    keys.tab_to("document.activeElement.id === 'model-logistic_regression'")
    for key in ("decision_tree", "random_forest", "svm", "naive_bayes", "xgboost", "dnn", "vqc"):
        keys.tab_to(f"document.activeElement.id === 'model-{key}'", limit=3)
        page.keyboard.press(" ")
    keys.tab_to("document.activeElement.tagName === 'SUMMARY'")
    page.keyboard.press("Enter")
    assert page.locator("#advanced").evaluate("d => d.open"), "Enter did not open Advanced settings"
    evidence.shot(page, "configure-keyboard", full_page=False)
    keys.tab_to("document.activeElement.id === 'start-training'")
    page.keyboard.press("Enter")
    expect(app.screen("train")).to_be_visible()
    assert page.get_attribute("#progress-bar", "role") == "progressbar"
    assert page.get_attribute("#progress-bar", "aria-valuenow") is not None
    app.wait_results()
    assert visible_screens(page) == 1
    keys.tab_to("document.activeElement.closest('#metrics-table thead') !== null")
    index = page.evaluate(
        "[...document.querySelectorAll('#metrics-table thead th')].indexOf(document.activeElement.closest('th'))"
    )
    th = page.locator("#metrics-table thead th").nth(index)
    before = th.get_attribute("aria-sort")
    page.keyboard.press("Enter")
    assert th.get_attribute("aria-sort") != before, "Enter did not sort the column"
    keys.tab_to("document.activeElement.id === 'export-csv'")
    with page.expect_download():
        page.keyboard.press("Enter")
    keys.tab_to("document.activeElement.classList.contains('toast-close')")
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "Dismiss message"
    evidence.shot(page, "toast-close-focused", full_page=False)
    count = page.locator("#toasts .toast").count()
    page.keyboard.press("Enter")
    expect(page.locator("#toasts .toast")).to_have_count(count - 1)

    # BODY/HTML get focus only when Tab wraps past the last control; they are not controls.
    controls = [f for f in keys.focused if f["tag"] not in ("BODY", "HTML")]
    keys.focused = controls
    no_indicator = sorted({f"{f['tag']}#{f['id']} {' '.join(f['label'].split())}" for f in controls if not (f["outline"] or f["shadow"])})
    weak = sorted({f"{f['tag']}#{f['id']} {' '.join(f['label'].split())} ({f['width']}px, {f['contrast']}:1)"
                   for f in controls if f["outline"] and (f["width"] < 2 or f["contrast"] < 3)})
    contrasts = [f["contrast"] for f in controls if f["outline"]]
    evidence.note(
        f"{len(controls)} focus stops checked. Without outline or box-shadow: {no_indicator or 'none'}. "
        f"Outline width/contrast below 2 px or 3:1 (WCAG 2.2 SC 1.4.11 / 2.4.13 thresholds): {weak or 'none'}. "
        f"Outline contrast range {min(contrasts):.2f}:1 to {max(contrasts):.2f}:1."
    )
    assert not no_indicator, f"focused controls without a visible focus style: {no_indicator}"
    assert not weak, f"focus outline too thin or low contrast: {weak}"


VIEWPORTS = [(1366, 768), (375, 812)]
# Area where the round "Best" stamp covers the text of the banner heading (kicker).
STAMP_OVERLAP = """() => {
  const stamp = document.querySelector('#winner-banner .stamp').getBoundingClientRect();
  const range = document.createRange();
  range.selectNodeContents(document.querySelector('#winner-banner .winner-kicker'));
  let area = 0;
  for (const r of range.getClientRects()) {
    const w = Math.max(0, Math.min(stamp.right, r.right) - Math.max(stamp.left, r.left));
    const h = Math.max(0, Math.min(stamp.bottom, r.bottom) - Math.max(stamp.top, r.top));
    area += w * h;
  }
  return Math.round(area);
}"""


def layout_checks(page, width):
    problems = []
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    if overflow > 1:
        problems.append(f"horizontal page scroll of {overflow}px")
    if visible_screens(page) != 1:
        problems.append("more than one screen visible")
    if width <= 768:
        lefts = page.evaluate(
            """() => [...document.querySelectorAll('.cards')].filter(c => c.offsetParent)
                 .map(c => new Set([...c.children].filter(x => x.offsetParent).map(x => Math.round(x.getBoundingClientRect().left))).size)"""
        )
        if any(n > 1 for n in lefts):
            problems.append(f"card grid not single column: {lefts}")
        wraps = page.evaluate(
            """() => [...document.querySelectorAll('.table-wrap')].filter(w => w.offsetParent)
                 .map(w => getComputedStyle(w).overflowX)"""
        )
        if any(v not in ("auto", "scroll") for v in wraps):
            problems.append(f"wide tables do not scroll in their container: {wraps}")
    return problems


@pytest.mark.st("ST-49")
def test_st49_responsive_layout_and_browser_compatibility(playwright, base_url, evidence):
    results = []
    failures = []
    overlaps = []
    for channel in ("chrome", "msedge", "chromium"):
        try:
            browser = playwright.chromium.launch(channel=channel) if channel != "chromium" else playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{channel}: browser could not start ({str(exc).splitlines()[0]})")
            continue
        try:
            for width, height in VIEWPORTS:
                context = browser.new_context(base_url=base_url, viewport={"width": width, "height": height})
                page = context.new_page()
                tag = f"{channel}-{width}"
                app = App(page).open()
                app.load_sample("Banking")
                problems = layout_checks(page, width)
                app.to_configure()
                problems += layout_checks(page, width)
                app.train(["logistic_regression", "dnn"])
                problems += layout_checks(page, width)
                evidence.shot(page, f"{tag}-results", full_page=False)
                overlap = page.evaluate(STAMP_OVERLAP)
                overlaps.append(f"{tag}: stamp overlaps banner heading text by {overlap} px²")
                if text(page.locator("footer")) != FOOTER:
                    problems.append("footer text differs")
                badges = [text(b) for b in page.locator("#metrics-table .badge").all()]
                if not badges or any(b not in ("ML", "DL") for b in badges):
                    problems.append(f"paradigm badges without text: {badges}")
                results.append(f"{browser.browser_type.name}/{channel} {browser.version} at {width}x{height}: {'ok' if not problems else problems}")
                failures += [f"{tag}: {p}" for p in problems]
                context.close()
        finally:
            browser.close()
    evidence.note("\n".join(results))
    evidence.note("Observation (not part of the expected result): " + "; ".join(overlaps))
    assert not failures, failures
