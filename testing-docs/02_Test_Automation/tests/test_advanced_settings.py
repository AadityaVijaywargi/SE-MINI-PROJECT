"""Group D: Advanced performance-tuning settings (ST-20 to ST-25)."""

import pytest
from playwright.sync_api import expect

from pages import App, text


@pytest.fixture
def configured(page):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    return app


def check_values(app, evidence, element_id, rejected, accepted, message):
    page = app.page
    results = []
    for value in rejected:
        app.set_option(element_id, value)
        error, disabled, reason = app.advanced_error(), app.start_disabled(), app.blocked_reason()
        results.append(f"{value!r}: error={error!r}, start disabled={disabled}")
        evidence.shot(page, f"{element_id}-{value or 'blank'}-rejected", full_page=False)
        assert error == message, f"{element_id}={value!r}: {error!r}"
        assert disabled and reason == message, f"{element_id}={value!r}: Start training not blocked"
    for value in accepted:
        app.set_option(element_id, value)
        error, disabled = app.advanced_error(), app.start_disabled()
        results.append(f"{value!r}: error={error!r}, start disabled={disabled}")
        assert error == "", f"{element_id}={value!r} unexpectedly rejected: {error}"
        assert not disabled, f"{element_id}={value!r}: Start training disabled"
    evidence.note("; ".join(results))


@pytest.mark.st("ST-20")
def test_st20_advanced_panel_defaults(configured, evidence):
    app, page = configured, configured.page
    assert not page.locator("#advanced").evaluate("d => d.open"), "panel should start collapsed"
    app.open_advanced()
    evidence.shot(page, "advanced-defaults", full_page=False)
    ratio = page.locator("#opt-test-ratio")
    assert [text(o) for o in ratio.locator("option").all()] == [
        "10% of rows",
        "20% of rows",
        "30% of rows",
    ]
    assert text(ratio.locator("option:checked")) == "20% of rows"
    assert page.input_value("#opt-seed") == "42"
    assert page.input_value("#opt-dnn-epochs") == "100"
    assert page.input_value("#opt-qml-epochs") == "20"
    assert page.input_value("#opt-qubits") == "4"
    assert (
        text(page.locator("#opt-threshold-mode option:checked"))
        == "Tune on validation rows (best F1)"
    )
    expect(page.locator("#threshold-field")).to_be_hidden()
    assert page.locator("#opt-balance").is_checked()


@pytest.mark.st("ST-21")
def test_st21_qubit_boundaries(configured, evidence):
    check_values(
        configured,
        evidence,
        "opt-qubits",
        ["1", "9", "4.5"],
        ["2", "8"],
        "Qubits must be a whole number from 2 to 8.",
    )


@pytest.mark.st("ST-22")
def test_st22_dnn_epoch_boundaries(configured, evidence):
    check_values(
        configured,
        evidence,
        "opt-dnn-epochs",
        ["0", "501"],
        ["1", "500"],
        "DNN epochs must be a whole number from 1 to 500.",
    )


@pytest.mark.st("ST-23")
def test_st23_qml_epoch_boundaries(configured, evidence):
    check_values(
        configured,
        evidence,
        "opt-qml-epochs",
        ["0", "101"],
        ["1", "100"],
        "QML epochs must be a whole number from 1 to 100.",
    )


@pytest.mark.st("ST-24")
def test_st24_fixed_threshold_boundaries(configured, evidence):
    app, page = configured, configured.page
    app.open_advanced()
    page.select_option("#opt-threshold-mode", "fixed")
    expect(page.locator("#threshold-field")).to_be_visible()
    assert page.input_value("#opt-threshold") == "0.5"
    check_values(
        app,
        evidence,
        "opt-threshold",
        ["0", "1", ""],
        ["0.01", "0.99"],
        "Fixed threshold must be a number from 0.01 to 0.99.",
    )
    app.set_option("opt-threshold", "1")
    assert app.advanced_error()
    page.select_option("#opt-threshold-mode", "auto")
    evidence.shot(page, "back-to-auto", full_page=False)
    expect(page.locator("#threshold-field")).to_be_hidden()
    assert app.advanced_error() == ""
    assert not app.start_disabled()


@pytest.mark.st("ST-25")
def test_st25_seed_validation_and_test_split_choices(configured, evidence):
    app, page = configured, configured.page
    check_values(
        app,
        evidence,
        "opt-seed",
        ["-1", "2.5", ""],
        ["0", "123"],
        "Random seed must be a whole number 0 or more.",
    )
    values = [o.get_attribute("value") for o in page.locator("#opt-test-ratio option").all()]
    assert values == ["0.1", "0.2", "0.3"]
