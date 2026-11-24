"""Group C: target selection and configuration (ST-13 to ST-19)."""

import pytest
from playwright.sync_api import expect

from pages import MODEL_KEYS, App, sample, td, text

QML_NOTE = (
    "Quantum training uses PCA to 4 features and samples at most 2,000 rows; expect 1 to 3 minutes."
)


def options(page):
    return [
        (o.get_attribute("value"), text(o)) for o in page.locator("#target-select option").all()
    ]


@pytest.mark.st("ST-13")
def test_st13_suggested_target_preselected_and_starred(page, evidence):
    app = App(page).open()
    for label, target in (
        ("Banking", "Exited"),
        ("Healthcare", "Diagnosis"),
        ("Telecom", "Churn"),
        ("Insurance", "Fraud"),
    ):
        app.load_sample(label)
        uniques = {name: int(row["unique"]) for name, row in app.column_rows().items()}
        app.to_configure()
        evidence.shot(page, f"configure-{label.lower()}", full_page=False)
        assert page.locator("#target-select").input_value() == target
        opts = options(page)
        assert [value for value, _ in opts] == list(uniques), "dropdown does not list every column"
        assert dict(opts)[target] == f"★ {target}"
        for value, label_text in opts:
            candidate = 2 <= uniques[value] <= 20
            assert (
                label_text.startswith("★ ") == candidate
            ), f"{value}: star does not match 2 to 20 values rule"
        page.click("#back-to-upload")
    # Any column can be chosen, and the analysis follows the choice.
    app.load_sample("Banking")
    app.to_configure()
    app.select_target("Age")
    age_badge = app.task_badge()
    app.select_target("Exited")
    assert app.task_badge() == "Binary"
    evidence.note(
        f"Choosing Age gave task badge '{age_badge}'; choosing Exited again gave 'Binary'."
    )


@pytest.mark.st("ST-14")
def test_st14_binary_task_analysis(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    evidence.shot(page, "analysis", full_page=False)
    analysis = text(page.locator("#task-analysis"))
    assert app.task_badge() == "Binary"
    assert "2 classes" in analysis
    assert "imbalance 1:1" in analysis
    counts = [
        text(li.locator(".dist-count"))
        for li in page.locator("#task-analysis .distribution li").all()
    ]
    assert counts == ["100", "100"]
    expect(page.locator("#task-analysis .warning")).to_have_count(0)
    assert not app.start_disabled()


@pytest.mark.st("ST-15")
def test_st15_imbalance_warning_only_above_3_to_1(page, evidence):
    app = App(page).open()
    app.upload(td("TD-09_imbalanced_4to1.csv"))
    app.to_configure()
    app.select_target("Exited")
    evidence.shot(page, "4-to-1", full_page=False)
    assert "imbalance 4:1" in app.analysis_text()
    warning = text(page.locator("#task-analysis .warning"))
    assert warning.startswith("⚠ The classes are imbalanced (4.0:1). Accuracy is misleading here:")
    assert "Compare PR-AUC, recall and F1" in warning
    assert not app.start_disabled()

    page.click("#back-to-upload")
    app.upload(td("TD-10_imbalanced_3to1.csv"))
    app.to_configure()
    app.select_target("Exited")
    evidence.shot(page, "3-to-1", full_page=False)
    assert "imbalance 3:1" in app.analysis_text()
    expect(page.locator("#task-analysis .warning")).to_have_count(0)
    assert not app.start_disabled()


@pytest.mark.st("ST-16")
def test_st16_training_blocked_for_unsupported_targets(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    app.select_target("Balance")
    evidence.shot(page, "balance-unsupported", full_page=False)
    assert app.task_badge() == "Unsupported"
    assert app.start_disabled()
    assert app.blocked_reason() == (
        "Column 'Balance' has 168 distinct values. This looks like a regression target; only "
        "classification with 2 to 20 classes is supported."
    )

    page.click("#back-to-upload")
    app.upload(td("TD-11_single_row_class.csv"))
    app.to_configure()
    app.select_target("Exited")
    evidence.shot(page, "single-row-class", full_page=False)
    assert app.start_disabled()
    assert app.blocked_reason().startswith(
        "Class 'Unknown' has a single row, so a stratified train/test split is impossible."
    )


@pytest.mark.st("ST-17")
def test_st17_multiclass_target_disables_qml(page, evidence):
    app = App(page).open()
    app.upload(sample("three_class.csv"))
    app.to_configure()
    assert page.locator("#target-select").input_value() == "Segment"
    evidence.shot(page, "multiclass", full_page=False)
    assert app.task_badge() == "Multi-class"
    assert "3 classes" in app.analysis_text()
    vqc = page.locator("#model-vqc")
    assert not vqc.is_checked() and vqc.is_disabled()
    assert text(page.locator("#label-vqc")) == "Variational Quantum Classifier *"
    assert text(page.locator("#qml-note")).startswith(
        "* Binary targets only. This target has more than two classes."
    )
    for key in MODEL_KEYS[:-1]:
        assert page.locator(f"#model-{key}").is_checked(), f"{key} should stay ticked"
    assert not app.start_disabled()


@pytest.mark.st("ST-18")
def test_st18_back_to_binary_reenables_qml(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    app.select_target("Geography")
    evidence.shot(page, "geography", full_page=False)
    vqc = page.locator("#model-vqc")
    assert not vqc.is_checked() and vqc.is_disabled()
    assert text(page.locator("#label-vqc")).endswith("*")
    app.select_target("Exited")
    evidence.shot(page, "exited", full_page=False)
    assert vqc.is_checked() and vqc.is_enabled()
    assert text(page.locator("#label-vqc")) == "Variational Quantum Classifier"
    assert "Binary targets only" not in text(page.locator("#qml-note"))


@pytest.mark.st("ST-19")
def test_st19_model_grid_defaults_qml_note_and_no_model_block(page, evidence):
    app = App(page).open()
    app.load_sample("Banking")
    app.to_configure()
    groups = {}
    for fieldset in page.locator("#model-grid fieldset").all():
        legend = text(fieldset.locator("legend"))
        groups[legend] = [text(label) for label in fieldset.locator("label").all()]
    evidence.shot(page, "model-grid", full_page=False)
    assert groups == {
        "ML Machine Learning": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Naive Bayes",
            "XGBoost",
        ],
        "DL Deep Learning": ["Deep Neural Network"],
        "QML Quantum ML": ["Variational Quantum Classifier"],
    }
    assert all(page.locator(f"#model-{k}").is_checked() for k in MODEL_KEYS)
    assert text(page.locator("#qml-note")) == QML_NOTE
    app.set_option("opt-qubits", 6)
    assert "PCA to 6 features" in text(page.locator("#qml-note"))
    app.set_option("opt-qubits", 4)
    app.set_models([])
    evidence.shot(page, "no-model", full_page=False)
    assert app.start_disabled()
    assert app.blocked_reason() == "Select at least one model."
