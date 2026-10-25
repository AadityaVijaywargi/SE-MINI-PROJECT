// Configure screen: target dropdown, task analysis, the model grid and the Advanced panel
// (REQ-TG-3 to REQ-TG-6, REQ-MS-1 to REQ-MS-5).

import { configure, startTraining } from "./api.js";
import { clear, el, paradigmBadge, PARADIGM_NAMES } from "./dom.js";
import { setState, showScreen, state } from "./state.js";
import { startPolling } from "./progress.js";

// Models in the order they train.
export const MODEL_CATALOG = [
  { key: "logistic_regression", name: "Logistic Regression", paradigm: "ML" },
  { key: "decision_tree", name: "Decision Tree", paradigm: "ML" },
  { key: "random_forest", name: "Random Forest", paradigm: "ML" },
  { key: "svm", name: "SVM", paradigm: "ML" },
  { key: "naive_bayes", name: "Naive Bayes", paradigm: "ML" },
  { key: "xgboost", name: "XGBoost", paradigm: "ML" },
  { key: "dnn", name: "Deep Neural Network", paradigm: "DL" },
  { key: "vqc", name: "Variational Quantum Classifier", paradigm: "QML" },
];

// Server limits, mirrored here so mistakes show before a request is sent.
const SVM_MAX_ROWS = 20000;
const QML_MAX_ROWS = 2000;
const OPTIONS = [
  { id: "opt-seed", key: "seed", label: "Random seed", min: 0, max: Number.MAX_SAFE_INTEGER },
  { id: "opt-dnn-epochs", key: "dnn_epochs", label: "DNN epochs", min: 1, max: 500 },
  { id: "opt-qml-epochs", key: "qml_epochs", label: "QML epochs", min: 1, max: 100 },
  { id: "opt-qubits", key: "qubits", label: "Qubits", min: 2, max: 8 },
];
const THRESHOLD_MIN = 0.01;
const THRESHOLD_MAX = 0.99;

// QML is unticked automatically for multi-class targets; remember that so it can be ticked
// again when the user goes back to a binary target.
let qmlAutoUnchecked = false;

function renderTargetOptions() {
  const select = clear(document.getElementById("target-select"));
  const suggested = new Set(state.suggestions.map((item) => item.column));
  state.profile.column_profiles.forEach((column) => {
    const label = suggested.has(column.name) ? `★ ${column.name}` : column.name;
    select.append(el("option", { value: column.name, text: label }));
  });
  const initial = state.target || state.suggestions[0]?.column || select.options[0]?.value;
  select.value = initial;
  return initial;
}

function renderAnalysis(analysis) {
  const box = clear(document.getElementById("task-analysis"));
  const taskNames = { binary: "Binary", multiclass: "Multi-class", unsupported: "Unsupported" };
  box.append(
    el("div", { class: "task-row" }, [
      el("span", { class: `badge badge-task-${analysis.task}`, text: taskNames[analysis.task] }),
      el("span", { class: "muted", text: `${analysis.n_classes} classes` }),
      analysis.imbalance_ratio
        ? el("span", { class: "muted", text: `imbalance ${analysis.imbalance_ratio}:1` })
        : null,
    ]),
  );
  const total = Object.values(analysis.distribution).reduce((sum, count) => sum + count, 0);
  if (analysis.task !== "unsupported") {
    box.append(
      el(
        "ul",
        { class: "distribution", "aria-label": "Rows per class" },
        Object.entries(analysis.distribution).map(([label, count]) =>
          el("li", {}, [
            el("span", { class: "dist-label", text: label }),
            el("span", { class: "dist-bar" }, [
              el("span", { class: "dist-fill", style: `width:${(100 * count) / total}%` }),
            ]),
            el("span", { class: "dist-count", text: count }),
          ]),
        ),
      ),
    );
  }
  // REQ-TG-5: plain-language imbalance warning.
  if (analysis.warning) box.append(el("p", { class: "warning", text: `⚠ ${analysis.warning}` }));
}

function selectedModels() {
  return [...document.querySelectorAll("#model-grid input:checked")].map((input) => input.value);
}

// "auto" tunes each binary model's threshold on validation rows; "fixed" uses one value.
function readThreshold() {
  if (document.getElementById("opt-threshold-mode").value === "auto") return { threshold: "auto" };
  const raw = document.getElementById("opt-threshold").value.trim();
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value) || value < THRESHOLD_MIN || value > THRESHOLD_MAX) {
    return { error: `Fixed threshold must be a number from ${THRESHOLD_MIN} to ${THRESHOLD_MAX}.` };
  }
  return { threshold: value };
}

function readOptions() {
  const settings = {
    test_ratio: Number(document.getElementById("opt-test-ratio").value),
    balance_classes: document.getElementById("opt-balance").checked,
  };
  for (const option of OPTIONS) {
    const raw = document.getElementById(option.id).value.trim();
    const value = Number(raw);
    if (raw === "" || !Number.isInteger(value) || value < option.min || value > option.max) {
      const range = option.key === "seed" ? "0 or more" : `from ${option.min} to ${option.max}`;
      return { error: `${option.label} must be a whole number ${range}.` };
    }
    settings[option.key] = value;
  }
  const { threshold, error } = readThreshold();
  if (error) return { error };
  settings.threshold = threshold;
  return { settings };
}

function updateTrainButton() {
  const button = document.getElementById("start-training");
  const blocked = document.getElementById("train-blocked");
  const advancedError = document.getElementById("advanced-error");
  const { error } = readOptions();
  advancedError.textContent = error || "";
  advancedError.classList.toggle("hidden", !error);
  if (error) document.getElementById("advanced").open = true;

  let reason = null;
  if (!state.analysis) reason = "Choose a target column first.";
  else if (!state.analysis.trainable) reason = state.analysis.reason;
  else if (!selectedModels().length) reason = "Select at least one model.";
  else if (error) reason = error;
  button.disabled = Boolean(reason);
  blocked.textContent = reason || "";
  blocked.classList.toggle("hidden", !reason || !state.analysis);
}

function qmlNoteText() {
  const qubits = Number(document.getElementById("opt-qubits").value) || 4;
  return (
    `Quantum training uses PCA to ${qubits} features and samples at most ` +
    `${QML_MAX_ROWS.toLocaleString()} rows; expect 1 to 3 minutes.`
  );
}

// REQ-MS-2 and REQ-MS-3: the QML note is always shown; multi-class targets disable QML.
function updateQmlAvailability() {
  const input = document.getElementById("model-vqc");
  const note = document.getElementById("qml-note");
  if (!input || !note) return;
  const multiclass = state.analysis && state.analysis.task === "multiclass";
  if (multiclass && !input.disabled) {
    qmlAutoUnchecked = input.checked;
    input.checked = false;
  } else if (!multiclass && input.disabled && qmlAutoUnchecked) {
    input.checked = true;
    qmlAutoUnchecked = false;
  }
  input.disabled = Boolean(multiclass);
  document.getElementById("label-vqc").textContent = multiclass
    ? "Variational Quantum Classifier *"
    : "Variational Quantum Classifier";
  clear(note).append(
    ...(multiclass
      ? [el("span", { class: "warn-text", text: "* Binary targets only. This target has more than two classes. " })]
      : []),
    qmlNoteText(),
  );
}

export async function onTargetChange(target) {
  setState({ target, analysis: null });
  const box = clear(document.getElementById("task-analysis"));
  box.append(el("span", { class: "spinner", "aria-label": "Analysing target" }));
  updateTrainButton();
  try {
    const analysis = await configure(state.datasetId, target);
    if (state.target !== target) return; // a newer choice is already loading
    setState({ analysis });
    renderAnalysis(analysis);
  } catch {
    clear(box);
  }
  updateQmlAvailability();
  updateTrainButton();
}

function modelCheckbox(model) {
  const id = `model-${model.key}`;
  return el("div", { class: "check" }, [
    el("input", { type: "checkbox", id, value: model.key, checked: true, onchange: updateTrainButton }),
    el("label", { for: id, id: `label-${model.key}`, text: model.name }),
  ]);
}

function renderModelGrid() {
  const grid = clear(document.getElementById("model-grid"));
  const bigData = state.profile.rows > SVM_MAX_ROWS;
  ["ML", "DL", "QML"].forEach((paradigm) => {
    const models = MODEL_CATALOG.filter((model) => model.paradigm === paradigm);
    grid.append(
      el("fieldset", { class: `model-group group-${paradigm.toLowerCase()}` }, [
        el("legend", {}, [paradigmBadge(paradigm), ` ${PARADIGM_NAMES[paradigm]}`]),
        ...models.map(modelCheckbox),
        // REQ-MS-4: SVM is subsampled on large datasets.
        paradigm === "ML" && bigData
          ? el("p", {
              class: "hint warn-text",
              text: `⚠ ${state.profile.rows.toLocaleString()} rows: SVM will train on a stratified sample of ${SVM_MAX_ROWS.toLocaleString()} rows.`,
            })
          : null,
        paradigm === "DL"
          ? el("p", { class: "hint", text: "Dense network with early stopping on a validation split of the training rows." })
          : null,
        paradigm === "QML" ? el("p", { class: "hint", id: "qml-note", text: qmlNoteText() }) : null,
      ]),
    );
  });
}

export function prepareConfigure() {
  qmlAutoUnchecked = false;
  renderModelGrid();
  const target = renderTargetOptions();
  onTargetChange(target);
}

async function onStartTraining() {
  const button = document.getElementById("start-training");
  const models = selectedModels();
  const { settings, error } = readOptions();
  if (error) {
    updateTrainButton();
    return;
  }
  button.disabled = true;
  try {
    const { job_id: jobId } = await startTraining({
      dataset_id: state.datasetId,
      target: state.target,
      models,
      ...settings,
    });
    setState({ jobId, selectedModels: models, settings, results: null });
    const checklist = MODEL_CATALOG.filter((model) => models.includes(model.key)).map(
      (model) => ({ key: model.key, model: model.name, status: "pending" }),
    );
    startPolling(jobId, checklist);
  } catch {
    updateTrainButton();
  }
}

export function initConfigure() {
  document
    .getElementById("target-select")
    .addEventListener("change", (event) => onTargetChange(event.target.value));
  document.getElementById("start-training").addEventListener("click", onStartTraining);
  document.getElementById("back-to-upload").addEventListener("click", () => showScreen("upload"));
  OPTIONS.forEach((option) =>
    document.getElementById(option.id).addEventListener("input", () => {
      updateTrainButton();
      if (option.key === "qubits") updateQmlAvailability();
    }),
  );
  document.getElementById("opt-threshold-mode").addEventListener("change", (event) => {
    document.getElementById("threshold-field").classList.toggle("hidden", event.target.value !== "fixed");
    updateTrainButton();
  });
  document.getElementById("opt-threshold").addEventListener("input", updateTrainButton);
}
