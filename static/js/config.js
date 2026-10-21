// Configure screen: target dropdown, task analysis and the model grid
// (REQ-TG-3 to REQ-TG-6, REQ-MS-1).

import { configure, startTraining } from "./api.js";
import { clear, el, paradigmBadge, PARADIGM_NAMES } from "./dom.js";
import { setState, showScreen, state } from "./state.js";
import { startPolling } from "./progress.js";

// Models in the order they train. DL and QML arrive in Sprint 2 (SM-38, SM-40).
export const MODEL_CATALOG = [
  { key: "logistic_regression", name: "Logistic Regression", paradigm: "ML", available: true },
  { key: "decision_tree", name: "Decision Tree", paradigm: "ML", available: true },
  { key: "random_forest", name: "Random Forest", paradigm: "ML", available: true },
  { key: "svm", name: "SVM", paradigm: "ML", available: true },
  { key: "naive_bayes", name: "Naive Bayes", paradigm: "ML", available: true },
  { key: "xgboost", name: "XGBoost", paradigm: "ML", available: true },
  { key: "dnn", name: "Deep Neural Network", paradigm: "DL", available: false },
  { key: "vqc", name: "Variational Quantum Classifier", paradigm: "QML", available: false },
];

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
  if (analysis.warning) box.append(el("p", { class: "warning", text: `⚠ ${analysis.warning}` }));
}

function selectedModels() {
  return [...document.querySelectorAll("#model-grid input:checked")].map((input) => input.value);
}

function updateTrainButton() {
  const button = document.getElementById("start-training");
  const blocked = document.getElementById("train-blocked");
  let reason = null;
  if (!state.analysis) reason = "Choose a target column first.";
  else if (!state.analysis.trainable) reason = state.analysis.reason;
  else if (!selectedModels().length) reason = "Select at least one model.";
  button.disabled = Boolean(reason);
  blocked.textContent = reason || "";
  blocked.classList.toggle("hidden", !reason || !state.analysis);
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
  updateTrainButton();
}

function renderModelGrid() {
  const grid = clear(document.getElementById("model-grid"));
  ["ML", "DL", "QML"].forEach((paradigm) => {
    const models = MODEL_CATALOG.filter((model) => model.paradigm === paradigm);
    const anyAvailable = models.some((model) => model.available);
    grid.append(
      el("fieldset", { class: `model-group group-${paradigm.toLowerCase()}` }, [
        el("legend", {}, [paradigmBadge(paradigm), ` ${PARADIGM_NAMES[paradigm]}`]),
        ...models.map((model) => {
          const id = `model-${model.key}`;
          return el("div", { class: "check" }, [
            el("input", {
              type: "checkbox",
              id,
              value: model.key,
              checked: model.available,
              disabled: !model.available,
              onchange: updateTrainButton,
            }),
            el("label", { for: id, text: model.name }),
          ]);
        }),
        anyAvailable ? null : el("p", { class: "hint", text: "Coming in Sprint 2." }),
      ]),
    );
  });
}

export function prepareConfigure() {
  renderModelGrid();
  const target = renderTargetOptions();
  onTargetChange(target);
}

async function onStartTraining() {
  const button = document.getElementById("start-training");
  const models = selectedModels();
  button.disabled = true;
  try {
    const { job_id: jobId } = await startTraining({
      dataset_id: state.datasetId,
      target: state.target,
      models,
    });
    setState({ jobId, selectedModels: models, results: null });
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
}
