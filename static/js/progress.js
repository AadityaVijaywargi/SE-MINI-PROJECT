// Train screen: polls the job every 2 seconds, shows real server progress and lets the user
// cancel (COM-2, REQ-JB-3, REQ-JB-5, REQ-JB-6).

import { cancelJob, getResults, getStatus, toast } from "./api.js";
import { clear, el } from "./dom.js";
import { setState, showScreen, state } from "./state.js";
import { renderResults } from "./results.js";

const POLL_MS = 2000;
const STATUS_ICONS = {
  pending: "○",
  running: "◐",
  testing: "◑",
  done: "✓",
  failed: "✕",
  cancelled: "–",
};

let pollTimer = null;
let clockTimer = null;

function stopTimers() {
  clearTimeout(pollTimer);
  clearInterval(clockTimer);
}

function renderProgress(percent) {
  document.getElementById("progress-bar").setAttribute("aria-valuenow", String(percent));
  document.getElementById("progress-fill").style.width = `${percent}%`;
  document.getElementById("progress-text").textContent = `${percent}%`;
}

function renderChecklist(models) {
  const list = clear(document.getElementById("model-checklist"));
  models.forEach((model) => {
    list.append(
      el("li", { class: `check-${model.status}`, title: model.error || null }, [
        el("span", { class: "check-icon", "aria-hidden": "true", text: STATUS_ICONS[model.status] }),
        el("span", { text: model.model }),
        el("span", { class: "visually-hidden", text: `: ${model.status}` }),
        model.error ? el("span", { class: "check-error", text: model.error }) : null,
      ]),
    );
  });
}

function renderLog(lines) {
  const log = document.getElementById("job-log");
  if (log.children.length === lines.length) return;
  clear(log).append(...lines.map((line) => el("li", { text: line })));
  log.scrollTop = log.scrollHeight;
}

function showFailedActions() {
  document.getElementById("cancel-job").classList.add("hidden");
  document.getElementById("train-failed-actions").classList.remove("hidden");
}

async function finish(jobId) {
  const loading = document.getElementById("results-loading");
  loading.classList.remove("hidden");
  try {
    const results = await getResults(jobId);
    setState({ results });
    renderResults();
    showScreen("results");
  } catch {
    showFailedActions();
  } finally {
    loading.classList.add("hidden");
  }
}

async function poll(jobId) {
  if (state.jobId !== jobId) return;
  let status;
  try {
    status = await getStatus(jobId);
  } catch {
    stopTimers();
    showFailedActions();
    return;
  }
  document.getElementById("stage-label").textContent = status.stage;
  renderProgress(status.progress);
  renderChecklist(status.models);
  renderLog(status.log);

  const anyDone = status.models.some((model) => model.status === "done");
  if (status.status === "complete") {
    stopTimers();
    finish(jobId);
  } else if (status.status === "cancelled") {
    stopTimers();
    document.getElementById("cancel-job").classList.add("hidden");
    if (anyDone) {
      toast("Training cancelled. Showing the models that finished.", "info");
      finish(jobId);
    } else {
      document.getElementById("stage-label").textContent = "Cancelled before any model finished.";
      showFailedActions();
    }
  } else if (status.status === "failed") {
    stopTimers();
    document.getElementById("stage-label").textContent = `Failed: ${status.error}`;
    showFailedActions();
  } else {
    pollTimer = setTimeout(() => poll(jobId), POLL_MS);
  }
}

async function onCancel() {
  const button = document.getElementById("cancel-job");
  button.disabled = true;
  button.textContent = "Cancelling…";
  try {
    await cancelJob(state.jobId);
    document.getElementById("stage-label").textContent = "Cancelling: stopping after the current step…";
  } catch {
    button.disabled = false;
    button.textContent = "Cancel training";
  }
}

export function startPolling(jobId, models) {
  stopTimers();
  document.getElementById("train-failed-actions").classList.add("hidden");
  document.getElementById("results-loading").classList.add("hidden");
  const cancel = document.getElementById("cancel-job");
  cancel.classList.remove("hidden");
  cancel.disabled = false;
  cancel.textContent = "Cancel training";
  document.getElementById("stage-label").textContent = "Starting…";
  clear(document.getElementById("job-log"));
  renderProgress(0);
  renderChecklist(models);
  showScreen("train");

  const started = Date.now();
  const elapsed = document.getElementById("elapsed");
  elapsed.textContent = "0s";
  clockTimer = setInterval(() => {
    elapsed.textContent = `${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);
  poll(jobId);
}

export function initProgress() {
  document.getElementById("cancel-job").addEventListener("click", onCancel);
  document
    .getElementById("back-to-configure")
    .addEventListener("click", () => showScreen("configure"));
}
