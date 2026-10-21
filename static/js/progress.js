// Train screen: polls the job every 2 seconds and shows real server progress
// (COM-2, REQ-JB-3, REQ-JB-5).

import { getResults, getStatus } from "./api.js";
import { clear, el } from "./dom.js";
import { setState, showScreen, state } from "./state.js";
import { renderResults } from "./results.js";

const POLL_MS = 2000;
const STATUS_ICONS = { pending: "○", running: "◐", done: "✓", failed: "✕" };

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

async function finish(jobId) {
  try {
    const results = await getResults(jobId);
    setState({ results });
    renderResults();
    showScreen("results");
  } catch {
    document.getElementById("train-failed-actions").classList.remove("hidden");
  }
}

async function poll(jobId) {
  if (state.jobId !== jobId) return;
  let status;
  try {
    status = await getStatus(jobId);
  } catch {
    stopTimers();
    document.getElementById("train-failed-actions").classList.remove("hidden");
    return;
  }
  document.getElementById("stage-label").textContent = status.stage;
  renderProgress(status.progress);
  renderChecklist(status.models);
  renderLog(status.log);

  if (status.status === "complete") {
    stopTimers();
    finish(jobId);
  } else if (status.status === "failed") {
    stopTimers();
    document.getElementById("stage-label").textContent = `Failed: ${status.error}`;
    document.getElementById("train-failed-actions").classList.remove("hidden");
  } else {
    pollTimer = setTimeout(() => poll(jobId), POLL_MS);
  }
}

export function startPolling(jobId, models) {
  stopTimers();
  document.getElementById("train-failed-actions").classList.add("hidden");
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
  document
    .getElementById("back-to-configure")
    .addEventListener("click", () => showScreen("configure"));
}
