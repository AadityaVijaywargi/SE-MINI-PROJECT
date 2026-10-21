// Results screen: winner banner, paradigm cards and the sortable metrics table
// (REQ-RS-1 to REQ-RS-3). Charts and CSV export arrive in Sprint 2.

import { clear, el, paradigmBadge, PARADIGM_NAMES } from "./dom.js";
import { showScreen, state } from "./state.js";

const COLUMNS = [
  { key: "model", label: "Model", type: "text" },
  { key: "paradigm", label: "Paradigm", type: "text" },
  { key: "accuracy", label: "Accuracy", type: "metric", best: "max" },
  { key: "precision", label: "Precision", type: "metric", best: "max" },
  { key: "recall", label: "Recall", type: "metric", best: "max" },
  { key: "f1", label: "F1", type: "metric", best: "max" },
  { key: "roc_auc", label: "ROC-AUC", type: "metric", best: "max" },
  { key: "train_time", label: "Train time", type: "time", best: "min" },
];

let sort = { key: "accuracy", descending: true };

const pct = (value) => (value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`);
const seconds = (value) => (value < 1 ? `${(value * 1000).toFixed(0)} ms` : `${value.toFixed(2)} s`);

function format(column, value) {
  if (column.type === "metric") return pct(value);
  if (column.type === "time") return seconds(value);
  return value;
}

function renderWinner(done, summary) {
  const banner = clear(document.getElementById("winner-banner"));
  const winner = done.find((result) => result.model === summary.best_overall);
  if (!winner) {
    banner.append(el("p", { text: "No model finished training." }));
    return;
  }
  banner.append(
    el("div", { class: "winner-kicker", text: "Best model" }),
    el("div", { class: "winner-name" }, [winner.model, " ", paradigmBadge(winner.paradigm)]),
    el("div", { class: "winner-stats" }, [
      el("span", {}, [el("strong", { text: pct(winner.accuracy) }), " accuracy"]),
      el("span", {}, [el("strong", { text: pct(winner.f1) }), " F1"]),
      winner.roc_auc !== null
        ? el("span", {}, [el("strong", { text: winner.roc_auc.toFixed(3) }), " ROC-AUC"])
        : null,
    ]),
  );
}

function renderParadigmCards(done, summary) {
  const cards = clear(document.getElementById("paradigm-cards"));
  ["ML", "DL", "QML"].forEach((paradigm) => {
    const bestName = summary[`best_${paradigm.toLowerCase()}`];
    const best = done.find((result) => result.model === bestName);
    cards.append(
      el("div", { class: `card paradigm-card card-${paradigm.toLowerCase()}` }, [
        el("div", { class: "card-label" }, [paradigmBadge(paradigm), ` ${PARADIGM_NAMES[paradigm]}`]),
        best
          ? el("div", {}, [
              el("div", { class: "card-value", text: best.model }),
              el("div", {
                class: "card-note",
                text: `${pct(best.accuracy)} accuracy · ${seconds(best.train_time)}`,
              }),
            ])
          : el("div", { class: "card-note", text: "Not run in this release (Sprint 2)." }),
      ]),
    );
  });
}

function bestValues(done) {
  const best = {};
  COLUMNS.filter((column) => column.best).forEach((column) => {
    const values = done.map((result) => result[column.key]).filter((value) => value !== null);
    if (values.length) best[column.key] = column.best === "max" ? Math.max(...values) : Math.min(...values);
  });
  return best;
}

function sortedRows(rows) {
  const direction = sort.descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const left = a[sort.key];
    const right = b[sort.key];
    if (left === right) return 0;
    if (left === null || left === undefined) return 1;
    if (right === null || right === undefined) return -1;
    return (left < right ? -1 : 1) * direction;
  });
}

function onSortClick(key) {
  sort = { key, descending: sort.key === key ? !sort.descending : key !== "train_time" };
  renderTable();
}

function renderTable() {
  const { results } = state.results;
  const done = results.filter((result) => result.status === "done");
  const failed = results.filter((result) => result.status === "failed");
  const best = bestValues(done);
  const table = clear(document.getElementById("metrics-table"));

  const header = el(
    "tr",
    {},
    COLUMNS.map((column) => {
      const active = sort.key === column.key;
      let ariaSort = "none";
      if (active) ariaSort = sort.descending ? "descending" : "ascending";
      return el("th", { scope: "col", "aria-sort": ariaSort }, [
        el("button", { type: "button", class: "sort", onclick: () => onSortClick(column.key) }, [
          column.label,
          el("span", { "aria-hidden": "true", text: active ? (sort.descending ? " ▼" : " ▲") : "" }),
        ]),
      ]);
    }),
  );

  const rows = sortedRows(done).map((result) =>
    el(
      "tr",
      {},
      COLUMNS.map((column) => {
        const value = result[column.key];
        if (column.key === "model") return el("th", { scope: "row", text: value });
        if (column.key === "paradigm") return el("td", {}, paradigmBadge(value));
        const isBest = value !== null && best[column.key] === value;
        return el("td", { class: `num ${isBest ? "best" : ""}`, text: format(column, value) });
      }),
    ),
  );
  const failedRows = failed.map((result) =>
    el("tr", { class: "failed-row" }, [
      el("th", { scope: "row", text: result.model }),
      el("td", {}, paradigmBadge(result.paradigm)),
      el("td", { colspan: COLUMNS.length - 2, text: `Failed: ${result.error}` }),
    ]),
  );
  table.append(el("thead", {}, header), el("tbody", {}, [...rows, ...failedRows]));
}

export function renderResults() {
  const { results, summary, metadata } = state.results;
  const done = results.filter((result) => result.status === "done");
  sort = { key: "accuracy", descending: true };
  renderWinner(done, summary);
  renderParadigmCards(done, summary);
  renderTable();
  document.getElementById("run-meta").textContent = metadata
    ? `${metadata.dataset} · target “${metadata.target}” (${metadata.task}) · ` +
      `${metadata.train_rows} train / ${metadata.test_rows} test rows · ` +
      `${metadata.features} features · seed ${metadata.seed} · split ${metadata.split_fingerprint}`
    : "";
}

export function initResults() {
  document.getElementById("retrain").addEventListener("click", () => showScreen("configure"));
  document.getElementById("new-run").addEventListener("click", () => showScreen("upload"));
}
