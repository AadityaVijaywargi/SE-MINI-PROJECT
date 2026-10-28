// Results screen: winner banner, paradigm cards, the sortable metrics table, charts,
// confusion matrices, run metadata and CSV export (REQ-RS-1 to REQ-RS-9).

import { resultsCsvUrl, toast } from "./api.js";
import { destroyCharts, paradigmColour, renderCharts } from "./charts.js";
import { clear, el, paradigmBadge, PARADIGM_NAMES } from "./dom.js";
import { showScreen, state } from "./state.js";

const COLUMNS = [
  { key: "model", label: "Model", type: "text" },
  { key: "paradigm", label: "Paradigm", type: "text" },
  { key: "accuracy", label: "Accuracy", type: "metric", best: "max" },
  { key: "precision", label: "Precision", type: "metric", best: "max" },
  { key: "recall", label: "Recall", type: "metric", best: "max" },
  { key: "f1", label: "F1", type: "metric", best: "max" },
  { key: "roc_auc", label: "ROC-AUC", type: "score", best: "max" },
  { key: "pr_auc", label: "PR-AUC", type: "score", best: "max" },
  { key: "threshold", label: "Threshold", type: "score" },
  { key: "train_time", label: "Train time", type: "time", best: "min" },
  { key: "test_time", label: "Test time", type: "time", best: "min" },
];
// Columns that only binary targets fill in; hidden for multi-class results.
const BINARY_ONLY = new Set(["roc_auc", "pr_auc", "threshold"]);
export const METRIC_NAMES = { pr_auc: "PR-AUC", f1: "F1", accuracy: "accuracy", roc_auc: "ROC-AUC" };

let sort = { key: "accuracy", descending: true };

const pct = (value) => (value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`);
const score = (value) => (value === null || value === undefined ? "—" : value.toFixed(3));

// The metric the server ranked models by (REQ-EV-4): PR-AUC for binary targets, F1 otherwise.
function rankKeys(summary) {
  return summary?.ranked_by?.length ? summary.ranked_by : ["accuracy", "f1"];
}

function rankText(summary) {
  return rankKeys(summary).map((key) => METRIC_NAMES[key] || key).join(", then ");
}

function metricText(key, value) {
  return key === "accuracy" || key === "f1" ? pct(value) : score(value);
}

function visibleColumns() {
  const binary = state.results.metadata?.task === "binary";
  return COLUMNS.filter((column) => binary || !BINARY_ONLY.has(column.key));
}
const seconds = (value) => (value < 1 ? `${(value * 1000).toFixed(0)} ms` : `${value.toFixed(2)} s`);

function format(column, value) {
  if (column.type === "metric") return pct(value);
  if (column.type === "score") return score(value);
  if (column.type === "time") return seconds(value);
  return value;
}

function doneResults() {
  return state.results.results.filter((result) => result.status === "done");
}

function renderWinner(done, summary, cancelled) {
  const banner = clear(document.getElementById("winner-banner"));
  const winner = done.find((result) => result.model === summary.best_overall);
  if (!winner) {
    banner.append(el("p", { text: "No model finished training." }));
    return;
  }
  const binary = winner.pr_auc !== null && winner.pr_auc !== undefined;
  const kicker = `Best model · ranked by ${rankText(summary)}`;
  banner.append(
    el("span", { class: "stamp", "aria-hidden": "true", text: "Best" }),
    el("div", {
      class: "winner-kicker",
      text: cancelled ? `${kicker} · run cancelled, partial results` : kicker,
    }),
    el("div", { class: "winner-name" }, [winner.model, " ", paradigmBadge(winner.paradigm)]),
    el("div", { class: "winner-stats" }, [
      binary ? el("span", {}, [el("strong", { text: score(winner.pr_auc) }), " PR-AUC"]) : null,
      el("span", {}, [el("strong", { text: pct(winner.f1) }), " F1"]),
      el("span", {}, [el("strong", { text: pct(winner.precision) }), " precision"]),
      el("span", {}, [el("strong", { text: pct(winner.recall) }), " recall"]),
      el("span", {}, [el("strong", { text: pct(winner.accuracy) }), " accuracy"]),
      winner.roc_auc !== null && winner.roc_auc !== undefined
        ? el("span", {}, [el("strong", { text: score(winner.roc_auc) }), " ROC-AUC"])
        : null,
      el("span", {}, [el("strong", { text: seconds(winner.train_time) }), " to train"]),
    ]),
  );
}

const PARADIGM_OF = { dnn: "DL", vqc: "QML", qsvm: "QML" };

function emptyCardNote(paradigm, results, metadata) {
  const entry = results.find((result) => result.paradigm === paradigm);
  if (entry && entry.status === "failed") return `Failed: ${entry.error}`;
  if (paradigm === "QML" && metadata && metadata.task !== "binary") {
    return "Binary targets only; not run for this target.";
  }
  const selected = state.selectedModels.some((key) => (PARADIGM_OF[key] || "ML") === paradigm);
  if (selected && state.results.status === "cancelled") return "Cancelled before it finished.";
  return "Not selected in this run.";
}

function renderParadigmCards(done, summary) {
  const { results, metadata } = state.results;
  const cards = clear(document.getElementById("paradigm-cards"));
  const [rankKey] = rankKeys(summary);
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
                text: `${metricText(rankKey, best[rankKey])} ${METRIC_NAMES[rankKey] || rankKey} · ${pct(best.recall)} recall · ${seconds(best.train_time)} to train`,
              }),
            ])
          : el("div", { class: "card-note", text: emptyCardNote(paradigm, results, metadata) }),
      ]),
    );
  });
}

function bestValues(done) {
  const best = {};
  COLUMNS.filter((column) => column.best).forEach((column) => {
    const values = done
      .map((result) => result[column.key])
      .filter((value) => value !== null && value !== undefined);
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
  const column = COLUMNS.find((entry) => entry.key === key);
  sort = { key, descending: sort.key === key ? !sort.descending : column.best !== "min" };
  renderTable();
}

function renderTable() {
  const { results } = state.results;
  const done = results.filter((result) => result.status === "done");
  const failed = results.filter((result) => result.status === "failed");
  const best = bestValues(done);
  const columns = visibleColumns();
  const table = clear(document.getElementById("metrics-table"));

  const header = el(
    "tr",
    {},
    columns.map((column) => {
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
      columns.map((column) => {
        const value = result[column.key];
        if (column.key === "model") return el("th", { scope: "row", text: value });
        if (column.key === "paradigm") return el("td", {}, paradigmBadge(value));
        const isBest = value !== null && value !== undefined && best[column.key] === value;
        return el("td", { class: `num ${isBest ? "best" : ""}`, text: format(column, value) });
      }),
    ),
  );
  const failedRows = failed.map((result) =>
    el("tr", { class: "failed-row" }, [
      el("th", { scope: "row", text: result.model }),
      el("td", {}, paradigmBadge(result.paradigm)),
      el("td", { colspan: columns.length - 2, text: `Failed: ${result.error}` }),
    ]),
  );
  table.append(el("thead", {}, header), el("tbody", {}, [...rows, ...failedRows]));
}

function detailText(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return String(value);
}

// Binary targets: the four cells named, so a model that never flags the positive class
// (0 true positives) is obvious at a glance.
function binaryCounts(result, positive) {
  const [[tn, fp], [fn, tp]] = result.confusion_matrix;
  const found = `Caught ${tp} of ${tp + fn} ${positive} rows`;
  const threshold = result.threshold === null || result.threshold === undefined ? "" : ` · threshold ${score(result.threshold)}`;
  return el("p", { class: "small cm-counts" }, [
    el("strong", { text: found }),
    ` · TP ${tp} · FP ${fp} (false alarms) · FN ${fn} (missed) · TN ${tn}${threshold}`,
  ]);
}

// REQ-RS-6: one heatmap per model. Cell shade is the share of the true class's rows.
function renderConfusion(done, classes, metadata) {
  const grid = clear(document.getElementById("confusion-grid"));
  const positive = metadata?.task === "binary" ? metadata.positive_class : null;
  const note = document.getElementById("confusion-note");
  note.classList.toggle("hidden", !positive);
  note.textContent = positive
    ? `Positive class: ${positive}. Precision, recall and F1 describe how well each model finds ${positive} rows. TP = ${positive} rows flagged, FN = ${positive} rows missed, FP = other rows wrongly flagged.`
    : "";
  done.forEach((result) => {
    const matrix = result.confusion_matrix;
    const head = el("tr", {}, [
      el("th", { scope: "col", class: "corner", text: "true ↓ / predicted →" }),
      ...classes.map((label) => el("th", { scope: "col", text: label })),
    ]);
    const body = matrix.map((row, i) => {
      const total = row.reduce((sum, count) => sum + count, 0) || 1;
      return el("tr", {}, [
        el("th", { scope: "row", text: classes[i] }),
        ...row.map((count, j) => {
          const share = count / total;
          return el("td", {
            class: `cm-cell ${i === j ? "cm-diag" : ""} ${share > 0.55 ? "cm-dark" : ""}`,
            style: `background:${paradigmColour(result.paradigm, 0.08 + 0.85 * share)}`,
            title: `${count} of ${total} rows (${(share * 100).toFixed(0)}%)`,
            text: count,
          });
        }),
      ]);
    });
    const details = Object.entries(result.details || {});
    grid.append(
      el("figure", { class: "cm-card" }, [
        el("figcaption", {}, [paradigmBadge(result.paradigm), ` ${result.model}`]),
        positive ? binaryCounts(result, positive) : null,
        el("div", { class: "table-wrap" }, [
          el("table", { class: "cm-table", "aria-label": `Confusion matrix for ${result.model}` }, [
            el("thead", {}, head),
            el("tbody", {}, body),
          ]),
        ]),
        details.length
          ? el(
              "p",
              { class: "muted small cm-details" },
              details.map(([key, value]) => `${key}: ${detailText(value)}`).join(" · "),
            )
          : null,
      ]),
    );
  });
}

// REQ-RS-8: the REQ-EV-5 fields plus what the preprocessing did.
function renderMetadata(metadata) {
  const list = clear(document.getElementById("run-meta"));
  if (!metadata) return;
  const listOrNone = (items) => (items && items.length ? items.join(", ") : "none");
  const binary = metadata.task === "binary";
  let thresholdText = "not used (multi-class: most probable class)";
  if (binary) {
    thresholdText =
      metadata.threshold_mode === "auto" || metadata.threshold_mode === undefined
        ? "tuned per model on the validation rows (best F1)"
        : `fixed at ${metadata.threshold_mode}`;
  }
  const entries = [
    ["Dataset", metadata.dataset],
    ["Rows", metadata.rows],
    ["Target", `${metadata.target} (${metadata.task})`],
    ["Classes", listOrNone(metadata.classes)],
    binary ? ["Positive class", metadata.positive_class] : null,
    binary && metadata.test_positive_rate !== null && metadata.test_positive_rate !== undefined
      ? ["Positive rate in test rows", pct(metadata.test_positive_rate)]
      : null,
    ["Features after encoding", metadata.features],
    [
      "Train / validation / test rows",
      `${metadata.train_rows} / ${metadata.validation_rows ?? 0} / ${metadata.test_rows} (test split ${Math.round(metadata.test_ratio * 100)}%)`,
    ],
    ["Decision threshold", thresholdText],
    ["Class weights", metadata.balance_classes === false ? "none" : "balanced"],
    ["Best model chosen by", rankText(state.results.summary)],
    ["Seed", metadata.seed],
    ["Split fingerprint", metadata.split_fingerprint],
    ["QML subsampled to", metadata.qml_subsampled_to ?? "QML not run"],
    [
      "PCA variance retained",
      metadata.pca_variance_retained === null || metadata.pca_variance_retained === undefined
        ? "QML not run"
        : `${(metadata.pca_variance_retained * 100).toFixed(1)}%`,
    ],
    ["DNN epochs (max) / QML epochs / qubits", `${metadata.dnn_epochs} / ${metadata.qml_epochs} / ${metadata.qubits}`],
    ["Dropped columns", listOrNone(metadata.dropped_columns)],
    ["Numeric columns", listOrNone(metadata.numeric_columns)],
    ["One-hot columns", listOrNone(metadata.one_hot_columns)],
    ["Ordinal columns", listOrNone(metadata.ordinal_columns)],
  ];
  entries.filter(Boolean).forEach(([label, value]) => {
    list.append(el("div", { class: "meta-item" }, [el("dt", { text: label }), el("dd", { text: value })]));
  });
}

// REQ-RS-9: the server builds the CSV and escapes formula characters (SEC-2).
function exportCsv() {
  const link = el("a", { href: resultsCsvUrl(state.jobId), download: "" });
  document.body.append(link);
  link.click();
  link.remove();
}

function renderChartsSection(done, metadata, summary) {
  const binary = metadata?.task === "binary";
  document.getElementById("roc-panel").classList.toggle("hidden", !binary);
  document.getElementById("pr-panel").classList.toggle("hidden", !binary);
  document.getElementById("roc-hidden").classList.toggle("hidden", binary);
  const [rankKey] = rankKeys(summary);
  const rankName = METRIC_NAMES[rankKey] || rankKey;
  document.getElementById("tradeoff-metric").textContent = rankName.charAt(0).toUpperCase() + rankName.slice(1);
  const ok = renderCharts(done, binary, {
    rankKey,
    rankName,
    positiveRate: metadata?.test_positive_rate,
  });
  const notice = document.getElementById("charts-unavailable");
  notice.classList.toggle("hidden", ok || !done.length);
  notice.textContent = ok
    ? ""
    : "Charts could not be drawn because Chart.js did not load. Check the internet connection and reload; the table above has every score.";
  document.querySelector(".chart-grid").classList.toggle("hidden", !ok);
}

export function renderResults() {
  const { summary, metadata, status } = state.results;
  const done = doneResults();
  sort = { key: rankKeys(summary)[0], descending: true };
  renderWinner(done, summary, status === "cancelled");
  renderParadigmCards(done, summary);
  renderTable();
  renderConfusion(done, metadata?.classes || [], metadata);
  renderMetadata(metadata);
  // Charts are drawn after the screen is visible so Chart.js can measure the canvases.
  requestAnimationFrame(() => renderChartsSection(done, metadata, summary));
}

export function initResults() {
  document.getElementById("export-csv").addEventListener("click", () => {
    if (!state.results) return;
    exportCsv();
    toast("Downloading the metrics table as CSV.", "info");
  });
  document.getElementById("retrain").addEventListener("click", () => {
    destroyCharts();
    showScreen("configure");
  });
  document.getElementById("new-run").addEventListener("click", () => {
    destroyCharts();
    showScreen("upload");
  });
}
