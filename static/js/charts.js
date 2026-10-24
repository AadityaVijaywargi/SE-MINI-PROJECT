// Results charts (REQ-RS-4 to REQ-RS-7, UI-2). Renders Chart.js instances from the results
// payload and never talks to the server. Every chart colours models by paradigm and also
// names the paradigm in the label, so colour is never the only signal.

const PARADIGM_VARS = { ML: "--ml", DL: "--dl", QML: "--qml" };
const METRICS = [
  { key: "accuracy", label: "Accuracy", alpha: 1 },
  { key: "precision", label: "Precision", alpha: 0.75 },
  { key: "recall", label: "Recall", alpha: 0.5 },
  { key: "f1", label: "F1", alpha: 0.3 },
];
// Models of the same paradigm share a colour, so ROC lines also differ by dash pattern.
const DASHES = [[], [6, 4], [2, 3], [10, 3, 2, 3], [1, 6], [12, 6]];

const charts = [];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function paradigmColour(paradigm, alpha = 1) {
  const hex = cssVar(PARADIGM_VARS[paradigm] || "--muted") || "#888888";
  const value = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(value >> 16) & 255, (value >> 8) & 255, value & 255];
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export const modelLabel = (result) => `${result.model} (${result.paradigm})`;

const percentAxis = {
  min: 0,
  max: 1,
  ticks: { callback: (value) => `${Math.round(value * 100)}%` },
};

function secondsLabel(value) {
  if (value < 1) return `${(value * 1000).toFixed(value < 0.01 ? 1 : 0)} ms`;
  return `${value.toFixed(value < 10 ? 2 : 1)} s`;
}

function make(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  charts.push(new window.Chart(canvas, config));
}

function metricsChart(done) {
  make("chart-metrics", {
    type: "bar",
    data: {
      labels: done.map(modelLabel),
      datasets: METRICS.map((metric) => ({
        label: metric.label,
        data: done.map((result) => result[metric.key]),
        backgroundColor: done.map((result) => paradigmColour(result.paradigm, metric.alpha)),
        borderColor: done.map((result) => paradigmColour(result.paradigm)),
        borderWidth: 1,
      })),
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: percentAxis },
      plugins: {
        legend: {
          labels: {
            // Legend swatches show the metric's shade in neutral grey, since bar colour is
            // the paradigm.
            generateLabels: () =>
              METRICS.map((metric, index) => ({
                text: metric.label,
                fillStyle: `rgba(80, 99, 84, ${metric.alpha})`,
                strokeStyle: "rgb(80, 99, 84)",
                lineWidth: 1,
                datasetIndex: index,
              })),
          },
        },
        tooltip: {
          callbacks: {
            label: (item) => `${item.dataset.label}: ${(item.raw * 100).toFixed(1)}%`,
          },
        },
      },
    },
  });
}

function rocChart(done) {
  const counts = {};
  const datasets = done
    .filter((result) => result.roc_curve)
    .map((result) => {
      const index = counts[result.paradigm] || 0;
      counts[result.paradigm] = index + 1;
      return {
        label: `${modelLabel(result)} AUC ${result.roc_auc.toFixed(3)}`,
        data: result.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: result.roc_curve.tpr[i] })),
        borderColor: paradigmColour(result.paradigm),
        backgroundColor: paradigmColour(result.paradigm),
        borderDash: DASHES[index % DASHES.length],
        borderWidth: 2,
        pointRadius: 0,
        showLine: true,
        fill: false,
      };
    });
  datasets.push({
    label: "Random baseline",
    data: [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
    ],
    borderColor: "rgba(29, 56, 41, 0.6)",
    borderDash: [5, 5],
    borderWidth: 1,
    pointRadius: 0,
    showLine: true,
  });
  make("chart-roc", {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { ...percentAxis, title: { display: true, text: "False positive rate" } },
        y: { ...percentAxis, title: { display: true, text: "True positive rate" } },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 18 } } },
    },
  });
}

function timeChart(done) {
  const sorted = [...done].sort((a, b) => a.train_time - b.train_time);
  make("chart-time", {
    type: "bar",
    data: {
      labels: sorted.map(modelLabel),
      datasets: [
        {
          label: "Training time",
          // A log axis cannot show 0, so instant models are drawn at 0.1 ms.
          data: sorted.map((result) => Math.max(result.train_time, 1e-4)),
          backgroundColor: sorted.map((result) => paradigmColour(result.paradigm, 0.85)),
        },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Seconds (log scale)" },
          ticks: { callback: (value) => secondsLabel(Number(value)) },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (item) => secondsLabel(item.raw) } },
      },
    },
  });
}

function tradeoffChart(done) {
  const paradigms = ["ML", "DL", "QML"].filter((p) => done.some((r) => r.paradigm === p));
  make("chart-tradeoff", {
    type: "scatter",
    data: {
      datasets: paradigms.map((paradigm) => ({
        label: paradigm,
        data: done
          .filter((result) => result.paradigm === paradigm)
          .map((result) => ({
            x: Math.max(result.train_time, 1e-4),
            y: result.accuracy,
            model: result.model,
          })),
        backgroundColor: paradigmColour(paradigm, 0.85),
        pointRadius: 7,
        pointHoverRadius: 9,
        pointStyle: { ML: "circle", DL: "rectRot", QML: "triangle" }[paradigm],
      })),
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Training time (log scale)" },
          ticks: { callback: (value) => secondsLabel(Number(value)) },
        },
        y: { ...percentAxis, min: undefined, title: { display: true, text: "Test accuracy" } },
      },
      plugins: {
        legend: { position: "bottom", labels: { usePointStyle: true } },
        tooltip: {
          callbacks: {
            label: (item) =>
              `${item.raw.model} (${item.dataset.label}): ${(item.raw.y * 100).toFixed(1)}%, ${secondsLabel(item.raw.x)}`,
          },
        },
      },
    },
  });
}

export function destroyCharts() {
  while (charts.length) charts.pop().destroy();
}

// Returns false when Chart.js could not be loaded (for example, no internet access).
export function renderCharts(done, binary) {
  destroyCharts();
  if (!window.Chart || !done.length) return false;
  // Herbarium styling: mono labels, ink text, rule-coloured gridlines, ink tooltips.
  const { defaults } = window.Chart;
  defaults.font.family = cssVar("--mono") || "monospace";
  defaults.font.size = 11;
  defaults.color = cssVar("--ink-2") || "#506354";
  defaults.borderColor = cssVar("--grid") || "#dde3d4";
  defaults.plugins.tooltip.backgroundColor = cssVar("--ink") || "#1d3829";
  defaults.plugins.tooltip.cornerRadius = 0;
  defaults.elements.bar.borderRadius = 0;
  metricsChart(done);
  if (binary) rocChart(done);
  timeChart(done);
  tradeoffChart(done);
  return true;
}
