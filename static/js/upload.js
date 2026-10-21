// Upload screen: drop zone, client-side checks, summary cards, preview and column table
// (REQ-UP-1, REQ-UP-2, REQ-UP-8).

import { toast, uploadFile } from "./api.js";
import { clear, el, formatBytes } from "./dom.js";
import { setState, showScreen, state } from "./state.js";
import { prepareConfigure } from "./config.js";

const ALLOWED_EXTENSIONS = [".csv", ".xlsx"];
const MAX_BYTES = 100 * 1024 * 1024;

function checkFile(file) {
  const name = file.name.toLowerCase();
  if (!ALLOWED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    return `"${file.name}" is not a .csv or .xlsx file.`;
  }
  if (file.size > MAX_BYTES) {
    return `"${file.name}" is ${formatBytes(file.size)}; the limit is 100 MB.`;
  }
  if (file.size === 0) return `"${file.name}" is empty.`;
  return null;
}

export async function onUploadDrop(file) {
  const problem = checkFile(file);
  if (problem) {
    toast(problem);
    return;
  }
  const progress = document.getElementById("upload-progress");
  const progressText = document.getElementById("upload-progress-text");
  progress.classList.remove("hidden");
  progressText.textContent = `Uploading ${file.name}…`;
  try {
    const body = await uploadFile(file, (percent) => {
      progressText.textContent =
        percent < 100 ? `Uploading ${file.name}… ${percent}%` : "Profiling columns…";
    });
    setState({
      datasetId: body.dataset_id,
      datasetName: body.name,
      profile: body.profile,
      suggestions: body.suggested_targets,
      preview: body.preview,
      target: null,
      analysis: null,
      jobId: null,
      results: null,
    });
    renderProfile();
  } catch {
    // api.js has already shown the error.
  } finally {
    progress.classList.add("hidden");
    document.getElementById("file-input").value = "";
  }
}

function summaryCard(label, value, note) {
  return el("div", { class: "card" }, [
    el("div", { class: "card-label", text: label }),
    el("div", { class: "card-value", text: value }),
    note ? el("div", { class: "card-note", text: note }) : null,
  ]);
}

function renderPreview(preview) {
  const table = clear(document.getElementById("preview-table"));
  table.append(
    el("thead", {}, el("tr", {}, preview.columns.map((name) => el("th", { scope: "col", text: name })))),
    el(
      "tbody",
      {},
      preview.rows.map((row) =>
        el("tr", {}, row.map((cell) => el("td", { class: cell === "" ? "missing" : "", text: cell }))),
      ),
    ),
  );
}

const ROLE_NOTES = {
  constant: "Dropped: only one value",
  identifier: "Dropped: looks like an ID",
};

function renderColumns(columns) {
  const table = clear(document.getElementById("column-table"));
  const headings = ["Column", "Type", "Unique", "Missing", "Role", "Examples"];
  table.append(
    el("thead", {}, el("tr", {}, headings.map((name) => el("th", { scope: "col", text: name })))),
    el(
      "tbody",
      {},
      columns.map((column) => {
        const warnings = [];
        if (ROLE_NOTES[column.role]) warnings.push(ROLE_NOTES[column.role]);
        if (column.missing_pct >= 50) {
          warnings.push(`${column.missing_pct}% missing: consider removing it`);
        }
        return el("tr", {}, [
          el("th", { scope: "row", text: column.name }),
          el("td", {}, el("span", { class: `badge badge-kind`, text: column.kind })),
          el("td", { class: "num", text: column.unique }),
          el("td", {
            class: `num ${column.missing ? "warn-text" : ""}`,
            text: column.missing ? `${column.missing} (${column.missing_pct}%)` : "0",
          }),
          el("td", {}, [
            el("span", { class: `badge badge-role-${column.role}`, text: column.role }),
            warnings.length
              ? el("span", { class: "warn-icon", title: warnings.join("; "), text: " ⚠" }, [
                  el("span", { class: "visually-hidden", text: warnings.join("; ") }),
                ])
              : null,
          ]),
          el("td", { class: "muted", text: column.samples.join(", ") }),
        ]);
      }),
    ),
  );
}

function renderProfile() {
  const { profile, preview, datasetName } = state;
  document.getElementById("dataset-name").textContent = datasetName;
  const dropped = profile.column_profiles.filter((column) => column.role !== "feature").length;
  clear(document.getElementById("summary-cards")).append(
    summaryCard("Rows", profile.rows.toLocaleString()),
    summaryCard("Columns", profile.columns, dropped ? `${dropped} will be dropped` : null),
    summaryCard("Missing values", profile.missing_cells.toLocaleString()),
    summaryCard("Memory", formatBytes(profile.memory_bytes)),
  );
  renderPreview(preview);
  renderColumns(profile.column_profiles);
  document.getElementById("profile").classList.remove("hidden");
  document.getElementById("to-configure").focus();
}

export function initUpload() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");

  input.addEventListener("change", () => {
    if (input.files.length) onUploadDrop(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, () => zone.classList.remove("dragging")),
  );
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    const [file] = event.dataTransfer.files;
    if (file) onUploadDrop(file);
  });
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  zone.tabIndex = 0;

  document.getElementById("to-configure").addEventListener("click", () => {
    prepareConfigure();
    showScreen("configure");
  });
}
