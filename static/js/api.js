// The only module that talks to the server, and the one place errors reach the user (UI-6).

export class ApiError extends Error {}

export function toast(message, kind = "error") {
  const box = document.getElementById("toasts");
  const item = document.createElement("div");
  item.className = `toast toast-${kind}`;
  item.setAttribute("role", kind === "error" ? "alert" : "status");
  item.textContent = message;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-close";
  close.setAttribute("aria-label", "Dismiss message");
  close.textContent = "×";
  close.addEventListener("click", () => item.remove());
  item.append(close);
  box.append(item);
  setTimeout(() => item.remove(), 8000);
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    const message = "Cannot reach the server. Check that it is running and try again.";
    toast(message);
    throw new ApiError(message);
  }
  let body = {};
  try {
    body = await response.json();
  } catch {
    body = {};
  }
  if (!response.ok) {
    const message = body.error || `Request failed (${response.status}).`;
    toast(message);
    throw new ApiError(message);
  }
  return body;
}

function postJson(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// Multipart upload with progress (COM-1). XHR is used because fetch cannot report upload progress.
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(Math.round((100 * event.loaded) / event.total));
    });
    xhr.addEventListener("load", () => {
      const body = xhr.response || {};
      if (xhr.status === 201) {
        resolve(body);
        return;
      }
      const message = body.error || `Upload failed (${xhr.status}).`;
      toast(message);
      reject(new ApiError(message));
    });
    xhr.addEventListener("error", () => {
      const message = "Upload failed: cannot reach the server.";
      toast(message);
      reject(new ApiError(message));
    });
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export const configure = (datasetId, target) =>
  postJson("/api/configure", { dataset_id: datasetId, target });

export const startTraining = (payload) => postJson("/api/train", payload);

export const getStatus = (jobId) => request(`/api/status/${jobId}`);

export const cancelJob = (jobId) => postJson(`/api/cancel/${jobId}`, {});

export const getResults = (jobId) => request(`/api/results/${jobId}`);

export const resultsCsvUrl = (jobId) => `/api/results/${encodeURIComponent(jobId)}/csv`;

export const listSamples = () => request("/api/samples");

export const loadSample = (name) => postJson(`/api/samples/${encodeURIComponent(name)}`, {});
