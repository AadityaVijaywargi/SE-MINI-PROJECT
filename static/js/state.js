// Single source of truth for the page (SRS 3.1). Other modules read `state` and change it
// only through setState / showScreen.

export const state = {
  screen: "upload",
  datasetId: null,
  datasetName: null,
  profile: null,
  suggestions: [],
  preview: null,
  target: null,
  analysis: null,
  selectedModels: [],
  settings: null,
  jobId: null,
  results: null,
};

const listeners = [];

export function setState(patch) {
  Object.assign(state, patch);
  listeners.forEach((listener) => listener(state));
}

export function onStateChange(listener) {
  listeners.push(listener);
}

// UI-1: exactly one screen is visible, toggled with the .hidden class.
export function showScreen(name) {
  document.querySelectorAll("[data-screen]").forEach((section) => {
    section.classList.toggle("hidden", section.dataset.screen !== name);
  });
  const order = ["upload", "configure", "train", "results"];
  document.querySelectorAll("[data-step]").forEach((step) => {
    const index = order.indexOf(step.dataset.step);
    step.classList.toggle("current", step.dataset.step === name);
    step.classList.toggle("done", index < order.indexOf(name));
    if (step.dataset.step === name) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  });
  setState({ screen: name });
  const heading = document.querySelector(`[data-screen="${name}"] h1`);
  if (heading) heading.focus({ preventScroll: true });
  window.scrollTo({ top: 0 });
}
