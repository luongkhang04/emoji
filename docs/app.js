import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

const MODEL_ID = "emoji-model";
const MODEL_BASE_PATH = "./models/";

env.allowRemoteModels = false;
env.localModelPath = MODEL_BASE_PATH;
env.useBrowserCache = true;

if (env.backends?.onnx?.wasm) {
  const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 4));
  env.backends.onnx.wasm.numThreads = threads;
}

const loadBtn = document.getElementById("loadBtn");
const unloadBtn = document.getElementById("unloadBtn");
const inputEl = document.getElementById("input");
const outputEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const latencyEl = document.getElementById("latency");
const topkEl = document.getElementById("topk");

let pipe = null;
let debounceId = null;
let inFlight = false;
let pending = false;
let labelCount = null;

function setStatus(message) {
  statusEl.textContent = message;
}

function setLatency(message) {
  latencyEl.textContent = message;
}

function getTexts() {
  return inputEl.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function updateStats() {
  const lines = getTexts().length;
  statsEl.textContent = `${lines} line${lines === 1 ? "" : "s"}`;
}

function clampTopK(value) {
  const maxValue = Number(topkEl.max) || labelCount || 10;
  const next = Math.max(1, Math.min(maxValue, value));
  if (Number(topkEl.value) !== next) {
    topkEl.value = String(next);
  }
  return next;
}

function formatResults(texts, results) {
  const normalized = Array.isArray(results) ? results : [results];
  const rows = normalized.map((entry) => (Array.isArray(entry) ? entry : [entry]));
  const blocks = rows.map((preds, index) => {
    const header = `Text: ${texts[index] || ""}`;
    const lines = preds.map(
      (pred, rank) => `  ${rank + 1}. ${pred.label} (${pred.score.toFixed(4)})`,
    );
    return [header, ...lines].join("\n");
  });
  return blocks.join("\n\n");
}

function hasInvalidScores(results) {
  const normalized = Array.isArray(results) ? results : [results];
  for (const entry of normalized) {
    const preds = Array.isArray(entry) ? entry : [entry];
    for (const pred of preds) {
      if (!Number.isFinite(pred.score)) {
        return true;
      }
    }
  }
  return false;
}

async function loadModelWithDtype(dtype, statusLabel) {
  const options = { device: "cpu", dtype };
  if (statusLabel) {
    setStatus(statusLabel);
  }
  pipe = await pipeline("text-classification", MODEL_ID, options);
  unloadBtn.disabled = false;
}

async function runPredict() {
  if (!pipe) {
    return;
  }
  if (inFlight) {
    pending = true;
    return;
  }
  inFlight = true;
  try {
    do {
      pending = false;
      const texts = getTexts();
      if (!texts.length) {
        outputEl.textContent = "Enter at least one line to predict.";
        setLatency("Idle");
        continue;
      }
      const topk = clampTopK(Number(topkEl.value) || 5);
      setStatus(`Running (${texts.length} inputs)...`);
      const start = performance.now();
      let results = await pipe(texts, {
        topk,
        return_all_scores: true,
        function_to_apply: "none",
      });
      if (hasInvalidScores(results)) {
        throw new Error("Model output contained invalid scores.");
      }
      const elapsed = performance.now() - start;
      outputEl.textContent = formatResults(texts, results);
      setLatency(`Last run: ${elapsed.toFixed(0)} ms`);
      setStatus("Ready");
    } while (pending);
  } catch (err) {
    outputEl.textContent = `Error: ${err?.message || err}`;
    setStatus("Error");
  } finally {
    inFlight = false;
  }
}

function schedulePredict() {
  updateStats();
  if (!pipe) {
    return;
  }
  if (debounceId) {
    clearTimeout(debounceId);
  }
  debounceId = setTimeout(() => {
    runPredict();
  }, 250);
}

async function loadModel() {
  try {
    loadBtn.disabled = true;
    setStatus("Loading model...");
    setLatency("Loading");
    await loadModelWithDtype("q8", "Loading model (q8)...");
    setStatus("Model loaded (q8).");
    setLatency("Ready");
    await runPredict();
  } catch (err) {
    setStatus(`Failed to load: ${err?.message || err}`);
    loadBtn.disabled = false;
  }
}

async function unloadModel() {
  unloadBtn.disabled = true;
  setStatus("Unloading model...");
  try {
    if (pipe) {
      await pipe.dispose();
      pipe = null;
    }
    outputEl.textContent = "Model unloaded.";
    setLatency("Idle");
    setStatus("Model not loaded.");
  } finally {
    loadBtn.disabled = false;
  }
}

async function loadLabelCount() {
  const candidates = [
    `${MODEL_BASE_PATH}${MODEL_ID}/id2label.json`,
    `${MODEL_BASE_PATH}${MODEL_ID}/config.json`,
  ];
  for (const url of candidates) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        continue;
      }
      const data = await res.json();
      const id2label = data.id2label || data;
      const count = typeof data.num_labels === "number" ? data.num_labels : null;
      if (id2label && typeof id2label === "object") {
        labelCount = Object.keys(id2label).length;
      } else if (count) {
        labelCount = count;
      }
      if (labelCount) {
        topkEl.max = String(labelCount);
        clampTopK(Number(topkEl.value) || 5);
        break;
      }
    } catch (err) {
      continue;
    }
  }
}

loadBtn.addEventListener("click", loadModel);
unloadBtn.addEventListener("click", unloadModel);
inputEl.addEventListener("input", schedulePredict);
topkEl.addEventListener("change", schedulePredict);

updateStats();
loadLabelCount().catch(() => {});
