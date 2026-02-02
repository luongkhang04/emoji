import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

const MODEL_ID = "emoji-model";
const MODEL_BASE_PATH = "./models/";
const SUGGESTION_COUNT = 5;
const DEBOUNCE_MS = 250;

env.allowRemoteModels = false;
env.localModelPath = MODEL_BASE_PATH;
env.useBrowserCache = true;

if (env.backends?.onnx?.wasm) {
  const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 4));
  env.backends.onnx.wasm.numThreads = threads;
}

const inputEl = document.getElementById("input");
const emojiBarEl = document.getElementById("emojiBar");
const statusEl = document.getElementById("status");
const copyBtn = document.getElementById("copyBtn");

let pipe = null;
let debounceId = null;
let inFlight = false;
let pending = false;
let lastSentence = "";

function setStatus(message) {
  statusEl.textContent = message;
}

function getCurrentSentence(text) {
  const trimmed = text.trim();
  if (!trimmed) {
    return "";
  }
  const parts = trimmed.split(/[\n.!?]+/);
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const sentence = stripEmojis(parts[i].trim());
    if (sentence) {
      return sentence;
    }
  }
  return stripEmojis(trimmed);
}

function stripEmojis(text) {
  return text.replace(
    /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{200D}]/gu,
    "",
  );
}

function isEmojiChar(char) {
  if (!char) {
    return false;
  }
  return /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{200D}]/u.test(char);
}

function renderPlaceholders(message) {
  emojiBarEl.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "emoji-placeholder";
  placeholder.textContent = message;
  emojiBarEl.appendChild(placeholder);
}

function renderSuggestions(predictions) {
  emojiBarEl.innerHTML = "";
  predictions.forEach((pred) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "emoji-chip";
    button.textContent = pred.label;

    button.addEventListener("click", () => {
      insertEmoji(pred.label);
    });
    emojiBarEl.appendChild(button);
  });
}

function getFirstSuggestion() {
  return emojiBarEl.querySelector(".emoji-chip");
}

function insertEmoji(emoji) {
  const start = inputEl.selectionStart ?? inputEl.value.length;
  const end = inputEl.selectionEnd ?? inputEl.value.length;
  const before = inputEl.value.slice(0, start);
  const after = inputEl.value.slice(end);
  let insert = emoji;
  const lastChar = before.slice(-1);
  const firstChar = after.slice(0, 1);
  if (before && !/\s$/.test(before) && !isEmojiChar(lastChar)) {
    insert = ` ${insert}`;
  }
  if (after && !/^\s/.test(after) && !isEmojiChar(firstChar)) {
    insert = `${insert} `;
  }
  const nextValue = before + insert + after;
  const cursor = (before + insert).length;
  inputEl.value = nextValue;
  inputEl.focus();
  inputEl.setSelectionRange(cursor, cursor);
  scheduleSuggest();
}

function hasInvalidScores(predictions) {
  return predictions.some((pred) => !Number.isFinite(pred.score));
}

async function suggest() {
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
      const sentence = getCurrentSentence(inputEl.value);
      if (!sentence) {
        renderPlaceholders("Start typing to see emoji suggestions.");
        continue;
      }
      if (sentence === lastSentence) {
        continue;
      }
      lastSentence = sentence;
      setStatus("Thinking...");
      const result = await pipe(sentence, {
        topk: SUGGESTION_COUNT,
        return_all_scores: true,
        function_to_apply: "none",
      });
      const predictions = Array.isArray(result) ? result : [result];
      if (hasInvalidScores(predictions)) {
        throw new Error("Model output contained invalid scores.");
      }
      renderSuggestions(predictions);
      setStatus("Ready");
    } while (pending);
  } catch (err) {
    renderPlaceholders("Unable to generate suggestions.");
    setStatus(`Error: ${err?.message || err}`);
  } finally {
    inFlight = false;
  }
}

function scheduleSuggest() {
  if (debounceId) {
    clearTimeout(debounceId);
  }
  debounceId = setTimeout(() => {
    suggest();
  }, DEBOUNCE_MS);
}

async function loadModel() {
  setStatus("Loading model...");
  renderPlaceholders("Loading model…");
  try {
    pipe = await pipeline("text-classification", MODEL_ID, {
      device: "cpu",
      dtype: "q8",
    });
    setStatus("Ready");
    renderPlaceholders("Start typing to see emoji suggestions.");
    scheduleSuggest();
  } catch (err) {
    renderPlaceholders("Failed to load model.");
    setStatus(`Failed to load: ${err?.message || err}`);
  }
}

inputEl.addEventListener("input", scheduleSuggest);
inputEl.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") {
    return;
  }
  if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }
  const firstSuggestion = getFirstSuggestion();
  if (!firstSuggestion) {
    return;
  }
  event.preventDefault();
  insertEmoji(firstSuggestion.textContent);
});
copyBtn.addEventListener("click", async () => {
  const text = inputEl.value;
  if (!text) {
    setStatus("Nothing to copy.");
    return;
  }
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      inputEl.select();
      document.execCommand("copy");
      inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length);
    }
    setStatus("Copied!");
  } catch (err) {
    setStatus("Copy failed.");
  }
});
loadModel();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  });
}
