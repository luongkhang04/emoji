(() => {
  "use strict";

  const MODEL_ID = "emoji-model";
  const MODEL_BASE_PATH = chrome.runtime.getURL("models/");
  const TRANSFORMERS_URL = chrome.runtime.getURL("vendor/transformers.min.js");
  const WASM_BASE_PATH = chrome.runtime.getURL("vendor/");
  const SUGGESTION_THRESHOLD_PERCENT = 10;
  const DEBOUNCE_MS = 200;
  const EMOJI_SEQUENCE_SOURCE =
    "\\p{Extended_Pictographic}(?:\\uFE0F|\\uFE0E)?(?:\\u200D\\p{Extended_Pictographic}(?:\\uFE0F|\\uFE0E)?)*|\\p{Emoji_Presentation}|\\p{Emoji}\\uFE0F";
  const EMOJI_SEQUENCE_REGEX = new RegExp(EMOJI_SEQUENCE_SOURCE, "gu");
  const EMOJI_SEQUENCE_TEST_REGEX = new RegExp(EMOJI_SEQUENCE_SOURCE, "u");
  const graphemeSegmenter =
    typeof Intl !== "undefined" && Intl.Segmenter
      ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
      : null;

  let pipe = null;
  let pipePromise = null;
  let loadError = null;

  let activeTarget = null;
  let suggestion = null;
  let suggestionEl = null;
  let debounceId = null;
  let lastSentence = "";
  let lastTarget = null;
  let inFlight = false;
  let pending = false;
  let positionPending = false;

  async function loadPipeline() {
    if (pipe) {
      return pipe;
    }
    if (pipePromise) {
      return pipePromise;
    }
    pipePromise = (async () => {
      const module = await import(TRANSFORMERS_URL);
      const pipeline = module.pipeline;
      const env = module.env;

      env.allowRemoteModels = false;
      env.localModelPath = MODEL_BASE_PATH;
      env.useBrowserCache = true;

      if (env.backends?.onnx?.wasm) {
        const threads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 4));
        env.backends.onnx.wasm.numThreads = threads;
        env.backends.onnx.wasm.wasmPaths = WASM_BASE_PATH;
      }

      const instance = await pipeline("text-classification", MODEL_ID, {
        device: "cpu",
        dtype: "q8",
      });
      pipe = instance;
      return instance;
    })();

    pipePromise.catch((err) => {
      loadError = err;
      pipePromise = null;
      console.warn("Emoji model failed to load:", err);
    });

    return pipePromise;
  }

  function normalizePredictions(result) {
    const normalized = Array.isArray(result) ? result : [result];
    const first = normalized[0];
    return Array.isArray(first) ? first : normalized;
  }

  async function predictEmoji(text) {
    if (loadError) {
      return null;
    }
    let pipelineInstance = null;
    try {
      pipelineInstance = await loadPipeline();
    } catch (err) {
      return null;
    }
    if (!pipelineInstance) {
      return null;
    }

    const result = await pipelineInstance(text, { topk: 1 });
    const preds = normalizePredictions(result);
    const best = preds[0];
    if (!best || !Number.isFinite(best.score)) {
      return null;
    }
    const probability = best.score * 100;
    if (probability <= SUGGESTION_THRESHOLD_PERCENT) {
      return null;
    }
    return {
      emoji: best.label,
      probability,
    };
  }

  function stripEmojis(text) {
    return text.replace(EMOJI_SEQUENCE_REGEX, "");
  }

  function getFirstGrapheme(text) {
    if (!text) {
      return "";
    }
    if (graphemeSegmenter) {
      const iterator = graphemeSegmenter.segment(text)[Symbol.iterator]();
      const first = iterator.next();
      return first.done ? "" : first.value.segment;
    }
    return Array.from(text)[0] || "";
  }

  function getLastGrapheme(text) {
    if (!text) {
      return "";
    }
    if (graphemeSegmenter) {
      let last = "";
      for (const part of graphemeSegmenter.segment(text)) {
        last = part.segment;
      }
      return last;
    }
    const chars = Array.from(text);
    return chars[chars.length - 1] || "";
  }

  function isEmojiChar(char) {
    if (!char) {
      return false;
    }
    return EMOJI_SEQUENCE_TEST_REGEX.test(char);
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

  function isEditable(el) {
    if (!el) {
      return false;
    }
    if (el.readOnly || el.disabled) {
      return false;
    }
    const tag = el.tagName;
    if (tag === "TEXTAREA") {
      return true;
    }
    if (tag === "INPUT") {
      const type = (el.type || "text").toLowerCase();
      return ["text", "search", "url", "email", "tel"].includes(type);
    }
    return Boolean(el.isContentEditable);
  }

  function getTextFromTarget(target) {
    if (!target) {
      return "";
    }
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
      return target.value || "";
    }
    if (target.isContentEditable) {
      return target.innerText || target.textContent || "";
    }
    return "";
  }

  function ensureSuggestionEl() {
    if (suggestionEl) {
      return suggestionEl;
    }
    const el = document.createElement("div");
    el.className = "emoji-suggest-overlay";
    el.setAttribute("aria-hidden", "true");
    el.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (activeTarget && suggestion) {
        applySuggestion(activeTarget, suggestion.emoji);
        hideSuggestion();
      }
    });
    document.body.appendChild(el);
    suggestionEl = el;
    return el;
  }

  function renderSuggestion(nextSuggestion) {
    const el = ensureSuggestionEl();
    el.innerHTML = "";
    const emojiSpan = document.createElement("span");
    emojiSpan.className = "emoji-suggest-emoji";
    emojiSpan.textContent = nextSuggestion.emoji;

    const keySpan = document.createElement("span");
    keySpan.className = "emoji-suggest-key";
    keySpan.textContent = "Tab";

    el.append(emojiSpan, keySpan);
    el.classList.add("visible");
  }

  function hideSuggestion() {
    if (suggestionEl) {
      suggestionEl.classList.remove("visible");
    }
    suggestion = null;
  }

  function getTargetRect(target) {
    if (!target) {
      return null;
    }
    if (target.isContentEditable) {
      const selection = window.getSelection();
      if (selection && selection.rangeCount) {
        const range = selection.getRangeAt(0).cloneRange();
        range.collapse(true);
        const rect = range.getBoundingClientRect();
        if (rect && (rect.width > 0 || rect.height > 0)) {
          return rect;
        }
      }
    }
    const rect = target.getBoundingClientRect();
    if (rect && (rect.width > 0 || rect.height > 0)) {
      return rect;
    }
    return null;
  }

  function positionSuggestion() {
    if (!suggestionEl || !suggestion || !activeTarget) {
      return;
    }
    const rect = getTargetRect(activeTarget);
    if (!rect) {
      hideSuggestion();
      return;
    }
    const padding = 8;
    const aboveTop = window.scrollY + rect.top - suggestionEl.offsetHeight - 6;
    const belowTop = window.scrollY + rect.bottom + 6;
    const top = aboveTop >= padding ? aboveTop : belowTop;
    const maxLeft =
      window.scrollX +
      document.documentElement.clientWidth -
      suggestionEl.offsetWidth -
      padding;
    const left = Math.min(window.scrollX + rect.left, maxLeft);
    suggestionEl.style.top = `${Math.max(top, padding)}px`;
    suggestionEl.style.left = `${Math.max(left, padding)}px`;
  }

  function schedulePosition() {
    if (positionPending) {
      return;
    }
    positionPending = true;
    requestAnimationFrame(() => {
      positionPending = false;
      positionSuggestion();
    });
  }

  async function updateSuggestion(target) {
    if (!isEditable(target)) {
      hideSuggestion();
      return;
    }
    const text = getTextFromTarget(target);
    const sentence = getCurrentSentence(text);
    if (!sentence) {
      hideSuggestion();
      return;
    }
    if (sentence === lastSentence && target === lastTarget && suggestion) {
      schedulePosition();
      return;
    }
    lastSentence = sentence;
    lastTarget = target;

    const nextSuggestion = await predictEmoji(sentence);
    if (target !== activeTarget) {
      return;
    }
    if (!nextSuggestion) {
      hideSuggestion();
      return;
    }
    suggestion = nextSuggestion;
    renderSuggestion(nextSuggestion);
    schedulePosition();
  }

  async function suggest(target) {
    if (inFlight) {
      pending = true;
      return;
    }
    inFlight = true;
    try {
      do {
        pending = false;
        await updateSuggestion(target);
      } while (pending);
    } finally {
      inFlight = false;
    }
  }

  function scheduleUpdate(target) {
    if (debounceId) {
      clearTimeout(debounceId);
    }
    debounceId = setTimeout(() => {
      suggest(target);
    }, DEBOUNCE_MS);
  }

  function insertEmojiIntoInput(target, emoji) {
    const start = Number.isInteger(target.selectionStart)
      ? target.selectionStart
      : target.value.length;
    const end = Number.isInteger(target.selectionEnd)
      ? target.selectionEnd
      : target.value.length;
    const before = target.value.slice(0, start);
    const after = target.value.slice(end);
    let insert = emoji;
    const lastChar = getLastGrapheme(before);
    const firstChar = getFirstGrapheme(after);
    if (before && !/\s$/.test(before) && !isEmojiChar(lastChar)) {
      insert = ` ${insert}`;
    }
    if (after && !/^\s/.test(after) && !isEmojiChar(firstChar)) {
      insert = `${insert} `;
    }
    const nextValue = before + insert + after;
    const cursor = (before + insert).length;
    target.value = nextValue;
    target.setSelectionRange(cursor, cursor);
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function insertEmojiIntoContentEditable(target, emoji) {
    const selection = window.getSelection();
    if (!selection || !selection.rangeCount) {
      return;
    }
    const range = selection.getRangeAt(0);
    if (!target.contains(range.startContainer)) {
      return;
    }
    let prefix = "";
    if (range.startContainer.nodeType === Node.TEXT_NODE) {
      const text = range.startContainer.textContent || "";
      const before = text.slice(0, range.startOffset);
      const lastChar = getLastGrapheme(before);
      if (before && !/\s$/.test(before) && !isEmojiChar(lastChar)) {
        prefix = " ";
      }
    }
    let suffix = " ";
    if (range.startContainer.nodeType === Node.TEXT_NODE) {
      const text = range.startContainer.textContent || "";
      const after = text.slice(range.startOffset);
      const firstChar = getFirstGrapheme(after);
      if (!after || /^\s/.test(after) || isEmojiChar(firstChar)) {
        suffix = "";
      }
    }
    const textNode = document.createTextNode(`${prefix}${emoji}${suffix}`);
    range.deleteContents();
    range.insertNode(textNode);
    range.setStartAfter(textNode);
    range.setEndAfter(textNode);
    selection.removeAllRanges();
    selection.addRange(range);
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function applySuggestion(target, emoji) {
    if (!target || !emoji) {
      return;
    }
    if (target.isContentEditable) {
      insertEmojiIntoContentEditable(target, emoji);
    } else {
      insertEmojiIntoInput(target, emoji);
    }
  }

  function handleInput(event) {
    const target = event.target;
    if (!isEditable(target)) {
      return;
    }
    activeTarget = target;
    scheduleUpdate(target);
  }

  function handleFocusIn(event) {
    const target = event.target;
    if (!isEditable(target)) {
      activeTarget = null;
      hideSuggestion();
      return;
    }
    activeTarget = target;
    if (!pipe && !pipePromise) {
      loadPipeline().catch(() => {});
    }
    scheduleUpdate(target);
  }

  function handleKeydown(event) {
    if (event.key !== "Tab") {
      return;
    }
    if (!suggestion || !activeTarget) {
      return;
    }
    if (event.target !== activeTarget) {
      return;
    }
    event.preventDefault();
    applySuggestion(activeTarget, suggestion.emoji);
    hideSuggestion();
  }

  document.addEventListener("input", handleInput, true);
  document.addEventListener("focusin", handleFocusIn, true);
  document.addEventListener("keydown", handleKeydown, true);
  window.addEventListener(
    "scroll",
    () => {
      if (suggestion) {
        schedulePosition();
      }
    },
    true,
  );
  window.addEventListener("resize", () => {
    if (suggestion) {
      schedulePosition();
    }
  });
})();
