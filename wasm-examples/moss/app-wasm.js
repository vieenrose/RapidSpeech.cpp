/* zh-TW meeting transcriber — RapidSpeech.cpp / WebAssembly edition.
 *
 * UI/UX ported from the onnxruntime-web demo (space_local/index.html +
 * app.js): drop zone, examples, live streaming transcript, speaker legend
 * with click-to-filter, search, SRT/JSON export, heartbeat, progress bar,
 * click-to-seek player. The inference engine is different: instead of an
 * onnxruntime-web worker it drives a ggml Q4_K GGUF through RapidSpeech.cpp
 * compiled to multi-threaded WASM, hosted in a Web Worker (moss-worker.js) so
 * the page stays responsive. The decoder streams each token back live (via a
 * C++ postMessage hook), so the transcript fills in token by token; Stop takes
 * effect between windows.
 *
 * Speaker diarization is the MODEL's own [Sxx] tags — no ECAPA embedding or
 * cross-window clustering.
 */

import { itn } from "./itn.js";

const $ = (id) => document.getElementById(id);

/* ---------------------------- configuration ------------------------------ */
const GGUF_URL = new URLSearchParams(location.search).get("gguf") ||
  "https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-gguf/resolve/main/moss-td-zhtw-v5kl-q4_k_m.gguf";
// CAM++ 192-d speaker encoder (~14 MB) — same-origin, for cross-window speaker
// linking. Absent -> falls back to per-window [Sxx] tags.
const SPK_GGUF_URL = new URLSearchParams(location.search).get("spk") || "./campplus.gguf";
const PROMPT = "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。";
const EXAMPLES =
  "https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-onnx/resolve/main/demo/";

const SR = 16000, N_MEL = 80, N_FRAMES = 3000;
// iOS WebKit refuses the desktop build's 4 GB shared-memory reservation and
// caps tab memory ~1.5 GB, so iOS gets a small-max WASM variant + short windows
// (cross-window CAM++ linking keeps speaker IDs consistent regardless).
const IS_IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const WASM_VARIANT = IS_IOS ? "ios" : "mt";
// Window size: user-selectable 90/180/300 s (default 180, like the ORT demo —
// 300 s peaks ~3 GB in-WASM and can fail on long meetings). iOS is memory-capped
// so it always uses a single 28 s chunk; cross-window CAM++ linking keeps
// speaker IDs consistent regardless of window size.
function currentWindowS() {
  if (IS_IOS) return 28;
  return +(document.querySelector('input[name="win"]:checked')?.value || 180);
}
const MIC_MAX_S = 120;
const WASM_BASE = "rapidspeech-wasm-mt";

const PALETTE = ["#4f7cff", "#e05563", "#2aa876", "#c78c2c", "#9761d8",
                 "#3aa6b9", "#d0679d", "#7a9a01", "#b05c45", "#5c7285",
                 "#8884d8", "#c05780", "#4a9d5f", "#b8860b", "#7b68ee",
                 "#20a4b5", "#cc6699", "#899a20"];

// Resolve on the next macrotask turn so the DOM can paint the just-rendered
// window and a pending "abort" click is processed before the next (blocking)
// WASM transcribe call.
const yieldToLoop = () => new Promise((r) => setTimeout(r));

/* ---------------------------- run state ---------------------------------- */
let busy = false, aborted = false;
let segs = [], rows = [], activeIdx = -1, hiddenSpk = new Set();
let tokenCount = 0;   // decoder tokens streamed this run (for tok/s)
let diarSegs = [];    // accumulated segments (with CAM++ embeddings) across windows
let diarize = false;  // true once the CAM++ speaker model loaded (cross-window linking)

function acquireBusy() {
  if (busy) return false;
  busy = true;
  setControlsEnabled(false);
  return true;
}
function releaseBusy() {
  busy = false;
  setControlsEnabled(true);
}
function setControlsEnabled(enabled) {
  $("file-in").disabled = !enabled;
  $("btn-rec").disabled = !enabled;
  drop.classList.toggle("disabled", !enabled);
  document.querySelectorAll("[data-ex]").forEach((b) => { b.disabled = !enabled; });
}

// OpenCC Simplified->Traditional (s2tw): the q4-quantized model emits zh-CN
// more often; convert to zh-TW like the ONNX demo. OpenCC (cn2t.js) is loaded
// as a classic script in index.html; the converter is created lazily.
let _s2twConv = null;
function s2tw(t) {
  if (_s2twConv === null) {
    try {
      _s2twConv = globalThis.OpenCC
        ? globalThis.OpenCC.Converter({ from: "cn", to: "tw" }) : false;
    } catch { _s2twConv = false; }
  }
  // Match the ORT demo: OpenCC (Simplified->Traditional) THEN number ITN
  // (十一人 -> 11人, 百分之X -> X%), with itn.js's idiom guards.
  return itn(_s2twConv ? _s2twConv(t) : t);
}
const dispCache = new Map();
function dispText(text) {
  let v = dispCache.get(text);
  if (v === undefined) { v = text.replace(/</g, "&lt;"); dispCache.set(text, v); }
  return v;
}

/* ======================= HF-exact Whisper log-mel ========================= */
/* Copied from space_local/pipeline.js melSpectrogram() + its DFT setup.
 * Reads window + filters from mel.bin and shape params from mel-config.json. */
let MEL = null;
async function loadMel() {
  const cfg = await (await fetch("mel-config.json")).json();
  const bin = new Float32Array(await (await fetch("mel.bin")).arrayBuffer());
  const { n_fft, hop, n_mel, n_freq, chunk_frames } = cfg;
  const window = bin.slice(0, n_fft);
  const filters = bin.slice(n_fft);          // [n_freq * n_mel], row-major [freq][mel]
  const dftCos = new Float32Array(n_freq * n_fft);
  const dftSin = new Float32Array(n_freq * n_fft);
  for (let k = 0; k < n_freq; k++)
    for (let n = 0; n < n_fft; n++) {
      const a = (2 * Math.PI * k * n) / n_fft;
      dftCos[k * n_fft + n] = Math.cos(a);
      dftSin[k * n_fft + n] = Math.sin(a);
    }
  MEL = { n_fft, hop, n_mel, n_freq, chunk_frames, window, filters, dftCos, dftSin };
}

function melSpectrogram(wav /* Float32Array, <=30 s @16 kHz */) {
  const { n_fft, hop, n_mel, n_freq, chunk_frames, window, filters, dftCos, dftSin } = MEL;
  const half = n_fft >> 1;
  const nSamples = 480000; // whisper pads/truncates to 30 s before STFT
  const padded = new Float32Array(nSamples + n_fft);
  const sig = wav.length > nSamples ? wav.subarray(0, nSamples) : wav;
  padded.set(sig, half);
  for (let i = 0; i < half; i++) padded[half - 1 - i] = sig[i + 1] || 0;
  if (sig.length === nSamples)
    for (let i = 0; i < half; i++) padded[half + nSamples + i] = sig[nSamples - 2 - i];
  const frame = new Float32Array(n_fft), power = new Float32Array(n_freq);
  const mel = new Float32Array(n_mel * chunk_frames); // [mel][frame]
  for (let t = 0; t < chunk_frames; t++) {
    const off = t * hop;
    for (let n = 0; n < n_fft; n++) frame[n] = padded[off + n] * window[n];
    for (let k = 0; k < n_freq; k++) {
      let re = 0, im = 0; const base = k * n_fft;
      for (let n = 0; n < n_fft; n++) { re += dftCos[base + n] * frame[n]; im += dftSin[base + n] * frame[n]; }
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < n_mel; m++) {
      let acc = 0; for (let k = 0; k < n_freq; k++) acc += filters[k * n_mel + m] * power[k];
      mel[m * chunk_frames + t] = Math.log10(Math.max(acc, 1e-10));
    }
  }
  let mx = -Infinity; for (let i = 0; i < mel.length; i++) if (mel[i] > mx) mx = mel[i];
  const floor = mx - 8.0;
  for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], floor) + 4.0) / 4.0;
  return mel; // [80 * 3000] bin-slow / frame-fast
}

/* ============================ transcript view ============================ */
function fmt(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60),
        s = Math.floor(t % 60);
  return (h ? h + ":" : "") + String(m).padStart(h ? 2 : 1, "0") + ":" +
         String(s).padStart(2, "0");
}

function colorsFor(list) {
  const speakers = [...new Set(list.map((s) => s.speaker).filter((x) => x !== "…"))];
  const c = Object.fromEntries(speakers.map((s, i) => [s, PALETTE[i % PALETTE.length]]));
  c["…"] = "var(--muted)";
  return c;
}

function renderLegend() {
  const talk = {};
  for (const s of segs) {
    if (s.speaker === "…") continue;
    talk[s.speaker] = (talk[s.speaker] || 0) + Math.max(0, s.end - s.start);
  }
  const color = colorsFor(segs);
  $("legend").innerHTML = Object.keys(talk)
    .sort((a, b) => talk[b] - talk[a])
    .map((s) => `<span class="lg${hiddenSpk.has(s) ? " off" : ""}" data-spk="${s}">` +
                `<i style="background:${color[s]}"></i>${s} · ${fmt(talk[s])}</span>`)
    .join("");
  document.querySelectorAll(".lg").forEach((el) => {
    el.onclick = () => {
      const s = el.dataset.spk;
      hiddenSpk.has(s) ? hiddenSpk.delete(s) : hiddenSpk.add(s);
      el.classList.toggle("off", hiddenSpk.has(s));
      applyFilter();
    };
  });
}

function renderTranscript(tail = "") {
  const color = colorsFor(segs);
  const box = $("transcript");
  box.innerHTML = (segs.map((s, i) =>
    `<div class="seg${s.speaker === "…" ? " live" : ""}" data-i="${i}">` +
    `<span class="ts">${fmt(s.start)}</span>` +
    `<span class="spk" style="color:${color[s.speaker]}">${s.speaker}</span>` +
    `<span>${dispText(s.text)}</span></div>`).join("") +
    (tail ? `<div class="tail">${tail.replace(/</g, "&lt;")}</div>` : "")) ||
    '<div class="placeholder">…</div>';
  rows = [...box.querySelectorAll(".seg")];
  rows.forEach((r, i) => {
    r.onclick = () => {
      const a = $("audio");
      if ($("player-box").style.display !== "none" && isFinite(segs[i].start)) {
        a.currentTime = segs[i].start + 0.01;
        a.play().catch(() => {});
      }
    };
  });
  applyFilter();
  activeIdx = -1;
  if ($("autoscroll").checked) box.scrollTop = box.scrollHeight;
}

function applyFilter() {
  const q = $("search").value.trim().toLowerCase();
  rows.forEach((r, i) => {
    const s = segs[i];
    r.classList.toggle("hidden",
      hiddenSpk.has(s.speaker) || (q && !s.text.toLowerCase().includes(q)));
  });
}
$("search").oninput = applyFilter;

$("audio").addEventListener("timeupdate", () => {
  const t = $("audio").currentTime;
  let idx = -1;
  for (let i = 0; i < segs.length; i++) {
    if (segs[i].start <= t && t < segs[i].end) { idx = i; break; }
    if (segs[i].start > t) break;
  }
  if (idx === activeIdx) return;
  if (activeIdx >= 0 && rows[activeIdx]) rows[activeIdx].classList.remove("active");
  activeIdx = idx;
  if (idx >= 0 && rows[idx]) {
    rows[idx].classList.add("active");
    if ($("autoscroll").checked && !rows[idx].classList.contains("hidden")) {
      rows[idx].scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
});

function offerDownloads(finalSegs) {
  const ts = (t) => {
    const h = String(Math.floor(t / 3600)).padStart(2, "0");
    const m = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
    const s = String(Math.floor(t % 60)).padStart(2, "0");
    return `${h}:${m}:${s},${String(Math.round((t % 1) * 1000)).padStart(3, "0")}`;
  };
  const srt = finalSegs.map((s, i) =>
    `${i + 1}\n${ts(s.start)} --> ${ts(s.end)}\n[${s.speaker}] ${s.text}\n`).join("\n");
  $("dl-srt").href = URL.createObjectURL(new Blob([srt], { type: "text/plain" }));
  $("dl-srt").style.display = "inline";
  const js = JSON.stringify(finalSegs.map((s) =>
    ({ start: s.start, end: s.end, speaker: s.speaker, text: s.text })), null, 1);
  $("dl-json").href = URL.createObjectURL(new Blob([js], { type: "application/json" }));
  $("dl-json").style.display = "inline";
}

function resetView(placeholder) {
  segs = []; rows = []; hiddenSpk = new Set(); activeIdx = -1;
  dispCache.clear();
  $("legend").innerHTML = "";
  $("stats").textContent = "";
  $("dl-srt").style.display = "none";
  $("dl-json").style.display = "none";
  $("bar").style.display = "none";
  $("transcript").innerHTML = `<div class="placeholder">${placeholder}</div>`;
}

/* ============================ model / engine ============================= */
// ── Inference runs in a Web Worker (moss-worker.js) ──────────────────────
// The WASM module + its ggml pthread pool live in the worker, so the page stays
// responsive during compute, and the decoder streams each token back live.
let worker = null, modelReady = false, modelPromise = null;
let onWindowResolve = null;      // resolves the in-flight transcribeWindow()
let tailProvisional = "";        // live partial transcript for the current window

function setModelState(cls, msg) {
  $("model-state").className = cls;
  $("model-msg").textContent = msg;
}

// Create the worker, download the GGUF (streaming, progress) and init the model.
// Guarded so it runs at most once; safe to await from multiple entry points.
function ensureModel() {
  if (modelReady) return Promise.resolve();
  if (modelPromise) return modelPromise;
  modelPromise = new Promise((resolve, reject) => {
    $("btn-load").disabled = true;
    setModelState("loading", "Loading WASM runtime…");
    $("dl-bars").innerHTML =
      `<div>Q4_K GGUF<progress id="pg-model" max="100" value="0"></progress></div>`;
    worker = new Worker(`./moss-worker.js?wasm=${WASM_VARIANT}`);
    worker.onerror = () => {
      setModelState("", "Worker failed to start."); modelPromise = null;
      reject(new Error("worker error"));
    };
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "dl") {
        const pg = $("pg-model"); if (pg && m.total) pg.value = (100 * m.got) / m.total;
        const mb = (m.got / 1048576) | 0;
        setModelState("loading",
          `Downloading model… ${mb}${m.total ? " / " + ((m.total / 1048576) | 0) : ""} MB (one-time, cached)`);
        dlState = { done: m.got, total: m.total }; beat("downloading model");
      } else if (m.type === "status") {
        setModelState("loading", m.text);
      } else if (m.type === "ready") {
        modelReady = true; dlState = null; $("dl-bars").innerHTML = "";
        diarize = !!m.diarize;
        setModelState("ready", `Model ready · Q4_K · WASM · ${m.threads} threads` +
          (diarize ? " · CAM++ diarization" : ""));
        resolve();
      } else if (m.type === "moss_token") {
        tailProvisional = s2tw(m.text);      // live partial (Traditional)
        tokenCount++;                        // one decoder token per stream event
        scheduleTailRender();
      } else if (m.type === "window") {
        if (m.heapMB) console.log(`[mem] wasm heap high-water: ${m.heapMB} MB`);
        if (onWindowResolve) { const r = onWindowResolve; onWindowResolve = null; r(m); }
      } else if (m.type === "error") {
        setModelState("", "Error: " + m.error);
        if (onWindowResolve) { const r = onWindowResolve; onWindowResolve = null; r(null); }
        else { modelPromise = null; reject(new Error(m.error)); }
      }
    };
    worker.postMessage({ type: "init", ggufUrl: GGUF_URL, spkUrl: SPK_GGUF_URL });
  }).catch((e) => { modelPromise = null; throw e; });
  return modelPromise;
}

$("btn-load").onclick = () => { ensureModel().catch(() => {}); };

let tailRafPending = false;
function scheduleTailRender() {
  if (tailRafPending) return;
  tailRafPending = true;
  requestAnimationFrame(() => { tailRafPending = false; renderTranscript(tailProvisional); });
}

// Map the worker's window-local segments (with CAM++ embeddings) to absolute
// time + Traditional/ITN'd text, keeping the embedding for cross-window linking.
function mapWindowSegs(m, base) {
  return (m.segs || []).map((s) => ({
    start: base + s.start,
    rawEnd: s.rawEnd != null ? base + s.rawEnd : null,
    spk: s.spk || "",
    text: s2tw(s.text),          // s2tw() also applies number ITN
    emb: s.emb || null,          // 192-d CAM++ embedding (or null if too short)
  }));
}

/* ---- one window: send PCM to the worker; tokens stream via onmessage -------- */
// Resolves with the window's segments (absolute time, ITN'd text, + embeddings).
function transcribeWindow(wav, base, durS) {
  return new Promise((resolve) => {
    const nTok = Math.floor((wav.length - 1) / 1280) + 1;
    tailProvisional = "";
    onWindowResolve = (m) => {
      tailProvisional = "";
      resolve(m ? mapWindowSegs(m, base) : []);
    };
    const w = wav.slice();                 // copy just this window's samples
    worker.postMessage(
      { type: "run", pcm: w.buffer, nTok, base, durS, prompt: PROMPT },
      [w.buffer]);
  });
}

/* ---- cross-window speaker linking (ported from the ORT demo pipeline.js) ----
 * Core-segment average-linkage AHC @ cosine 0.45; segments < 3 s snap to the
 * nearest cluster centroid; no-embedding segments inherit the previous label.
 * Gives globally-consistent S01/S02… across windows via CAM++ embeddings. */
function linkSpeakers(segs, threshold = 0.45, minCoreDur = 3.0) {
  const withEmb = [];
  segs.forEach((s, i) => { if (s.emb) withEmb.push(i); });
  const core = withEmb.filter((i) => segs[i].end - segs[i].start >= minCoreDur);
  const labOf = new Map();
  if (core.length >= 2) {
    const n = core.length;
    const D = new Float64Array(n * n);
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      let dot = 0; const ea = segs[core[a]].emb, eb = segs[core[b]].emb;
      for (let k = 0; k < ea.length; k++) dot += ea[k] * eb[k];
      D[a * n + b] = D[b * n + a] = 1 - dot;
    }
    const clusters = core.map((_, i) => [i]);
    const active = new Set(clusters.map((_, i) => i));
    for (;;) {
      let bi = -1, bj = -1, bd = threshold;
      for (const i of active) for (const j of active) {
        if (j <= i) continue;
        const d = D[i * n + j];
        if (d < bd) { bd = d; bi = i; bj = j; }
      }
      if (bi < 0) break;
      const ni = clusters[bi].length, nj = clusters[bj].length;
      for (const k of active) {
        if (k === bi || k === bj) continue;
        const d = (ni * D[bi * n + k] + nj * D[bj * n + k]) / (ni + nj);
        D[bi * n + k] = D[k * n + bi] = d;
      }
      clusters[bi] = clusters[bi].concat(clusters[bj]);
      active.delete(bj);
    }
    const cents = [];
    for (const ci of active) {
      const members = clusters[ci];
      const dim = segs[core[0]].emb.length;
      const c = new Float64Array(dim);
      for (const mm of members) { const e = segs[core[mm]].emb; for (let k = 0; k < dim; k++) c[k] += e[k]; }
      let nrm = 0; for (let k = 0; k < dim; k++) nrm += c[k] * c[k];
      nrm = Math.sqrt(nrm) || 1;
      for (let k = 0; k < dim; k++) c[k] /= nrm;
      cents.push(c);
      for (const mm of members) labOf.set(core[mm], cents.length - 1);
    }
    for (const i of withEmb) {
      if (labOf.has(i)) continue;
      let best = 0, bestDot = -2;
      for (let c = 0; c < cents.length; c++) {
        let dot = 0; const e = segs[i].emb;
        for (let k = 0; k < e.length; k++) dot += cents[c][k] * e[k];
        if (dot > bestDot) { bestDot = dot; best = c; }
      }
      labOf.set(i, best);
    }
  } else if (withEmb.length) {
    for (const i of withEmb) labOf.set(i, 0);
  }
  const canon = new Map();
  const out = []; let prev = "S01";
  segs.forEach((s, i) => {
    let spk; const l = labOf.get(i);
    if (l === undefined) spk = prev;
    else { if (!canon.has(l)) canon.set(l, `S${String(canon.size + 1).padStart(2, "0")}`); spk = canon.get(l); }
    prev = spk;
    out.push({ start: s.start, end: s.end, speaker: spk, text: s.text });
  });
  return out;
}

// Fill each segment's display `end`: prefer the model's own end timestamp,
// else the next segment's start, else a small default. Also carry the last
// seen [Sxx] speaker forward when a segment omits its own tag.
function normalizeSegs(list) {
  let prev = "S01";
  for (let i = 0; i < list.length; i++) {
    const s = list[i];
    s.speaker = s.spk ? s.spk : prev;
    prev = s.speaker;
    if (s.rawEnd != null && s.rawEnd > s.start) s.end = s.rawEnd;
    else if (i + 1 < list.length) s.end = Math.max(list[i + 1].start, s.start + 0.1);
    else s.end = s.start + 3;
  }
}

/* ============================ heartbeat ================================== */
let lastBeat = 0, beatWhat = "", runT0 = 0, hbTimer = null;
let dlState = null;
function beat(what) { lastBeat = Date.now(); beatWhat = what; }
function hbStart() {
  runT0 = Date.now();
  beat("starting");
  $("heartbeat").style.display = "flex";
  hbTimer = setInterval(() => {
    const idle = (Date.now() - lastBeat) / 1000;
    const el = (Date.now() - runT0) / 1000;
    const dot = $("hb-dot");
    if (dlState) {
      dot.className = "";
      const mb = (dlState.done / 1048576).toFixed(0);
      const tot = dlState.total ? ` / ${(dlState.total / 1048576).toFixed(0)} MB` : " MB";
      $("hb-text").textContent =
        `${fmt(el)} elapsed · downloading model ${mb}${tot} (one-time, cached)`;
      return;
    }
    let note;
    if (idle < 8) { dot.className = ""; note = "model active"; }
    else if (idle < 90) {
      dot.className = "warn";
      note = `computing a window — ${idle.toFixed(0)}s since last output (normal)`;
    } else {
      dot.className = "bad";
      note = `no progress for ${idle.toFixed(0)}s — likely stalled; use ✕ Stop and retry`;
    }
    $("hb-text").textContent = `${fmt(el)} elapsed · ${beatWhat} · ${note}`;
  }, 1000);
}
function hbStop() { clearInterval(hbTimer); $("heartbeat").style.display = "none"; }

/* ============================ live transcription ========================= */
async function transcribe(source) {
  // Caller already acquireBusy()'d synchronously; this owns releasing it.
  // `source` is a window source ({durS, getWindow, close}) — audio is decoded
  // PER WINDOW on demand, so an hour-long file never sits fully decoded in
  // memory (~460 MB/2 h saved). durS may be an estimate for compressed audio.
  aborted = false;
  $("btn-abort").style.display = "";
  const WINDOW_S = currentWindowS();   // snapshot the selector for this run
  let nWin = Math.max(1, Math.ceil(source.durS / WINDOW_S));
  const t0 = Date.now();
  hbStart();
  $("bar").style.display = "";

  const finish = () => {
    hbStop(); $("btn-abort").style.display = "none"; releaseBusy();
    try { source.close(); } catch {}
  };

  try {
    beat("loading model");
    $("status").textContent = "Preparing model…";
    await ensureModel();
  } catch (e) {
    $("status").textContent = "Model load failed: " + e.message;
    finish();
    return;
  }

  segs = [];
  diarSegs = [];
  tokenCount = 0;
  let processedS = 0;
  $("status").textContent =
    `${fmt(source.durS)} of audio → ~${nWin} × ${WINDOW_S}s window(s)`;

  // Unbounded: nWin is an estimate for compressed audio; we stop when the
  // source runs dry (short/empty window).
  for (let w = 0; ; w++) {
    if (aborted) break;
    beat(`decoding window ${w + 1}`);
    let piece;
    try {
      piece = await source.getWindow(w * WINDOW_S * SR, WINDOW_S * SR);
    } catch (e) {
      $("status").textContent = "Audio decode failed: " + e.message;
      break;
    }
    const durS = piece.length / SR;
    if (durS < 0.3) break;
    nWin = Math.max(nWin, w + 1);          // durS was an estimate for mp3 etc.
    beat(`window ${w + 1}/${nWin}`);
    $("status").textContent = `Window ${w + 1}/${nWin} · transcribing…`;
    await yieldToLoop(); // paint the previous window + let an abort click land

    let winSegs;
    try {
      winSegs = await transcribeWindow(piece, w * WINDOW_S, durS);
    } catch (e) {
      $("status").textContent = "Inference failed: " + e.message;
      finish();
      return;
    }
    processedS = w * WINDOW_S + durS;
    for (const s of winSegs) diarSegs.push(s);
    normalizeSegs(diarSegs);                 // fill end + window-local speaker
    // Re-link speakers globally across all windows so far (CAM++ AHC), or fall
    // back to the model's per-window [Sxx] tags when the speaker model is absent.
    segs = diarize ? linkSpeakers(diarSegs) : diarSegs;
    renderLegend();
    renderTranscript();

    const el = (Date.now() - t0) / 1000;
    const done = w + 1;
    const eta = (el / done) * (nWin - done);
    $("bar").firstElementChild.style.width = `${(100 * done) / nWin}%`;
    $("stats").textContent =
      `${done}/${nWin} windows · ${segs.length} segments · ` +
      `${new Set(segs.map((s) => s.speaker)).size} speakers` +
      (tokenCount ? ` · ${(tokenCount / el).toFixed(1)} tok/s` : "") +
      (done < nWin ? ` · ~${fmt(eta)} left` : "");
  }

  const el = (Date.now() - t0) / 1000;
  $("bar").firstElementChild.style.width = "100%";
  renderLegend();
  renderTranscript();
  $("status").textContent = aborted ? "Stopped — partial result kept." : "Done · all local.";
  $("stats").textContent =
    `${fmt(processedS)} audio · ${segs.length} segments · ` +
    `${new Set(segs.map((s) => s.speaker)).size} speakers · ` +
    `${fmt(el)} compute (${(processedS / el).toFixed(2)}× realtime` +
    (tokenCount ? `, ${(tokenCount / el).toFixed(1)} tok/s` : "") + `)`;
  if (segs.length) offerDownloads(segs);
  finish();
}

$("btn-abort").onclick = () => { aborted = true; };

/* ============================ examples ================================== */
// Run LIVE through the WASM path (no precomputed JSON). Fetches the example
// audio and pushes it through the exact same decode + transcribe pipeline.
document.querySelectorAll("[data-ex]").forEach((btn) => {
  btn.onclick = async () => {
    if (!acquireBusy()) {
      $("status").textContent =
        "Still working on the previous audio — click ✕ Stop first, or wait for it to finish.";
      return;
    }
    const stem = btn.dataset.ex;
    resetView(`載入範例 ${stem}…`);
    try {
      const url = `${EXAMPLES}${stem}.mp3`;
      $("audio").src = url;
      $("player-box").style.display = "";
      const blob = await (await fetch(url)).blob();
      const file = new File([blob], `${stem}.mp3`, { type: blob.type || "audio/mpeg" });
      await transcribe(await makeAudioSource(file, (msg) => { $("status").textContent = msg; }));
    } catch (err) {
      $("status").textContent = "無法載入範例：" + err.message;
      releaseBusy();
    }
  };
});

/* ============================ audio input =============================== */
/* Bounded-memory decode → 16 kHz mono. Copied from space_local/app.js:
 * WAV/MP3 are sliced and decoded piecewise; other formats use a one-shot
 * browser decode while short enough, otherwise a lazily-loaded self-hosted
 * ffmpeg.wasm (./ffmpeg/) — only reachable for exotic/long inputs. */
const CHUNK_S = 20;
const MP3_CHUNK_BYTES = 2097152;
const GENERIC_DECODE_MAX_S = 40 * 60;

let ffmpegPromise = null;
async function toBlobURL(url, mimeType) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split("/").pop()}: HTTP ${r.status}`);
  return URL.createObjectURL(new Blob([await r.arrayBuffer()], { type: mimeType }));
}
function getFFmpeg() {
  if (ffmpegPromise) return ffmpegPromise;
  ffmpegPromise = (async () => {
    const { FFmpeg } = await import("./ffmpeg/index.js");
    const ff = new FFmpeg();
    const base = new URL("./ffmpeg/", location.href).href;
    await ff.load({
      coreURL: await toBlobURL(base + "core/ffmpeg-core.js", "text/javascript"),
      wasmURL: await toBlobURL(base + "core/ffmpeg-core.wasm", "application/wasm"),
    });
    return ff;
  })().catch((e) => { ffmpegPromise = null; throw e; });
  return ffmpegPromise;
}
async function ffmpegTo16k(blob, onStatus) {
  let ff;
  try {
    onStatus && onStatus("Loading audio converter (~32 MB, one-time)…");
    ff = await getFFmpeg();
  } catch (e) {
    throw new Error("Couldn't load the audio converter: " + e.message);
  }
  const onProg = ({ progress }) => onStatus &&
    onStatus(`Converting audio… ${Math.min(100, Math.round((progress || 0) * 100))}%`);
  ff.on("progress", onProg);
  try {
    await ff.writeFile("in", new Uint8Array(await blob.arrayBuffer()));
    const code = await ff.exec(["-i", "in", "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "out.raw"]);
    if (code !== 0) throw new Error("ffmpeg exited with code " + code);
    const data = await ff.readFile("out.raw");
    const aligned = data.slice().buffer;
    return new Float32Array(aligned);
  } catch (e) {
    throw new Error("Audio conversion failed: " + e.message);
  } finally {
    ff.off("progress", onProg);
    ff.deleteFile("in").catch(() => {});
    ff.deleteFile("out.raw").catch(() => {});
  }
}

function probeDuration(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const el = new Audio();
    el.preload = "metadata";
    el.src = url;
    el.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(el.duration); };
    el.onerror = () => { URL.revokeObjectURL(url); reject(new Error("could not read audio metadata")); };
  });
}

async function resampleTo16kMono(decoded) {
  const oac = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * SR)), SR);
  const src = oac.createBufferSource();
  src.buffer = decoded;
  src.connect(oac.destination);
  src.start();
  return (await oac.startRendering()).getChannelData(0);
}

async function decodeChunked(estimatedTotalS, chunkCount, getChunkBlob) {
  let out = new Float32Array(Math.ceil(estimatedTotalS * SR) + SR);
  const ac = new AudioContext();
  let pos = 0;
  for (let i = 0; i < chunkCount; i++) {
    const decoded = await ac.decodeAudioData(await (await getChunkBlob(i)).arrayBuffer());
    const resampled = await resampleTo16kMono(decoded);
    if (pos + resampled.length > out.length) {
      const grown = new Float32Array(Math.max(out.length * 2, pos + resampled.length));
      grown.set(out.subarray(0, pos));
      out = grown;
    }
    out.set(resampled, pos);
    pos += resampled.length;
  }
  ac.close();
  return pos < out.length ? out.subarray(0, pos) : out;
}

async function parseWavHeader(file) {
  const head = new DataView(await file.slice(0, 65536).arrayBuffer());
  if (head.byteLength < 44 || head.getUint32(0, false) !== 0x52494646 ||
      head.getUint32(8, false) !== 0x57415645) return null;
  let off = 12, fmt = null, dataOff = -1, dataLen = 0;
  while (off + 8 <= head.byteLength) {
    const id = String.fromCharCode(
      head.getUint8(off), head.getUint8(off + 1), head.getUint8(off + 2), head.getUint8(off + 3));
    const len = head.getUint32(off + 4, true);
    if (id === "fmt ") {
      const audioFormat = head.getUint16(off + 8, true);
      if (audioFormat !== 1 && audioFormat !== 3) return null;
      fmt = {
        audioFormat,
        channels: head.getUint16(off + 10, true),
        sampleRate: head.getUint32(off + 12, true),
        bitsPerSample: head.getUint16(off + 22, true),
      };
    } else if (id === "data") {
      dataOff = off + 8;
      const remaining = file.size - dataOff;
      dataLen = (len > 0 && len <= remaining) ? len : remaining;
      break;
    }
    off += 8 + len + (len % 2);
  }
  if (!fmt || dataOff < 0 || dataLen <= 0) return null;
  return { ...fmt, dataOff, dataLen };
}

function wavHeaderBytes({ audioFormat, channels, sampleRate, bitsPerSample }, dataLen) {
  const buf = new ArrayBuffer(44);
  const v = new DataView(buf);
  const blockAlign = channels * (bitsPerSample / 8);
  v.setUint32(0, 0x52494646, false); v.setUint32(4, 36 + dataLen, true);
  v.setUint32(8, 0x57415645, false); v.setUint32(12, 0x666d7420, false);
  v.setUint32(16, 16, true); v.setUint16(20, audioFormat, true); v.setUint16(22, channels, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * blockAlign, true);
  v.setUint16(32, blockAlign, true); v.setUint16(34, bitsPerSample, true);
  v.setUint32(36, 0x64617461, false); v.setUint32(40, dataLen, true);
  return new Uint8Array(buf);
}

async function decodeWavChunked(file, header) {
  const bytesPerFrame = header.channels * (header.bitsPerSample / 8);
  const totalFrames = Math.floor(header.dataLen / bytesPerFrame);
  const totalS = totalFrames / header.sampleRate;
  const framesPerChunk = Math.max(1, Math.floor(CHUNK_S * header.sampleRate));
  const chunkCount = Math.ceil(totalFrames / framesPerChunk);
  return decodeChunked(totalS, chunkCount, async (i) => {
    const frame0 = i * framesPerChunk;
    const nFrames = Math.min(framesPerChunk, totalFrames - frame0);
    const byteStart = header.dataOff + frame0 * bytesPerFrame;
    const byteLen = nFrames * bytesPerFrame;
    const pcm = await file.slice(byteStart, byteStart + byteLen).arrayBuffer();
    return new Blob([wavHeaderBytes(header, byteLen), pcm], { type: "audio/wav" });
  });
}

async function decodeMp3Chunked(file, totalS) {
  const chunkCount = Math.ceil(file.size / MP3_CHUNK_BYTES);
  return decodeChunked(totalS, chunkCount, (i) =>
    Promise.resolve(file.slice(i * MP3_CHUNK_BYTES, Math.min((i + 1) * MP3_CHUNK_BYTES, file.size))));
}

async function blobTo16k(blob, maxS = 0, onStatus = null) {
  const durationS = await probeDuration(blob).catch(() => 0);
  const isMp3 = /mpeg|mp3/i.test(blob.type) || /\.mp3$/i.test(blob.name || "");
  const wavHeader = !isMp3 ? await parseWavHeader(blob).catch(() => null) : null;

  let wav;
  try {
    if (wavHeader) wav = await decodeWavChunked(blob, wavHeader);
    else if (isMp3 && durationS > 0) wav = await decodeMp3Chunked(blob, durationS);
  } catch { /* fall through */ }

  if (!wav && durationS <= GENERIC_DECODE_MAX_S) {
    try {
      const arr = await blob.arrayBuffer();
      const ac = new AudioContext();
      const decoded = await ac.decodeAudioData(arr);
      ac.close();
      wav = await resampleTo16kMono(decoded);
    } catch { /* undecodable — ffmpeg fallback below */ }
  }
  if (!wav) wav = await ffmpegTo16k(blob, onStatus);

  return maxS > 0 ? wav.subarray(0, Math.min(wav.length, Math.ceil(maxS * SR))) : wav;
}

/* ================= on-demand window sources ==============================
 * transcribe() pulls audio one window at a time, so long recordings never sit
 * fully decoded in memory (2 h @16 kHz f32 ≈ 460 MB). WAV gets random access
 * via byte slicing; MP3 gets a sequential rolling decoder; anything else falls
 * back to a one-shot full decode. */

// Trivial source over an already-decoded array (mic, fallback decode).
function arraySource(wav) {
  return {
    durS: wav.length / SR,
    async getWindow(s0, n) {
      const a = Math.min(s0, wav.length);
      return wav.subarray(a, Math.min(a + n, wav.length));
    },
    close() {},
  };
}

// WAV: exact random access — slice the PCM byte range of the requested window,
// prepend a synthesized header, decode + resample just that window.
function wavWindowSource(file, header) {
  const bytesPerFrame = header.channels * (header.bitsPerSample / 8);
  const totalFrames = Math.floor(header.dataLen / bytesPerFrame);
  const srcRate = header.sampleRate;
  let ac = null;
  return {
    durS: totalFrames / srcRate,
    async getWindow(s0, n) {
      const f0 = Math.floor((s0 / SR) * srcRate);
      if (f0 >= totalFrames) return new Float32Array(0);
      const nf = Math.min(Math.ceil((n / SR) * srcRate), totalFrames - f0);
      const byteStart = header.dataOff + f0 * bytesPerFrame;
      const pcm = await file.slice(byteStart, byteStart + nf * bytesPerFrame).arrayBuffer();
      const chunk = new Blob([wavHeaderBytes(header, pcm.byteLength), pcm], { type: "audio/wav" });
      if (!ac) ac = new AudioContext();
      const decoded = await ac.decodeAudioData(await chunk.arrayBuffer());
      const out = await resampleTo16kMono(decoded);
      return out.length > n ? out.subarray(0, n) : out;
    },
    close() { if (ac) { ac.close(); ac = null; } },
  };
}

// MP3: sequential streaming — decode fixed-size byte chunks forward into a
// small rolling buffer, dropping samples already consumed by earlier windows.
// (Windows are requested in order by transcribe().)
function mp3StreamSource(file, estDurS) {
  const chunkCount = Math.ceil(file.size / MP3_CHUNK_BYTES);
  let nextChunk = 0, bufStart = 0, eof = false;
  let buf = new Float32Array(0);
  let ac = null;
  async function fillTo(endSample) {
    while (!eof && bufStart + buf.length < endSample) {
      if (nextChunk >= chunkCount) { eof = true; break; }
      const part = file.slice(nextChunk * MP3_CHUNK_BYTES,
                              Math.min((nextChunk + 1) * MP3_CHUNK_BYTES, file.size));
      nextChunk++;
      if (!ac) ac = new AudioContext();
      let decoded;
      try { decoded = await ac.decodeAudioData(await part.arrayBuffer()); }
      catch { continue; }               // skip an undecodable boundary chunk
      const res = await resampleTo16kMono(decoded);
      const grown = new Float32Array(buf.length + res.length);
      grown.set(buf); grown.set(res, buf.length);
      buf = grown;
    }
  }
  return {
    durS: estDurS,
    async getWindow(s0, n) {
      if (s0 > bufStart) {              // drop consumed samples
        const drop = Math.min(s0 - bufStart, buf.length);
        buf = buf.slice(drop); bufStart += drop;
      }
      await fillTo(s0 + n);
      const a = Math.max(0, s0 - bufStart);
      const b = Math.min(buf.length, a + n);
      return b > a ? buf.subarray(a, b) : new Float32Array(0);
    },
    close() { if (ac) { ac.close(); ac = null; } buf = new Float32Array(0); },
  };
}

// Pick the best source for a blob.
async function makeAudioSource(blob, onStatus) {
  const durationS = await probeDuration(blob).catch(() => 0);
  const isMp3 = /mpeg|mp3/i.test(blob.type) || /\.mp3$/i.test(blob.name || "");
  const wavHeader = !isMp3 ? await parseWavHeader(blob).catch(() => null) : null;
  if (wavHeader) return wavWindowSource(blob, wavHeader);
  if (isMp3 && durationS > 0) return mp3StreamSource(blob, durationS);
  // m4a/other: no reliable random access — one-shot decode (ffmpeg fallback inside).
  return arraySource(await blobTo16k(blob, 0, onStatus));
}

/* ---- drop zone + file input -------------------------------------------- */
const drop = $("drop");
drop.onclick = () => { if (!drop.classList.contains("disabled")) $("file-in").click(); };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("hover"); };
drop.ondragleave = () => drop.classList.remove("hover");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
};
$("file-in").onchange = (e) => {
  const f = e.target.files[0];
  if (f) handleFile(f);
  e.target.value = "";
};

async function handleFile(f) {
  if (!acquireBusy()) {
    $("status").textContent =
      "Still working on the previous audio — click ✕ Stop first, or wait for it to finish.";
    return;
  }
  resetView(`解析 ${f.name}…`);
  $("status").textContent = `Decoding ${f.name}…`;
  try {
    $("audio").src = URL.createObjectURL(f);
    $("player-box").style.display = "";
    await transcribe(await makeAudioSource(f, (msg) => { $("status").textContent = msg; }));
  } catch (err) {
    $("status").textContent = "Could not decode this file: " + err.message;
    releaseBusy();
  }
}

/* ---- mic recording ----------------------------------------------------- */
let recorder = null, recChunks = [];
$("btn-rec").onclick = async () => {
  const btn = $("btn-rec");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  if (!acquireBusy()) {
    $("status").textContent =
      "Still working on the previous audio — click ✕ Stop first, or wait for it to finish.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Record mic";
      btn.classList.remove("rec-on");
      try {
        const blob = new Blob(recChunks);
        $("audio").src = URL.createObjectURL(blob);
        $("player-box").style.display = "";
        resetView("處理錄音…");
        await transcribe(arraySource(await blobTo16k(blob, MIC_MAX_S)));
      } catch (err) {
        $("status").textContent = "Could not process the recording: " + err.message;
        releaseBusy();
      }
    };
    recorder.start();
    btn.textContent = "■ Stop & transcribe";
    btn.classList.add("rec-on");
    $("status").textContent = `Recording… (max ${MIC_MAX_S / 60} min)`;
    setTimeout(() => { if (recorder.state === "recording") recorder.stop(); }, MIC_MAX_S * 1000);
  } catch (e) {
    $("status").textContent = "Mic error: " + e.message;
    releaseBusy();
  }
};

/* ============================ boot / isolation ========================== */
// The multi-threaded WASM build needs SharedArrayBuffer → cross-origin
// isolation. Top-level pages can gain it via the coi-sw.js service worker
// (one reload). Inside HuggingFace's iframe that never works, so offer an
// "open in a new tab" link instead of reload-looping. (Preserved verbatim
// from the previous wasm-examples/moss/index.html boot logic.)
const inIframe = (() => { try { return window.self !== window.top; } catch { return true; } })();
function reloadCount() { try { return +(sessionStorage.getItem("coiReloads") || 0); } catch { return 99; } }
function markReload() { try { sessionStorage.setItem("coiReloads", "1"); } catch {} }

async function init() {
  // Mel is computed in the worker (C++ WhisperMelExtractor), so nothing to
  // preload on the main thread.
  if (IS_IOS) {
    const wr = $("win-row"); if (wr) wr.style.display = "none"; // fixed 28 s on iOS
  }
  $("input-note").textContent = "CPU · multi-threaded WASM · live token streaming";
  $("btn-load").disabled = false;
  setControlsEnabled(true);
  setModelState("", "Not loaded — starts automatically when needed");
  $("status").textContent = "Pick an example or provide audio.";
}

function showNewTabPrompt() {
  const url = location.href;
  $("status").innerHTML =
    "This demo needs a secure (cross-origin-isolated) context for its " +
    "multi-threaded engine, which the embedded preview can't provide.<br>" +
    `<a id="newtab" href="${url}" target="_blank" rel="noopener">▶ Open in a new tab to run</a>`;
}

async function boot() {
  if (!self.crossOriginIsolated && !inIframe && "serviceWorker" in navigator
      && reloadCount() < 1) {
    try {
      await navigator.serviceWorker.register("coi-sw.js");
      await navigator.serviceWorker.ready;
      if (!navigator.serviceWorker.controller) { markReload(); location.reload(); return; }
    } catch {}
  }
  if (self.crossOriginIsolated) {
    init();
  } else {
    showNewTabPrompt();
  }
}
boot();

// Show the deployed HF Space commit (X-Repo-Commit header) so testers can see
// exactly which build is live. Same-origin HEAD — all headers readable.
(async () => {
  const el = $("build");
  if (!el) return;
  try {
    const r = await fetch("./index.html", { method: "HEAD", cache: "no-store" });
    const c = r.headers.get("X-Repo-Commit");
    el.textContent = c ? c.slice(0, 8) : "(local dev)";
    if (c) el.title = "deployed Space commit " + c;
  } catch { el.textContent = "(local dev)"; }
})();
