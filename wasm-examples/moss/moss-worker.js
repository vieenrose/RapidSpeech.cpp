/* moss-worker.js — runs the RapidSpeech MossTD + CAM++ WASM off the main thread.
 *
 * The WASM module and its ggml pthread pool live in this worker, so the page
 * stays responsive during compute and the decoder streams each token back live
 * (emitted from C++ via postMessage as {type:"moss_token"}).
 *
 * Two models are loaded: MOSS-Transcribe-Diarize (transcription + per-window
 * [Sxx] tags) and CAM++ (192-d speaker embedding). After each window is
 * transcribed, every segment's audio slice is embedded so the page can link
 * speakers ACROSS windows (AHC clustering), matching the ORT demo.
 *
 * Nested-pthread handshake: emscripten spawns pthread workers as
 *   new Worker(_scriptName, {name:"em-pthread"})
 * and _scriptName resolves to THIS file's URL. So each ggml pthread worker also
 * loads moss-worker.js. The module's own tail does `isPthread && Module()` on
 * importScripts, which claims the pthread and installs its pthread-protocol
 * onmessage. We must NOT run our own init/onmessage in that role, or we clobber
 * the handshake and the pool never comes up (it recursively re-spawns instead).
 */
// Runtime variant: "mt" (desktop, 4 GB max) or "ios" (1.5 GB max — iOS WebKit
// refuses large shared-memory reservations). Passed by the page as
// moss-worker.js?wasm=ios; pthread re-spawns reuse the same URL, so every
// pthread worker loads the same variant.
const WASM_VARIANT = (() => {
  const v = new URLSearchParams(self.location.search).get("wasm");
  return v === "ios" || v === "gpu" ? v : "mt";
})();
importScripts(`./rapidspeech-wasm-${WASM_VARIANT}.js`);

if (globalThis.name !== "em-pthread") {
  // ── main worker role ────────────────────────────────────────────────────
  let Module = null, transcribePcm = null;
  let speakerEmbed = null, spkDim = 0;
  // Default thread count. On hybrid CPUs (Intel P+E+LPE) more threads can be
  // SLOWER: ggml barriers every op on the slowest core, so 16 threads on a
  // 6P+8E+2LPE laptop runs at low-power-E-core pace. Overridable per run via
  // the page's ?threads= URL param (passed in the init message).
  let N_THREADS = Math.max(1, Math.min(self.navigator?.hardwareConcurrency || 4, 16));
  const SR = 16000;
  // raw [start(-end)][Sxx]text parser (embedding boundaries; s2tw/ITN done on page)
  const SEG_RE = /\[(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?\](?:\[(S\d+)\])?([^\[]*)/g;

  // Stream a GGUF from a URL and expose it to the module's FS, reporting
  // progress with `tag`. Preferred path: stream the download into CACHE
  // STORAGE (disk-backed) and mount the resulting Blob via WORKERFS — the
  // bytes never sit in JS RAM and live OUTSIDE the WASM heap; ggml reads them
  // lazily. This is what keeps iOS alive: accumulating ~700 MB of chunks in
  // RAM tripped Safari's memory watchdog (silent tab reload right as the
  // download finished). Bonus: revisits hit the cache and skip the download.
  // Returns the FS path to pass to the model init.
  let wfsCount = 0;
  function mountBlob(name, blob) {
    const WORKERFS = Module.FS.filesystems && Module.FS.filesystems.WORKERFS;
    if (!WORKERFS) return null;
    const dir = `/wfs${wfsCount++}`;
    Module.FS.mkdir(dir);
    Module.FS.mount(WORKERFS, { blobs: [{ name, data: blob }] }, dir);
    return `${dir}/${name}`;
  }
  // The model is cached as ~64 MB PART entries plus a manifest, not one big
  // entry: WebKit's cache.put() buffers the ENTIRE body in RAM before
  // committing, so a single streamed ~700 MB put re-triggered iOS Safari's
  // memory watchdog (silent tab reload) right at download completion. Small
  // puts keep peak RAM at ~2 parts; after each put the part is re-read from
  // the cache so the composite Blob references DISK-backed parts (a Blob of
  // Blobs holds references — no copy).
  const CACHE_NAME = "moss-models-v3";
  const PART_BYTES = 64 * 1048576;
  async function fetchToFS(url, name, tag) {
    let cache = null;
    try {
      cache = await caches.open(CACHE_NAME);
      caches.delete("moss-models-v1").catch(() => {});
      caches.delete("moss-models-v2").catch(() => {});
    } catch {}
    // NOTE: Cache API keys ignore URL FRAGMENTS (#...) — every #part key
    // collapsed to the same entry and overwrote itself. Use query params.
    const ckey = (suffix) => url + (url.includes("?") ? "&" : "?") + "rspart=" + suffix;
    const manifestKey = ckey("manifest");
    // Revisit: rebuild the composite from the cached disk-backed parts.
    if (cache) {
      try {
        const man = await cache.match(manifestKey);
        if (man) {
          const { parts, size } = await man.json();
          const blobs = [];
          for (let i = 0; i < parts; i++) {
            const r = await cache.match(ckey(i));
            if (!r) { blobs.length = 0; break; }
            blobs.push(await r.blob());
          }
          if (parts > 0 && blobs.length === parts) {
            postMessage({ type: "dl", tag, got: size, total: size });
            const p = mountBlob(name, new Blob(blobs));
            if (p) return p;
          }
        }
      } catch {}
    }
    // Download: stream, committing each ~64 MB part as its own cache entry.
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(tag + " fetch " + resp.status);
    const total = +resp.headers.get("content-length") || 0;
    const reader = resp.body.getReader();
    const partBlobs = []; let chunks = [], sz = 0, got = 0, pi = 0;
    const flush = async () => {
      if (!sz) return;
      let part = new Blob(chunks); chunks = []; sz = 0;
      if (cache) {
        try {
          await cache.put(ckey(pi), new Response(part));
          const rt = await cache.match(ckey(pi));
          if (rt) part = await rt.blob();      // swap to the disk-backed copy
        } catch { cache = null; }              // quota/private mode: keep RAM part
      }
      partBlobs.push(part); pi++;
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); sz += value.length; got += value.length;
      if (sz >= PART_BYTES) await flush();
      postMessage({ type: "dl", tag, got, total });
    }
    await flush();
    if (cache) {
      try {
        await cache.put(manifestKey,
                        new Response(JSON.stringify({ parts: pi, size: got })));
      } catch {}
    }
    const blob = new Blob(partBlobs);
    const p = mountBlob(name, blob);
    if (p) return p;
    // Last resort: MEMFS copy (in-heap).
    const buf = new Uint8Array(await blob.arrayBuffer());
    Module.FS.writeFile(`/${name}`, buf);
    return `/${name}`;
  }

  // Embed one PCM slice -> Array(spkDim) L2-normalized, or null if too
  // short/failed. Never throws: a failed embed degrades that one segment to
  // "inherit previous speaker" — it must not error the whole window.
  function embedSlice(pcm, a, b) {
    try {
      if (!speakerEmbed || !spkDim) return null;
      a = Math.max(0, a | 0); b = Math.min(pcm.length, b | 0);
      if (b - a < 0.3 * SR) return null;               // ORT: skip < 0.3 s
      const seg = pcm.subarray(a, b);
      const sp = Module._malloc(seg.length * 4);
      if (!sp) return null;
      new Float32Array(Module.HEAPF32.buffer, sp, seg.length).set(seg);
      const ep = Module._malloc(spkDim * 4);
      if (!ep) { Module._free(sp); return null; }
      const rc = speakerEmbed(sp, seg.length, ep, spkDim);
      let emb = null;
      if (rc === 0) emb = Array.from(new Float32Array(Module.HEAPF32.buffer, ep, spkDim));
      Module._free(sp); Module._free(ep);
      return emb;
    } catch (err) {
      console.warn("embedSlice failed:", err);
      return null;
    }
  }

  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      if (msg.type === "init") {
        if (msg.threads > 0) N_THREADS = Math.max(1, Math.min(msg.threads | 0, 16));
        // Truth about which backend the scheduler actually got — the engine
        // logs it during init; "requested GPU" alone proves nothing.
        let gpuActive = false;
        const scanLog = (t) => { if (/WebGPU backend added to scheduler/.test(t)) gpuActive = true; };
        Module = await globalThis.RapidSpeechModule({
          locateFile: (p) =>
            (p.endsWith(".wasm") ? `./rapidspeech-wasm-${WASM_VARIANT}.wasm` : p),
          print: (t) => { scanLog(String(t)); console.log(t); },
          // Forward engine stderr to the page: iOS Safari has no console, and
          // ggml reports failures (e.g. buffer allocation) there before the
          // C++ returns an empty transcript "gracefully".
          printErr: (t) => { scanLog(String(t));
            try { postMessage({ type: "cxxlog", text: String(t) }); } catch {} },
        });
        // Engine env knobs from the page (?env_NAME=VALUE) — must be set
        // before model init; the C++ reads them via getenv.
        if (msg.env) {
          const setenv = Module.cwrap("rs_wasm_setenv", "number", ["string", "string"]);
          for (const [k, v] of Object.entries(msg.env)) {
            setenv(k, String(v));
            console.log(`[env] ${k}=${v}`);
          }
        }
        // 1. MOSS transcription model (~700 MB).
        const modelPath = await fetchToFS(msg.ggufUrl, "model.gguf", "asr");
        postMessage({ type: "status", text: "Loading transcription model…" });
        const init = Module.cwrap("rs_wasm_init_ex", "number",
                                  ["string", "number", "number"]);
        let rc = init(modelPath, 0, N_THREADS);
        if (rc instanceof Promise) rc = await rc;
        if (rc !== 0) throw new Error("model init returned " + rc);
        transcribePcm = Module.cwrap("rs_wasm_moss_transcribe_pcm", "string",
          ["number", "number", "number", "number", "string"]);

        // 2. CAM++ speaker-embedding model (~14 MB) for cross-window diarization.
        if (msg.spkUrl) {
          try {
            const spkPath = await fetchToFS(msg.spkUrl, "spk.gguf", "spk");
            postMessage({ type: "status", text: "Loading speaker model…" });
            const sInit = Module.cwrap("rs_wasm_speaker_init", "number", ["string", "number"]);
            let src = sInit(spkPath, N_THREADS);
            if (src instanceof Promise) src = await src;
            if (src === 0) {
              speakerEmbed = Module.cwrap("rs_wasm_speaker_embed", "number",
                ["number", "number", "number", "number"]);
              spkDim = Module.cwrap("rs_wasm_speaker_dim", "number", [])();
            } else {
              postMessage({ type: "status", text: "Speaker model failed; per-window tags only." });
            }
          } catch (se) {
            postMessage({ type: "status", text: "Speaker model unavailable; per-window tags only." });
          }
        }
        postMessage({ type: "ready", threads: N_THREADS, diarize: !!speakerEmbed,
                      gpu: gpuActive });
        return;
      }

      if (msg.type === "run") {
        if (!transcribePcm) { postMessage({ type: "error", error: "not ready" }); return; }
        const pcm = new Float32Array(msg.pcm);           // transferred buffer
        const ptr = Module._malloc(pcm.length * 4);
        if (!ptr) throw new Error(`malloc failed for ${pcm.length * 4} bytes (window pcm)`);
        new Float32Array(Module.HEAPF32.buffer, ptr, pcm.length).set(pcm);
        // While this runs, C++ posts {type:"moss_token", text} per token.
        let text = transcribePcm(ptr, pcm.length, msg.nTok, 1 /*stream*/, msg.prompt);
        if (text instanceof Promise) text = await text;
        Module._free(ptr);

        // Parse segments and embed each (for cross-window speaker linking).
        // Strip stray [hh:mm:ss] artifacts the QAT model occasionally emits —
        // they carry no info and would break the [start][Sxx] segment regex.
        text = text.replace(/\[\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\]/g, "");
        // ... and collapse doubled open-brackets ("[0.25][[S01]", a QAT-model
        // first-segment quirk) so the [start][Sxx] regex still matches.
        text = text.replace(/\[+\[/g, "[");
        const durS = msg.durS;
        const raw = []; let m;
        SEG_RE.lastIndex = 0;
        while ((m = SEG_RE.exec(text))) {
          const start = parseFloat(m[1]);
          if (!m[4] || !m[4].trim() || start > durS + 0.5) continue;
          raw.push({ start, rawEnd: m[2] !== undefined ? parseFloat(m[2]) : null,
                     spk: m[3] || "", text: m[4].trim() });
        }
        const segs = raw.map((s, i) => {
          const end = s.rawEnd != null && s.rawEnd > s.start ? s.rawEnd
                    : (i + 1 < raw.length ? raw[i + 1].start : Math.min(durS, s.start + 3));
          return {
            start: s.start, end, rawEnd: s.rawEnd, spk: s.spk, text: s.text,
            emb: embedSlice(pcm, s.start * SR, end * SR),
          };
        });
        // Real audio that produces NO text at all is almost always an engine
        // failure (encode/prefill alloc fail on memory-capped iOS) — C++ logs
        // the cause to stderr and returns "" rather than throwing. Tag the
        // window so the page can surface it instead of silently rendering
        // nothing.
        const failed = (!text || !text.trim()) && durS > 2;
        // WASM heap high-water (grows monotonically) — memory telemetry.
        // NOTE: HEAPU8, not HEAP8 — HEAP8 is not in EXPORTED_RUNTIME_METHODS;
        // reading .length of undefined here killed the whole window message
        // (transcript vanished at end of every window). Defensive anyway.
        const heapMB = Module.HEAPU8 ? (Module.HEAPU8.length / 1048576) | 0 : 0;
        postMessage({ type: "window", text, base: msg.base, durS,
                      wi: msg.wi, nw: msg.nw, segs, heapMB, failed });
        return;
      }
    } catch (err) {
      console.error("[worker] handler failed:", err && err.stack || err);
      postMessage({ type: "error", error: String(err && err.message || err) });
    }
  };
}
// else: this worker is an emscripten pthread; the module's tail already called
// RapidSpeechModule() and owns onmessage. Nothing more to do here.
