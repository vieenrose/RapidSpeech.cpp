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
const WASM_VARIANT =
  new URLSearchParams(self.location.search).get("wasm") === "ios" ? "ios" : "mt";
importScripts(`./rapidspeech-wasm-${WASM_VARIANT}.js`);

if (globalThis.name !== "em-pthread") {
  // ── main worker role ────────────────────────────────────────────────────
  let Module = null, transcribePcm = null;
  let speakerEmbed = null, spkDim = 0;
  const N_THREADS = Math.max(1, Math.min(self.navigator?.hardwareConcurrency || 4, 16));
  const SR = 16000;
  // raw [start(-end)][Sxx]text parser (embedding boundaries; s2tw/ITN done on page)
  const SEG_RE = /\[(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?\](?:\[(S\d+)\])?([^\[]*)/g;

  // Stream a GGUF from a URL and expose it to the module's FS, reporting
  // progress with `tag`. Preferred path: mount the bytes as a Blob via WORKERFS
  // — the Blob lives OUTSIDE the WASM heap (the browser may even disk-back it)
  // and ggml reads it lazily, avoiding a ~700 MB in-heap MEMFS copy. Critical
  // on iOS, where heap + copy + weight buffers would exceed the memory cap.
  // Returns the FS path to pass to the model init.
  let wfsCount = 0;
  async function fetchToFS(url, name, tag) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(tag + " fetch " + resp.status);
    const total = +resp.headers.get("content-length") || 0;
    const reader = resp.body.getReader();
    const chunks = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      postMessage({ type: "dl", tag, got, total });
    }
    const WORKERFS = Module.FS.filesystems && Module.FS.filesystems.WORKERFS;
    if (WORKERFS) {
      const dir = `/wfs${wfsCount++}`;
      Module.FS.mkdir(dir);
      Module.FS.mount(WORKERFS, { blobs: [{ name, data: new Blob(chunks) }] }, dir);
      chunks.length = 0;
      return `${dir}/${name}`;
    }
    // Fallback: MEMFS copy (in-heap).
    const buf = new Uint8Array(got);
    let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    chunks.length = 0;
    Module.FS.writeFile(`/${name}`, buf);
    return `/${name}`;
  }

  // Embed one PCM slice -> Array(spkDim) L2-normalized, or null if too short/failed.
  function embedSlice(pcm, a, b) {
    if (!speakerEmbed || !spkDim) return null;
    a = Math.max(0, a | 0); b = Math.min(pcm.length, b | 0);
    if (b - a < 0.3 * SR) return null;                 // ORT: skip < 0.3 s
    const seg = pcm.subarray(a, b);
    const sp = Module._malloc(seg.length * 4);
    new Float32Array(Module.HEAPF32.buffer, sp, seg.length).set(seg);
    const ep = Module._malloc(spkDim * 4);
    const rc = speakerEmbed(sp, seg.length, ep, spkDim);
    let emb = null;
    if (rc === 0) emb = Array.from(new Float32Array(Module.HEAPF32.buffer, ep, spkDim));
    Module._free(sp); Module._free(ep);
    return emb;
  }

  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      if (msg.type === "init") {
        Module = await globalThis.RapidSpeechModule({
          locateFile: (p) =>
            (p.endsWith(".wasm") ? `./rapidspeech-wasm-${WASM_VARIANT}.wasm` : p),
        });
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
        postMessage({ type: "ready", threads: N_THREADS, diarize: !!speakerEmbed });
        return;
      }

      if (msg.type === "run") {
        if (!transcribePcm) { postMessage({ type: "error", error: "not ready" }); return; }
        const pcm = new Float32Array(msg.pcm);           // transferred buffer
        const ptr = Module._malloc(pcm.length * 4);
        new Float32Array(Module.HEAPF32.buffer, ptr, pcm.length).set(pcm);
        // While this runs, C++ posts {type:"moss_token", text} per token.
        let text = transcribePcm(ptr, pcm.length, msg.nTok, 1 /*stream*/, msg.prompt);
        if (text instanceof Promise) text = await text;
        Module._free(ptr);

        // Parse segments and embed each (for cross-window speaker linking).
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
            start: s.start, rawEnd: s.rawEnd, spk: s.spk, text: s.text,
            emb: embedSlice(pcm, s.start * SR, end * SR),
          };
        });
        postMessage({ type: "window", text, base: msg.base, durS,
                      wi: msg.wi, nw: msg.nw, segs,
                      // WASM heap high-water (grows monotonically) — memory telemetry
                      heapMB: (Module.HEAP8.length / 1048576) | 0 });
        return;
      }
    } catch (err) {
      postMessage({ type: "error", error: String(err && err.message || err) });
    }
  };
}
// else: this worker is an emscripten pthread; the module's tail already called
// RapidSpeechModule() and owns onmessage. Nothing more to do here.
