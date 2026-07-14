/* moss-worker.js — runs the RapidSpeech MossTD WASM off the main thread.
 *
 * The WASM module and its ggml pthread pool live in this worker, so the page
 * stays responsive during compute and the decoder streams each token back live
 * (emitted from C++ via postMessage as {type:"moss_token"}).
 *
 * Nested-pthread handshake: emscripten spawns pthread workers as
 *   new Worker(_scriptName, {name:"em-pthread"})
 * and _scriptName resolves to THIS file's URL. So each ggml pthread worker also
 * loads moss-worker.js. The module's own tail does `isPthread && Module()` on
 * importScripts, which claims the pthread and installs its pthread-protocol
 * onmessage. We must NOT run our own init/onmessage in that role, or we clobber
 * the handshake and the pool never comes up (it recursively re-spawns instead).
 */
importScripts("./rapidspeech-wasm-mt.js");

if (globalThis.name !== "em-pthread") {
  // ── main worker role ────────────────────────────────────────────────────
  let Module = null, transcribePcm = null;
  const N_THREADS = Math.max(1, Math.min(self.navigator?.hardwareConcurrency || 4, 16));

  self.onmessage = async (e) => {
    const msg = e.data;
    try {
      if (msg.type === "init") {
        Module = await globalThis.RapidSpeechModule({
          locateFile: (p) => (p.endsWith(".wasm") ? "./rapidspeech-wasm-mt.wasm" : p),
        });
        // Stream the ~700 MB GGUF into the module's MEMFS with progress.
        const resp = await fetch(msg.ggufUrl);
        if (!resp.ok) throw new Error("model fetch " + resp.status);
        const total = +resp.headers.get("content-length") || 0;
        const reader = resp.body.getReader();
        const chunks = []; let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value); got += value.length;
          postMessage({ type: "dl", got, total });
        }
        const gguf = new Uint8Array(got);
        let o = 0; for (const c of chunks) { gguf.set(c, o); o += c.length; }
        Module.FS.writeFile("/model.gguf", gguf);
        postMessage({ type: "status", text: "Loading model into WASM…" });
        const init = Module.cwrap("rs_wasm_init_ex", "number",
                                  ["string", "number", "number"]);
        let rc = init("/model.gguf", 0, N_THREADS);
        if (rc instanceof Promise) rc = await rc;
        if (rc !== 0) throw new Error("model init returned " + rc);
        transcribePcm = Module.cwrap("rs_wasm_moss_transcribe_pcm", "string",
          ["number", "number", "number", "number", "string"]);
        postMessage({ type: "ready", threads: N_THREADS });
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
        postMessage({ type: "window", text, base: msg.base, durS: msg.durS,
                      wi: msg.wi, nw: msg.nw });
        return;
      }
    } catch (err) {
      postMessage({ type: "error", error: String(err && err.message || err) });
    }
  };
}
// else: this worker is an emscripten pthread; the module's tail already called
// RapidSpeechModule() and owns onmessage. Nothing more to do here.
