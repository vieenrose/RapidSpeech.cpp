/* Minimal COOP/COEP service worker: static Spaces cannot set response
 * headers, but SharedArrayBuffer (multi-threaded wasm) requires
 * cross-origin isolation. This SW re-serves every same-scope response with
 * the isolation headers; COEP:credentialless keeps cross-origin model
 * fetches (HF CDN) working without CORP headers on them.
 *
 * Model-weight caching (the CDN sends Cache-Control: no-store, so the
 * browser's normal HTTP cache can never keep these files) is handled
 * directly inside infer-worker.js via the Cache Storage API instead of
 * here: whether a Service Worker reliably intercepts fetches issued from a
 * dedicated Worker (rather than the page itself) is browser/timing
 * dependent, and every model fetch in this app is issued from the worker,
 * never from the page — routing the caching through this SW added
 * unreliable overhead with no benefit. The Cache Storage API itself is
 * available directly in the worker's own global scope, no SW needed. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  let url;
  try { url = new URL(req.url); } catch { return; }
  // Re-serve SAME-ORIGIN responses (the HTML document, the wasm glue + its
  // pthread worker script, the .wasm) with the isolation headers — the page
  // AND its workers must carry COEP to be cross-origin-isolated. These are
  // small, so re-streaming them is safe.
  // PASS THROUGH cross-origin requests untouched — notably the ~700 MB GGUF
  // from the HF CDN: re-streaming that large body through the SW fails on
  // Firefox ("Error in input stream"), and under a credentialless document a
  // plain cross-origin fetch already loads fine.
  if (url.origin !== self.location.origin) return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0 || res.type === "opaque") return res;
      const h = new Headers(res.headers);
      h.set("Cross-Origin-Embedder-Policy", "require-corp");
      h.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, {
        status: res.status, statusText: res.statusText, headers: h,
      });
    }).catch(() => fetch(req))
  );
});
