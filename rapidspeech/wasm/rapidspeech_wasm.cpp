/**
 * RapidSpeech WASM entry point — C API wrappers for Emscripten.
 *
 * Exports a minimal set of functions callable from JavaScript via ccall/cwrap.
 * Model data is loaded through Emscripten's virtual filesystem:
 *   1. JS fetches .gguf from a URL
 *   2. JS writes it to /model.gguf via FS.writeFile()
 *   3. JS calls _rs_wasm_init("/model.gguf", n_threads)
 */

#include "rapidspeech.h"

#include <emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Globals (single-context for browser demos) ──────────────
static rs_context_t *g_ctx = nullptr;
static char g_last_text[4096] = {0};

// ── Init / Free ─────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_init(const char *model_path, int n_threads) {
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads = n_threads;
  params.use_gpu = false; // WASM is CPU-only

  g_ctx = rs_init_from_file(params);
  if (!g_ctx) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_init_from_file failed: " + UTF8ToString($0),
                    $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_free(void) {
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }
  rs_clear_error();
}

// ── Audio input ─────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_audio(const float *pcm, int n_samples) {
  if (!g_ctx) return -1;
  return (int)rs_push_audio(g_ctx, pcm, n_samples);
}

// ── Inference ───────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_process(void) {
  if (!g_ctx) return -1;
  int ret = rs_process(g_ctx);
  if (ret > 0) {
    const char *text = rs_get_text_output(g_ctx);
    if (text) {
      int i = 0;
      while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
      g_last_text[i] = '\0';
    }
  }
  return ret;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_text(void) {
  return g_last_text;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_reset(void) {
  if (!g_ctx) return -1;
  g_last_text[0] = '\0';
  return (int)rs_reset(g_ctx);
}

// ── Metadata ────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_get_sample_rate(void) {
  if (!g_ctx) return 16000;
  rs_model_meta_t meta = rs_get_model_meta(g_ctx);
  return meta.audio_sample_rate;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_arch_name(void) {
  if (!g_ctx) return "unknown";
  static char name[64];
  rs_model_meta_t meta = rs_get_model_meta(g_ctx);
  int i = 0;
  while (meta.arch_name[i] && i < 63) { name[i] = meta.arch_name[i]; i++; }
  name[i] = '\0';
  return name;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_is_ready(void) {
  return g_ctx ? 1 : 0;
}

// ── Utility ─────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_version(void) {
  return rs_get_version();
}

#ifdef __cplusplus
}
#endif
