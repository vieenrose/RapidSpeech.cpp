/**
 * RapidSpeech WASM entry point — C API wrappers for Emscripten.
 *
 * Exports a minimal C-callable surface for both ASR and TTS.
 *
 * Model data is loaded through Emscripten's virtual filesystem:
 *   1. JS fetches .gguf from a URL
 *   2. JS writes it to /model.gguf via FS.writeFile()
 *   3. JS calls _rs_wasm_init("/model.gguf", task_type, n_threads)
 *
 * Task types (mirror rs_task_type_t):
 *   0 = ASR_OFFLINE   (default — matches the old single-arg init)
 *   1 = ASR_ONLINE
 *   2 = TTS_OFFLINE
 *   3 = TTS_ONLINE
 *   4 = E2E_SPEECH_LLM
 */

#include "rapidspeech.h"

#include "arch/keyword_loader.h"
#include "arch/moss_td.h"
#include "arch/sensevoice.h"
#include "core/rs_context.h"
#include "core/rs_kws.h"

#include <emscripten.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward-declare explicit arch registrations so LTO cannot strip them.
void rs_register_sensevoice();
void rs_register_moss_td();

#ifdef __cplusplus
extern "C" {
#endif

// ── Globals (single-context for browser demos) ──────────────
static rs_context_t *g_ctx = nullptr;
static char g_last_text[65536] = {0};  // 3072-token budget x ~3B CJK + tags

// Buffer for TTS streaming PCM — owned by C, lifetime tied to g_ctx.
// Each successful TTS chunk is copied here so JS can read it via
// rs_wasm_get_audio_ptr / rs_wasm_get_audio_len without managing a
// separate heap allocation per chunk.
static float *g_audio_buf = nullptr;
static int    g_audio_cap = 0;
static int    g_audio_len = 0;

static void audio_buf_free(void) {
  std::free(g_audio_buf);
  g_audio_buf = nullptr;
  g_audio_cap = 0;
  g_audio_len = 0;
}

static int audio_buf_set(const float *src, int n) {
  if (n <= 0) {
    g_audio_len = 0;
    return 0;
  }
  if (n > g_audio_cap) {
    float *p = (float *)std::realloc(g_audio_buf, sizeof(float) * (size_t)n);
    if (!p) return -1;
    g_audio_buf = p;
    g_audio_cap = n;
  }
  std::memcpy(g_audio_buf, src, sizeof(float) * (size_t)n);
  g_audio_len = n;
  return 0;
}

// ── Init / Free ─────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_init_ex(const char *model_path, int task_type, int n_threads) {
  static bool archs_registered = false;
  if (!archs_registered) {
    rs_register_sensevoice();
    rs_register_moss_td();
    archs_registered = true;
  }
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }
  audio_buf_free();

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads  = n_threads;
  // Ask for GPU; rs_context falls back to CPU if no backend is compiled in.
  // For WASM the GPU backend is WebGPU (when RS_WASM_WEBGPU=ON at build time).
  params.use_gpu    = true;
  params.task_type  = (rs_task_type_t)task_type;

  g_ctx = rs_init_from_file(params);
  if (!g_ctx) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_init_from_file failed: " + UTF8ToString($0), $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

// Backwards-compatible ASR init (defaults to RS_TASK_ASR_OFFLINE).
EMSCRIPTEN_KEEPALIVE
int rs_wasm_init(const char *model_path, int n_threads) {
  return rs_wasm_init_ex(model_path, (int)RS_TASK_ASR_OFFLINE, n_threads);
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_free(void) {
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }
  audio_buf_free();
  g_last_text[0] = '\0';
  rs_clear_error();
}

// ── MossTD: transcribe from externally-computed (JS) log-mel ─────────
// The browser computes an HF-exact Whisper log-mel (see pipeline.js) and passes
// it here as [n_mel, n_frames] bin-slow / frame-fast. This bypasses the C++
// WhisperMelExtractor. Returns the diarized transcription string (owned by C).
// n_audio_tokens: real-audio token count = (n_samples-1)/1280 + 1; drops the
// trailing silence tokens of a 30 s-padded window. Pass 0 to keep all.
EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_moss_transcribe_mel(const float *mel, int n_frames,
                                        int n_mel, int n_audio_tokens,
                                        const char *prompt) {
  g_last_text[0] = '\0';
  if (!g_ctx || !g_ctx->model || !mel || n_frames <= 0) return g_last_text;
  auto *m = dynamic_cast<MossTDModel *>(g_ctx->model.get());
  if (!m) return g_last_text;
  if (prompt && prompt[0]) m->SetUserInputPrompt(prompt);
  std::vector<float> melv(mel, mel + (size_t)n_frames * n_mel);
  auto st = m->CreateState();
  if (!m->EncodeMel(melv, n_frames, *st, g_ctx->sched, n_audio_tokens))
    return g_last_text;
  if (!m->Decode(*st, g_ctx->sched)) return g_last_text;
  std::string text = m->GetTranscription(*st);
  std::strncpy(g_last_text, text.c_str(), sizeof(g_last_text) - 1);
  g_last_text[sizeof(g_last_text) - 1] = '\0';
  return g_last_text;
}

// ── MossTD: transcribe from raw 16 kHz mono PCM (mel computed IN WASM) ───────
// Faster + off the JS main thread vs computing the Whisper log-mel in JS. When
// `stream` is nonzero, the decoder posts each partial transcript to the host
// worker as {type:"moss_token", text} — used for live token-by-token output.
// Returns the final diarized transcription (owned by C).
EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_moss_transcribe_pcm(const float *pcm, int n_samples,
                                        int n_audio_tokens, int stream,
                                        const char *prompt) {
  g_last_text[0] = '\0';
  if (!g_ctx || !g_ctx->model || !pcm || n_samples <= 0) return g_last_text;
  auto *m = dynamic_cast<MossTDModel *>(g_ctx->model.get());
  if (!m) return g_last_text;
  if (prompt && prompt[0]) m->SetUserInputPrompt(prompt);
  if (stream) {
    m->SetOnToken([](const std::string &partial) {
      // Post the live partial transcript to the host page. This runs on the
      // module's main thread (the Web Worker that loaded the module), so a
      // plain postMessage reaches the page.
      EM_ASM({
        if (typeof postMessage === "function")
          postMessage({ type: "moss_token", text: UTF8ToString($0) });
      }, partial.c_str());
    });
    // Phase progress (encoder chunk N/M, prefill start, decode start) so the
    // page can show liveness through the long silent phases of big windows.
    m->SetOnPhase([](const char *phase, int cur, int total) {
      EM_ASM({
        if (typeof postMessage === "function")
          postMessage({ type: "moss_phase", phase: UTF8ToString($0),
                        cur: $1, total: $2 });
      }, phase, cur, total);
    });
  } else {
    m->SetOnToken(nullptr);
    m->SetOnPhase(nullptr);
  }
  // For long (multi-chunk) windows the f32 KV cache would blow past the WASM
  // 4 GB budget (300 s ~= 3854 ctx tokens -> ~1 GB f32 KV, 3.7 GB peak). q8_0 KV
  // is near-lossless and cuts ~0.7 GB. Short clips stay f32 (fully lossless).
  // KV policy: the engine default is now f16 (great natively), but in WASM a
  // 180 s window at f16 KV measured 3743 MB heap high-water vs the 4096 MB
  // hard cap — 350 MB of headroom is an OOM waiting to happen. With the
  // v6-stream model (trained for a bounded 45 s audio-KV window) long windows
  // use monotonic KV EVICTION instead of q8: audio KV stays O(45 s) at full
  // f16 precision, heap stays flat, and sub-second timestamps survive.
  // Short windows get the engine default (f16, no eviction).
  // Eviction is the ENGINE default now (45 s; RS_AUDIO_KV_WINDOW=0 opts out),
  // so no per-call setenv here — which also stops clobbering the page's
  // ?env_RS_* debug knobs (rs_wasm_setenv).

  std::vector<float> pcmv(pcm, pcm + n_samples);
  auto st = m->CreateState();
  // Encode() runs the C++ WhisperMelExtractor (mel) + encoder graph. The encoder
  // is truncated to the real audio length (see MossTDModel::RunEncoder), so short
  // clips don't pay for 30 s of silence.
  if (!m->Encode(pcmv, *st, g_ctx->sched)) { m->SetOnToken(nullptr); m->SetOnPhase(nullptr); return g_last_text; }
  // Drop trailing silence tokens to the real-audio length.
  auto *mst = static_cast<MossTDState *>(st.get());
  if (n_audio_tokens > 0 && n_audio_tokens < mst->T_audio) {
    mst->audio_embeds.resize((size_t)mst->n_embd * n_audio_tokens);
    mst->T_audio = n_audio_tokens;
  }
  if (!m->Decode(*st, g_ctx->sched)) { m->SetOnToken(nullptr); m->SetOnPhase(nullptr); return g_last_text; }
  m->SetOnToken(nullptr);
  m->SetOnPhase(nullptr);
  std::string text = m->GetTranscription(*st);
  std::strncpy(g_last_text, text.c_str(), sizeof(g_last_text) - 1);
  g_last_text[sizeof(g_last_text) - 1] = '\0';
  return g_last_text;
}

// ── Audio / text input ──────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_audio(const float *pcm, int n_samples) {
  if (!g_ctx) return -1;
  return (int)rs_push_audio(g_ctx, pcm, n_samples);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_text(const char *text) {
  if (!g_ctx || !text) return -1;
  return (int)rs_push_text(g_ctx, text);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_reference_audio(const float *pcm, int n_samples, int sample_rate) {
  if (!g_ctx) return -1;
  return rs_push_reference_audio(g_ctx, pcm, n_samples, sample_rate);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_reference_text(const char *text) {
  if (!g_ctx || !text) return -1;
  return (int)rs_push_reference_text(g_ctx, text);
}

// ── Inference ───────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_process(void) {
  if (!g_ctx) return -1;
  int ret = rs_process(g_ctx);

  // ASR side-effect: capture latest text
  if (ret > 0) {
    const char *text = rs_get_text_output(g_ctx);
    if (text) {
      int i = 0;
      while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
      g_last_text[i] = '\0';
    }
  }

  // TTS side-effect: capture next PCM chunk if any
  float *chunk = nullptr;
  int n = rs_get_audio_output(g_ctx, &chunk);
  if (n > 0 && chunk) {
    audio_buf_set(chunk, n);
  } else {
    g_audio_len = 0;
  }

  return ret;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_text(void) {
  return g_last_text;
}

// Pointer to the most recent TTS PCM chunk (float32, host-endian).
// Valid until the next rs_wasm_process() / rs_wasm_reset() / rs_wasm_free().
EMSCRIPTEN_KEEPALIVE
const float *rs_wasm_get_audio_ptr(void) {
  return g_audio_buf;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_get_audio_len(void) {
  return g_audio_len;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_reset(void) {
  if (!g_ctx) return -1;
  g_last_text[0] = '\0';
  g_audio_len = 0;
  return (int)rs_reset(g_ctx);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_redecode(void) {
  if (!g_ctx) return -1;
  int ret = rs_redecode(g_ctx);
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

// ── ASR knobs (FunASRNano) ──────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_user_input_prompt(const char *prompt) {
  if (!g_ctx || !prompt) return -1;
  return (int)rs_set_user_input_prompt(g_ctx, prompt);
}

// ── True streaming ASR (X-ASR) ──────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_asr_stream_supported(void) {
  return (g_ctx && rs_asr_stream_supported(g_ctx)) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_asr_stream_set_chunk_len(int n_fbank_frames) {
  if (!g_ctx) return -1;
  return (int)rs_asr_stream_set_chunk_len(g_ctx, n_fbank_frames);
}

// Returns 1 if new tokens were emitted, 0 if not, -1 on error. On new tokens
// the running hypothesis is cached for rs_wasm_get_text().
EMSCRIPTEN_KEEPALIVE
int rs_wasm_asr_stream_push(const float *pcm, int n_samples) {
  if (!g_ctx) return -1;
  int ret = rs_asr_stream_push(g_ctx, pcm, n_samples);
  const char *text = rs_asr_stream_get_text(g_ctx);
  if (text) {
    int i = 0;
    while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
    g_last_text[i] = '\0';
  }
  return ret;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_asr_stream_finish(void) {
  if (!g_ctx) return -1;
  int ret = (int)rs_asr_stream_finish(g_ctx);
  const char *text = rs_asr_stream_get_text(g_ctx);
  if (text) {
    int i = 0;
    while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
    g_last_text[i] = '\0';
  }
  return ret;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_asr_stream_reset(void) {
  if (!g_ctx) return -1;
  g_last_text[0] = '\0';
  return (int)rs_asr_stream_reset(g_ctx);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_use_llm(int enable) {
  if (!g_ctx) return -1;
  return (int)rs_set_use_llm(g_ctx, enable != 0);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_ctc_precheck(int enable) {
  if (!g_ctx) return -1;
  return (int)rs_set_ctc_precheck(g_ctx, enable != 0);
}

// ── TTS knobs (OmniVoice) ───────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_tts_params(const char *instruct, const char *language, int seed) {
  if (!g_ctx) return -1;
  return (int)rs_set_tts_params(g_ctx, instruct, language, seed);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_tts_diffusion_steps(int n_steps) {
  if (!g_ctx) return -1;
  return (int)rs_set_tts_diffusion_steps(g_ctx, n_steps);
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

// ── VAD (independent of g_ctx — supports both silero-vad and firered-vad) ──
//
// Single-instance per WASM module; each tab creates its own MODULARIZE'd
// module so two simultaneous VADs can coexist across tabs.

static rs_vad_t *g_vad = nullptr;

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_init(const char *model_path, int n_threads) {
  if (g_vad) { rs_vad_free(g_vad); g_vad = nullptr; }
  // Pin VAD to CPU. Both silero-vad and firered-vad are tiny streaming
  // models invoked from a high-frequency AudioWorklet callback — WebGPU's
  // per-submit overhead dominates and the queue eventually fails with
  // "Queue work failed with status 3". CPU is both faster and async-safe
  // (no ASYNCIFY suspension inside vad.pushAudio).
  g_vad = rs_vad_init_from_file(model_path, n_threads, false);
  if (!g_vad) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_vad_init_from_file failed: " + UTF8ToString($0), $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_vad_free(void) {
  if (g_vad) { rs_vad_free(g_vad); g_vad = nullptr; }
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_reset(void) {
  if (!g_vad) return -1;
  return (int)rs_vad_reset(g_vad);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_set_threshold(float threshold) {
  if (!g_vad) return -1;
  return (int)rs_vad_set_threshold(g_vad, threshold);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_push_audio(const float *pcm, int n_samples) {
  if (!g_vad) return -1;
  return (int)rs_vad_push_audio(g_vad, pcm, n_samples);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_is_speech(void) {
  return g_vad ? rs_vad_is_speech(g_vad) : 0;
}

EMSCRIPTEN_KEEPALIVE
float rs_wasm_vad_get_probability(void) {
  return g_vad ? rs_vad_get_probability(g_vad) : 0.0f;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_vad_get_arch(void) {
  return g_vad ? rs_vad_get_arch(g_vad) : "";
}

// out: caller-allocated buffer of `capacity` rs_vad_segment_t (8 bytes each).
// Returns number of segments written.
EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_drain_segments(rs_vad_segment_t *out, int capacity) {
  if (!g_vad) return 0;
  return (int)rs_vad_drain_segments(g_vad, out, capacity);
}

// out: caller-allocated buffer of `capacity` rs_vad_frame_t (24 bytes each).
EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_drain_frames(rs_vad_frame_t *out, int capacity) {
  if (!g_vad) return 0;
  return (int)rs_vad_drain_frames(g_vad, out, capacity);
}

// ── Utility ─────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_version(void) {
  return rs_get_version();
}

// Set a process env var from JS BEFORE model init — the engine reads tuning
// and debug knobs via getenv (RS_REP_PENALTY, GGML_WEBGPU_NO_SUBGROUPS, ...),
// and emscripten's Module.ENV is baked at runtime start, too early for a page
// to reach. libc setenv works at any time.
EMSCRIPTEN_KEEPALIVE
int rs_wasm_setenv(const char *name, const char *value) {
  if (!name || !*name) return -1;
  return setenv(name, value ? value : "", 1);
}

// ── KWS (open-vocabulary wake-word, independent of g_ctx) ──
//
// Mirrors the VAD pattern: KWS owns its own rs_context_t so it can hold a
// SenseVoiceModel and reuse the loaded sched, independently of the main
// ASR/TTS context. JS reads hits via two host buffers:
//
//   - g_kws_records: contiguous array of HitRecord (24B each, 8B aligned),
//                    holds {time_s, avg_prob, phrase_off, phrase_len}.
//   - g_kws_phrases: single UTF-8 byte buffer; phrase_off/len index into it.
//
// Both are rebuilt on every rs_wasm_kws_poll() call.

namespace {

struct HitRecord {
  double   time_s;     // offset 0
  float    avg_prob;   // offset 8
  uint32_t phrase_off; // offset 12
  uint32_t phrase_len; // offset 16
  uint32_t _pad;       // offset 20 — pad to 24 so the next record is 8B-aligned
};
static_assert(sizeof(HitRecord) == 24, "HitRecord layout");

static rs_context_t                 *g_kws_ctx = nullptr;
static std::unique_ptr<rs::RSKws>    g_kws;
static std::vector<HitRecord>        g_kws_records;
static std::string                   g_kws_phrases;
static char                          g_kws_arch[64] = {0};

// SenseVoice prefix-token ids that the CTC stream wraps every utterance with
// (language, SER, AED, ITN). They are inside <…|…> bracket pairs in
// id_to_token. Treat them as no-ops during CTC collapse.
std::unordered_set<int32_t>
build_ignore_ids(const std::unordered_map<int, std::string> &id_to_token) {
  std::unordered_set<int32_t> out;
  for (const auto &kv : id_to_token) {
    const std::string &s = kv.second;
    if (s.size() >= 4 && s.front() == '<' && s.back() == '>') {
      out.insert(kv.first);
    }
  }
  return out;
}

// Punctuation token detection — see examples/kws/rs-kws.cpp:171-201 for the
// rationale. We accept ASCII punctuation and single-codepoint UTF-8 entries
// in the CJK / general punctuation blocks.
bool is_punct_token(const std::string &s) {
  if (s.empty()) return false;
  if (s.size() == 1) {
    char c = s[0];
    return std::strchr(",.!?;:()[]{}<>'\"`-_/\\|@#$%^&*+=~", c) != nullptr;
  }
  unsigned char b0 = static_cast<unsigned char>(s[0]);
  int cp_len = 0;
  uint32_t cp = 0;
  if      ((b0 & 0x80) == 0x00) { cp_len = 1; cp = b0; }
  else if ((b0 & 0xE0) == 0xC0) { cp_len = 2; cp = b0 & 0x1F; }
  else if ((b0 & 0xF0) == 0xE0) { cp_len = 3; cp = b0 & 0x0F; }
  else if ((b0 & 0xF8) == 0xF0) { cp_len = 4; cp = b0 & 0x07; }
  else return false;
  if (static_cast<int>(s.size()) != cp_len) return false;
  for (int i = 1; i < cp_len; ++i) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    if ((b & 0xC0) != 0x80) return false;
    cp = (cp << 6) | (b & 0x3F);
  }
  if (cp >= 0x3000 && cp <= 0x303F) return true;
  if (cp >= 0xFF00 && cp <= 0xFF65) return true;
  if (cp >= 0x2000 && cp <= 0x206F) return true;
  return false;
}

std::unordered_set<int32_t>
build_punct_ids(const std::unordered_map<int, std::string> &id_to_token) {
  std::unordered_set<int32_t> out;
  for (const auto &kv : id_to_token) {
    if (is_punct_token(kv.second)) out.insert(kv.first);
  }
  return out;
}

void kws_free_internal(void) {
  g_kws.reset();
  if (g_kws_ctx) { rs_free(g_kws_ctx); g_kws_ctx = nullptr; }
  g_kws_records.clear();
  g_kws_records.shrink_to_fit();
  g_kws_phrases.clear();
  g_kws_phrases.shrink_to_fit();
  g_kws_arch[0] = '\0';
}

} // namespace

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_init(const char *model_path, const char *keywords_path,
                     int n_threads, int window_ms, int hop_ms,
                     int debounce_ms, float default_threshold) {
  static bool archs_registered = false;
  if (!archs_registered) {
    rs_register_sensevoice();
    rs_register_moss_td();
    archs_registered = true;
  }
  if (!model_path || !keywords_path) return -1;

  kws_free_internal();

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads  = n_threads > 0 ? n_threads : 2;
  // Pin to CPU — KWS runs SenseVoice every hop (~200ms). WebGPU's per-submit
  // overhead dominates for short streaming windows; CPU + SIMD is faster and
  // async-safe (no ASYNCIFY suspension inside Poll()).
  params.use_gpu    = false;
  params.task_type  = RS_TASK_ASR_OFFLINE;

  g_kws_ctx = rs_init_from_file(params);
  if (!g_kws_ctx) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({ console.error("rs_wasm_kws_init: " + UTF8ToString($0)); }, err.message);
    return -1;
  }

  auto sv = std::dynamic_pointer_cast<SenseVoiceModel>(g_kws_ctx->model);
  if (!sv) {
    EM_ASM({ console.error("rs_wasm_kws_init: model is not SenseVoice"); });
    kws_free_internal();
    return -2;
  }

  rs::KWSLoaderConfig lcfg;
  lcfg.default_threshold = default_threshold;
  auto graph = rs::LoadKeywordsFromFile(keywords_path, sv->GetIdToToken(), lcfg);
  if (!graph) {
    EM_ASM({ console.error("rs_wasm_kws_init: no valid keywords loaded"); });
    kws_free_internal();
    return -3;
  }

  rs::RSKwsConfig kcfg;
  kcfg.sample_rate              = sv->GetMeta().audio_sample_rate;
  kcfg.window_ms                = window_ms   > 0 ? window_ms   : 1600;
  kcfg.hop_ms                   = hop_ms      > 0 ? hop_ms      : 200;
  kcfg.debounce_ms              = debounce_ms > 0 ? debounce_ms : 1500;
  kcfg.decoder.blank_id            = sv->GetBlankId();
  kcfg.decoder.skip_prefix_frames  = 4;
  kcfg.decoder.beam_size           = 1;
  kcfg.decoder.top_k_per_frame     = 8;

  auto ignore_ids = build_ignore_ids(sv->GetIdToToken());
  auto punct_ids  = build_punct_ids(sv->GetIdToToken());

  g_kws.reset(new rs::RSKws(sv, g_kws_ctx->sched, std::move(graph),
                            std::move(ignore_ids), std::move(punct_ids),
                            kcfg));

  const auto &name = sv->GetMeta().arch_name;
  size_t n = std::min(name.size(), sizeof(g_kws_arch) - 1);
  std::memcpy(g_kws_arch, name.data(), n);
  g_kws_arch[n] = '\0';
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_kws_free(void) {
  kws_free_internal();
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_reset(void) {
  if (!g_kws) return -1;
  g_kws->Reset();
  g_kws_records.clear();
  g_kws_phrases.clear();
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_push_audio(const float *pcm, int n_samples) {
  if (!g_kws || !pcm || n_samples <= 0) return -1;
  g_kws->PushAudio(pcm, static_cast<size_t>(n_samples));
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_poll(void) {
  if (!g_kws) return -1;
  g_kws_records.clear();
  g_kws_phrases.clear();
  g_kws->Poll([](const rs::KWSHit &h) {
    HitRecord rec;
    rec.time_s     = h.time_s;
    rec.avg_prob   = h.avg_prob;
    rec.phrase_off = static_cast<uint32_t>(g_kws_phrases.size());
    rec.phrase_len = static_cast<uint32_t>(h.phrase.size());
    rec._pad       = 0;
    g_kws_phrases.append(h.phrase);
    g_kws_records.push_back(rec);
  });
  return static_cast<int>(g_kws_records.size());
}

EMSCRIPTEN_KEEPALIVE
const void *rs_wasm_kws_get_hits_ptr(void) {
  return g_kws_records.empty() ? nullptr : g_kws_records.data();
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_get_hits_count(void) {
  return static_cast<int>(g_kws_records.size());
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_kws_get_phrase_pool_ptr(void) {
  return g_kws_phrases.empty() ? "" : g_kws_phrases.data();
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_get_phrase_pool_len(void) {
  return static_cast<int>(g_kws_phrases.size());
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_kws_get_sample_rate(void) {
  if (!g_kws_ctx) return 16000;
  rs_model_meta_t meta = rs_get_model_meta(g_kws_ctx);
  return meta.audio_sample_rate;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_kws_get_arch_name(void) {
  return g_kws_arch;
}

// ── Speaker embedding (CAM++, independent of g_ctx) ──
//
// Single CAM++ handle per WASM module. Mirrors the VAD/KWS pattern: its own
// global so speaker-id runs independently of the MOSS ASR context.

static rs_speaker_t *g_spk = nullptr;

EMSCRIPTEN_KEEPALIVE
int rs_wasm_speaker_init(const char *model_path, int n_threads) {
  if (g_spk) { rs_speaker_free(g_spk); g_spk = nullptr; }
  // use_gpu = 0: WASM has no GPU backend compiled in; CAM++ is CPU-only here.
  g_spk = rs_speaker_init_from_file(model_path, n_threads, /*use_gpu=*/false);
  if (!g_spk) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_speaker_init_from_file failed: " + UTF8ToString($0), $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_speaker_embed(const float *pcm, int n_samples, float *out_emb,
                          int out_size) {
  if (!g_spk || !pcm || n_samples <= 0 || !out_emb) return -1;
  rs_error_t rc = rs_speaker_embed(g_spk, pcm, n_samples, out_emb, out_size);
  return (rc == RS_OK) ? 0 : (int)rc;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_speaker_dim(void) {
  if (!g_spk) return 0;
  return (int)rs_speaker_dim(g_spk);
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_speaker_free(void) {
  if (g_spk) { rs_speaker_free(g_spk); g_spk = nullptr; }
}

#ifdef __cplusplus
}
#endif
