/**
 * rs-kws.cpp — Open-vocabulary streaming keyword spotter built on SenseVoice.
 *
 * Uses SenseVoice as a CTC encoder. A sliding window (default 1.6 s / 200 ms
 * hop) runs the encoder + CTC head; an Aho-Corasick trie matches keyword
 * token-id sequences against the greedy CTC output. Per-keyword debounce
 * prevents adjacent overlapping windows from double-triggering.
 *
 * Usage:
 *   rs-kws -m <sensevoice.gguf> -k <keywords.txt> -w <audio.wav>
 *
 * keywords.txt: each non-empty line is one keyword phrase, encoded as
 * space-separated SenseVoice tokens, optionally followed by `:boost`,
 * `#threshold` and `@phrase`. Use scripts/sensevoice_tokenize.py to
 * generate this file from natural-language phrases.
 */

#include "core/rs_context.h"
#include "core/rs_kws.h"
#include "arch/sensevoice.h"
#include "arch/keyword_loader.h"
#include "rapidspeech.h"
#include "utils/rs_wav.h"
#include "../common/rs_cli_utf8.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#define LOG_INFO(fmt, ...) std::printf("[kws] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) std::fprintf(stderr, "[kws] ERROR: " fmt "\n", ##__VA_ARGS__)

struct Args {
  const char *model_path = nullptr;
  const char *keywords_path = nullptr;
  const char *wav_path = nullptr;
  bool use_mic = false;
  int n_threads = 4;
  bool use_gpu = true;
  int window_ms = 1600;
  int hop_ms = 200;
  int debounce_ms = 1500;
  int num_trailing_blanks = 0;
  int beam_size = 1;
  int top_k = 8;
  float default_threshold = 0.0f;
};

static void print_usage(const char *argv0) {
  std::printf(
      "Usage: %s -m <sensevoice.gguf> -k <keywords.txt> (-w <audio.wav> | --mic) [options]\n"
      "  -m, --model <path>       SenseVoice gguf (required)\n"
      "  -k, --keywords <path>    keywords.txt (required)\n"
      "  -w, --wav <path>         input wav file\n"
      "      --mic                capture from default microphone (16kHz mono)\n"
      "  -t, --threads <n>        cpu threads (default 4)\n"
      "      --gpu <true|false>   enable gpu (default true)\n"
      "      --window-ms <n>      analysis window (default 1600)\n"
      "      --hop-ms <n>         hop (default 200)\n"
      "      --debounce-ms <n>    per-keyword debounce (default 1500)\n"
      "      --trailing-blanks <n>  required CTC trailing blanks (default 0)\n"
      "      --threshold <f>      default avg-prob gate (default 0.0)\n"
      "      --beam <n>           CTC beam size (1=greedy MVP, >1=beam V2, default 1)\n"
      "      --top-k <n>          per-frame candidates considered in beam (default 8)\n",
      argv0);
}

static bool parse_args(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    const char *k = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        LOG_ERR("%s requires a value", name);
        return nullptr;
      }
      return argv[++i];
    };
    if (!std::strcmp(k, "-h") || !std::strcmp(k, "--help")) {
      print_usage(argv[0]);
      return false;
    } else if (!std::strcmp(k, "-m") || !std::strcmp(k, "--model")) {
      a.model_path = need("--model"); if (!a.model_path) return false;
    } else if (!std::strcmp(k, "-k") || !std::strcmp(k, "--keywords")) {
      a.keywords_path = need("--keywords"); if (!a.keywords_path) return false;
    } else if (!std::strcmp(k, "-w") || !std::strcmp(k, "--wav")) {
      a.wav_path = need("--wav"); if (!a.wav_path) return false;
    } else if (!std::strcmp(k, "--mic")) {
      a.use_mic = true;
    } else if (!std::strcmp(k, "-t") || !std::strcmp(k, "--threads")) {
      const char *v = need("--threads"); if (!v) return false;
      a.n_threads = std::atoi(v);
    } else if (!std::strcmp(k, "--gpu")) {
      const char *v = need("--gpu"); if (!v) return false;
      a.use_gpu = !std::strcmp(v, "true") || !std::strcmp(v, "1");
    } else if (!std::strcmp(k, "--window-ms")) {
      const char *v = need("--window-ms"); if (!v) return false;
      a.window_ms = std::atoi(v);
    } else if (!std::strcmp(k, "--hop-ms")) {
      const char *v = need("--hop-ms"); if (!v) return false;
      a.hop_ms = std::atoi(v);
    } else if (!std::strcmp(k, "--debounce-ms")) {
      const char *v = need("--debounce-ms"); if (!v) return false;
      a.debounce_ms = std::atoi(v);
    } else if (!std::strcmp(k, "--trailing-blanks")) {
      const char *v = need("--trailing-blanks"); if (!v) return false;
      a.num_trailing_blanks = std::atoi(v);
    } else if (!std::strcmp(k, "--threshold")) {
      const char *v = need("--threshold"); if (!v) return false;
      a.default_threshold = std::strtof(v, nullptr);
    } else if (!std::strcmp(k, "--beam")) {
      const char *v = need("--beam"); if (!v) return false;
      a.beam_size = std::atoi(v);
    } else if (!std::strcmp(k, "--top-k")) {
      const char *v = need("--top-k"); if (!v) return false;
      a.top_k = std::atoi(v);
    } else {
      LOG_ERR("unknown arg: %s", k);
      print_usage(argv[0]);
      return false;
    }
  }
  if (!a.model_path || !a.keywords_path || (!a.wav_path && !a.use_mic)) {
    print_usage(argv[0]);
    return false;
  }
  if (a.wav_path && a.use_mic) {
    LOG_ERR("--wav and --mic are mutually exclusive");
    return false;
  }
  return true;
}

// Collect the SenseVoice prefix-token ids so the decoder treats them as
// no-ops during CTC collapse. These are language tags (<|zh|> <|en|> ...),
// SER (<|EMO_*|>), AED (<|Speech|> <|BGM|> ...), and ITN (<|withitn|>
// <|woitn|>). They live inside angle-and-pipe brackets in id_to_token.
static std::unordered_set<int32_t>
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

// Collect punctuation token ids so the CTC stream from SenseVoice doesn't
// break the trie walk when ASR-style ITN inserts "，。！？" between keyword
// characters. SenseVoice tokens are mostly UTF-8 single CJK chars or BPE
// pieces (with "▁" leading). We treat a token as punctuation if its UTF-8
// payload is a single codepoint in the Unicode punctuation/symbol ranges
// most commonly emitted by FunASR ITN.
static bool is_punct_token(const std::string &s) {
  if (s.empty()) return false;
  // ASCII punctuation
  if (s.size() == 1) {
    char c = s[0];
    return std::strchr(",.!?;:()[]{}<>'\"`-_/\\|@#$%^&*+=~", c) != nullptr;
  }
  // Multi-byte: accept only if total bytes equal a single UTF-8 codepoint
  // length AND that codepoint falls in CJK / general punctuation blocks.
  unsigned char b0 = static_cast<unsigned char>(s[0]);
  int cp_len = 0;
  uint32_t cp = 0;
  if ((b0 & 0x80) == 0) { cp_len = 1; cp = b0; }
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
  // CJK Symbols and Punctuation 3000–303F
  if (cp >= 0x3000 && cp <= 0x303F) return true;
  // Fullwidth ASCII punctuation in Halfwidth/Fullwidth Forms FF00–FF65
  if (cp >= 0xFF00 && cp <= 0xFF65) return true;
  // General Punctuation 2000–206F
  if (cp >= 0x2000 && cp <= 0x206F) return true;
  return false;
}

static std::unordered_set<int32_t>
build_punct_ids(const std::unordered_map<int, std::string> &id_to_token) {
  std::unordered_set<int32_t> out;
  for (const auto &kv : id_to_token) {
    if (is_punct_token(kv.second)) out.insert(kv.first);
  }
  return out;
}

// ─────────────────────────────────────────────────────
// Microphone capture via miniaudio. Writes float32 mono samples into a
// shared buffer guarded by a mutex; the main thread drains it into
// rs::RSKws::PushAudio().
// ─────────────────────────────────────────────────────
struct MicCapture {
  ma_device device{};
  std::mutex mu;
  std::vector<float> pending;
  std::atomic<bool> running{false};

  static void data_callback(ma_device *dev, void *, const void *in,
                            ma_uint32 frame_count) {
    auto *self = static_cast<MicCapture *>(dev->pUserData);
    if (!in) return;
    const float *src = static_cast<const float *>(in);
    std::lock_guard<std::mutex> g(self->mu);
    self->pending.insert(self->pending.end(), src, src + frame_count);
  }

  bool start(int sample_rate) {
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = static_cast<ma_uint32>(sample_rate);
    cfg.dataCallback = data_callback;
    cfg.pUserData = this;
    if (ma_device_init(nullptr, &cfg, &device) != MA_SUCCESS) {
      LOG_ERR("ma_device_init failed");
      return false;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
      LOG_ERR("ma_device_start failed");
      ma_device_uninit(&device);
      return false;
    }
    running = true;
    LOG_INFO("mic capture started @ %d Hz", sample_rate);
    return true;
  }

  std::vector<float> drain() {
    std::vector<float> out;
    std::lock_guard<std::mutex> g(mu);
    pending.swap(out);
    return out;
  }

  void stop() {
    if (running.exchange(false)) {
      if (ma_device_is_started(&device)) ma_device_stop(&device);
      ma_device_uninit(&device);
    }
  }
  ~MicCapture() { stop(); }
};

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

int main(int argc, char **argv) {
  rs::cli::Utf8Args utf8_args(argc, argv);

  Args args;
  if (!parse_args(utf8_args.argc(), utf8_args.argv(), args)) return 1;

  LOG_INFO("RapidSpeech.cpp v%s", rs_get_version());
  LOG_INFO("model: %s", args.model_path);
  LOG_INFO("keywords: %s", args.keywords_path);
  LOG_INFO("wav: %s", args.wav_path);
  LOG_INFO("window=%dms hop=%dms debounce=%dms trailing-blanks=%d",
           args.window_ms, args.hop_ms, args.debounce_ms,
           args.num_trailing_blanks);

  // 1. Init SenseVoice via the standard rs_context, then grab the underlying
  //    model + scheduler. We bypass RSProcessor because KWS needs raw
  //    [T, V] log_probs, not text output.
  rs_init_params_t params = rs_default_params();
  params.model_path = args.model_path;
  params.n_threads = args.n_threads;
  params.use_gpu = args.use_gpu;

  rs_context_t *ctx = rs_init_from_file(params);
  if (!ctx) {
    LOG_ERR("rs_init_from_file failed: %s", rs_get_last_error().message);
    return 1;
  }
  LOG_INFO("backend: %s", rs_get_backend_name(ctx));

  auto sv = std::dynamic_pointer_cast<SenseVoiceModel>(ctx->model);
  if (!sv) {
    LOG_ERR("model is not SenseVoice (got arch '%s')",
            ctx->model->GetMeta().arch_name.c_str());
    rs_free(ctx);
    return 1;
  }

  // 2. Load keywords → ContextGraph.
  rs::KWSLoaderConfig lcfg;
  lcfg.default_threshold = args.default_threshold;
  auto graph =
      rs::LoadKeywordsFromFile(args.keywords_path, sv->GetIdToToken(), lcfg);
  if (!graph) {
    LOG_ERR("no valid keywords loaded");
    rs_free(ctx);
    return 1;
  }

  // 3. Build the streaming KWS engine.
  rs::RSKwsConfig kcfg;
  kcfg.sample_rate = sv->GetMeta().audio_sample_rate;
  kcfg.window_ms = args.window_ms;
  kcfg.hop_ms = args.hop_ms;
  kcfg.debounce_ms = args.debounce_ms;
  kcfg.decoder.blank_id = sv->GetBlankId();
  kcfg.decoder.num_trailing_blanks = args.num_trailing_blanks;
  kcfg.decoder.skip_prefix_frames = 4; // SenseVoice prepends 4 prompt frames
  kcfg.decoder.beam_size = args.beam_size;
  kcfg.decoder.top_k_per_frame = args.top_k;

  auto ignore_ids = build_ignore_ids(sv->GetIdToToken());
  auto punct_ids = build_punct_ids(sv->GetIdToToken());
  LOG_INFO("ignore_ids=%zu  punct_ids=%zu  beam=%d  top_k=%d",
           ignore_ids.size(), punct_ids.size(), args.beam_size, args.top_k);

  rs::RSKws kws(sv, ctx->sched, graph, std::move(ignore_ids),
                std::move(punct_ids), kcfg);

  if (args.use_mic) {
    // Microphone mode: capture indefinitely; ^C to stop.
    MicCapture mic;
    if (!mic.start(kcfg.sample_rate)) {
      rs_free(ctx);
      return 1;
    }
    std::signal(SIGINT, on_sigint);
    LOG_INFO("listening... press ^C to stop");
    while (!g_stop.load()) {
      auto chunk = mic.drain();
      if (!chunk.empty()) {
        kws.PushAudio(chunk.data(), chunk.size());
      }
      kws.Poll([&](const rs::KWSHit &h) {
        LOG_INFO("HIT  t=%.2fs  avg_prob=%.3f  phrase='%s'", h.time_s,
                 h.avg_prob, h.phrase.c_str());
      });
      // Sleep a bit less than one hop so we never lag behind the audio.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(std::max(10, args.hop_ms / 2)));
    }
    mic.stop();
    LOG_INFO("stopped.");
    rs_free(ctx);
    return 0;
  }

  // 4. WAV mode: load file, resample if needed.
  std::vector<float> pcm;
  int wav_sr = kcfg.sample_rate;
  if (!load_wav_file(args.wav_path, pcm, &wav_sr)) {
    LOG_ERR("failed to load %s", args.wav_path);
    rs_free(ctx);
    return 1;
  }
  if (wav_sr != kcfg.sample_rate) {
    std::vector<float> rs;
    if (!resample_pcm(pcm, wav_sr, rs, kcfg.sample_rate)) {
      LOG_ERR("resample %d -> %d failed", wav_sr, kcfg.sample_rate);
      rs_free(ctx);
      return 1;
    }
    pcm = std::move(rs);
  }
  LOG_INFO("wav: %.2fs, %d Hz, %zu samples",
           pcm.size() / float(kcfg.sample_rate), kcfg.sample_rate, pcm.size());

  // 5. Stream the wav through the engine in 200 ms chunks to simulate live.
  const int chunk = kcfg.sample_rate * args.hop_ms / 1000;
  int hits_total = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (size_t off = 0; off < pcm.size(); off += chunk) {
    size_t n = std::min<size_t>(chunk, pcm.size() - off);
    kws.PushAudio(pcm.data() + off, n);
    kws.Poll([&](const rs::KWSHit &h) {
      LOG_INFO("HIT  t=%.2fs  avg_prob=%.3f  phrase='%s'", h.time_s, h.avg_prob,
               h.phrase.c_str());
      ++hits_total;
    });
  }
  auto t1 = std::chrono::steady_clock::now();
  double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count() /
              1e6;
  double dur = pcm.size() / double(kcfg.sample_rate);
  LOG_INFO("done. hits=%d  wall=%.2fs  audio=%.2fs  rtf=%.3f", hits_total, dt,
           dur, dt / dur);

  rs_free(ctx);
  return 0;
}
