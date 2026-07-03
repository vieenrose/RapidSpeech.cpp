// rs-asr-online — true chunked streaming ASR (X-ASR Zipformer2 transducer).
//
// Unlike rs-asr-vad-online (VAD segmentation + per-segment offline decode),
// this drives the model's chunked encoder with per-layer left-context caches:
// audio goes in as fixed chunks, partial text comes out with sub-second
// latency, and the transducer hypothesis is continuous across chunks.
//
// Usage:
//   rs-asr-online -m xasr.gguf -w audio.wav [--chunk-len 32] [--fast]
//   rs-asr-online -m xasr.gguf --mic [--chunk-len 32]

#include "arch/xasr.h"
#include "core/rs_context.h"
#include "rapidspeech.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#include "../common/rs_cli_utf8.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// ─────────────────────────────────────────────────────
// Cross-platform microphone capture via miniaudio
// ─────────────────────────────────────────────────────
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

struct MicRing {
  std::mutex mu;
  std::vector<float> buf;
  void write(const float *p, size_t n) {
    std::lock_guard<std::mutex> lk(mu);
    buf.insert(buf.end(), p, p + n);
  }
  std::vector<float> drain() {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<float> out;
    out.swap(buf);
    return out;
  }
};

struct MicCapture {
  ma_device device;
  MicRing *ring = nullptr;
  bool started = false;

  static void data_callback(ma_device *pDevice, void *, const void *pInput,
                            ma_uint32 frameCount) {
    auto *self = static_cast<MicCapture *>(pDevice->pUserData);
    if (self->ring && pInput) {
      self->ring->write(static_cast<const float *>(pInput), frameCount);
    }
  }

  bool start(int sample_rate, MicRing *rb) {
    ring = rb;
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = (ma_uint32)sample_rate;
    config.dataCallback = data_callback;
    config.pUserData = this;
    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS) {
      fprintf(stderr, "[online] failed to init microphone\n");
      return false;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
      fprintf(stderr, "[online] failed to start microphone\n");
      ma_device_uninit(&device);
      return false;
    }
    started = true;
    return true;
  }
  void stop() {
    if (started) {
      ma_device_stop(&device);
      ma_device_uninit(&device);
      started = false;
    }
  }
  ~MicCapture() { stop(); }
};

// ─────────────────────────────────────────────────────
// single-line partial display
//
// `\r\033[K` only rewrites the current terminal row; once the text exceeds
// the console width it soft-wraps onto extra rows that `\r` can't reach, and
// every refresh piles up stale lines. So the live partial is truncated to fit
// one row (keeping the tail, prefixed with `…`), and the full text is printed
// once at the end.
// ─────────────────────────────────────────────────────

static bool stdout_is_tty() {
#ifdef _WIN32
  return _isatty(_fileno(stdout)) != 0;
#else
  return isatty(fileno(stdout)) != 0;
#endif
}

static int term_cols() {
#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO info;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
    return info.srWindow.Right - info.srWindow.Left + 1;
  }
#else
  struct winsize ws;
  if (ioctl(fileno(stdout), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return ws.ws_col;
  }
#endif
  return 80;
}

// display columns of a codepoint: CJK / fullwidth take 2 cells
static int cp_columns(uint32_t cp) {
  if ((cp >= 0x1100 && cp <= 0x115f) || (cp >= 0x2e80 && cp <= 0xa4cf) ||
      (cp >= 0xac00 && cp <= 0xd7a3) || (cp >= 0xf900 && cp <= 0xfaff) ||
      (cp >= 0xfe30 && cp <= 0xfe4f) || (cp >= 0xff00 && cp <= 0xff60) ||
      (cp >= 0xffe0 && cp <= 0xffe6) || (cp >= 0x20000 && cp <= 0x3fffd)) {
    return 2;
  }
  return 1;
}

// return the tail of `text` that fits in `max_cols` display columns,
// prefixed with "…" when truncated
static std::string fit_tail(const std::string &text, int max_cols) {
  // decode utf-8 into (byte offset, columns) per codepoint
  std::vector<std::pair<size_t, int>> cps;
  int total = 0;
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = text[i];
    size_t n = 1;
    uint32_t cp = c;
    if ((c >> 5) == 0x6) { cp = c & 0x1f; n = 2; }
    else if ((c >> 4) == 0xe) { cp = c & 0x0f; n = 3; }
    else if ((c >> 3) == 0x1e) { cp = c & 0x07; n = 4; }
    for (size_t k = 1; k < n && i + k < text.size(); k++) {
      cp = (cp << 6) | (text[i + k] & 0x3f);
    }
    const int w = cp_columns(cp);
    cps.emplace_back(i, w);
    total += w;
    i += n;
  }
  if (total <= max_cols) return text;
  const int budget = max_cols - 2; // room for the "…" prefix
  int used = 0;
  size_t start = text.size();
  for (size_t j = cps.size(); j-- > 0;) {
    if (used + cps[j].second > budget) break;
    used += cps[j].second;
    start = cps[j].first;
  }
  return "\xe2\x80\xa6" + text.substr(start);
}

static void print_partial(const std::string &text) {
  if (!stdout_is_tty()) return; // pipes get only the final line
  printf("\r\033[K%s", fit_tail(text, term_cols() - 1).c_str());
  fflush(stdout);
}

static void print_final(const std::string &text) {
  if (stdout_is_tty()) {
    printf("\r\033[K");
  }
  printf("%s\n", text.c_str());
  fflush(stdout);
}

int main(int argc_raw, char **argv_raw) {
  rs::cli::Utf8Args utf8_args(argc_raw, argv_raw);
  const int argc = utf8_args.argc();
  char **argv = utf8_args.argv();

  const char *model_path = nullptr;
  const char *wav_path = nullptr;
  bool use_mic = false;
  bool fast = false;
  bool verbose = false;
  int chunk_len = 32; // fbank frames (10 ms each) => 320 ms
  int n_threads = 4;
  bool use_gpu = true;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
    else if (!strcmp(argv[i], "-w") && i + 1 < argc) wav_path = argv[++i];
    else if (!strcmp(argv[i], "--mic")) use_mic = true;
    else if (!strcmp(argv[i], "--fast")) fast = true;
    else if (!strcmp(argv[i], "-V") || !strcmp(argv[i], "--verbose")) verbose = true;
    else if (!strcmp(argv[i], "--chunk-len") && i + 1 < argc)
      chunk_len = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc)
      n_threads = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cpu")) use_gpu = false;
    else {
      fprintf(stderr,
              "usage: %s -m model.gguf (-w file.wav | --mic) "
              "[--chunk-len 16|32|48|64|96|192] [--fast] [-t N] [--cpu] [-V]\n",
              argv[0]);
      return 1;
    }
  }
  if (!model_path || (!wav_path && !use_mic)) {
    fprintf(stderr, "[online] need -m and one of -w/--mic\n");
    return 1;
  }
  if (chunk_len % 16 != 0) {
    fprintf(stderr, "[online] --chunk-len must be a multiple of 16\n");
    return 1;
  }

  // Silence framework + ggml chatter unless --verbose is set, so the partial
  // transcript line stays readable.
  if (!verbose) {
    rs_log_set_level(RSLogLevel::RS_LOG_LEVEL_WARN);
    ggml_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
  }

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads = n_threads;
  params.use_gpu = use_gpu;
  rs_context_t *ctx = rs_init_from_file(params);
  if (!ctx) {
    fprintf(stderr, "[online] failed to load model %s\n", model_path);
    return 1;
  }
  auto *xasr = dynamic_cast<XASRModel *>(ctx->model.get());
  if (!xasr) {
    fprintf(stderr, "[online] model is not an X-ASR GGUF (arch=%s)\n",
            ctx->model ? ctx->model->GetMeta().arch_name.c_str() : "?");
    rs_free(ctx);
    return 1;
  }
  xasr->SetChunkLen(chunk_len);

  auto state = xasr->CreateState();
  auto &st = static_cast<XASRState &>(*state);

  if (verbose) {
    fprintf(stderr, "[online] X-ASR streaming  chunk=%d ms  backend=%s\n",
            chunk_len * 10, use_gpu ? "gpu" : "cpu");
  }

  const int sr = 16000;
  const size_t step = (size_t)sr / 10; // feed 100 ms at a time

  if (wav_path) {
    std::vector<float> pcm;
    if (!load_wav_file_resampled(wav_path, pcm, sr)) {
      fprintf(stderr, "[online] failed to load %s\n", wav_path);
      rs_free(ctx);
      return 1;
    }
    if (verbose) {
      fprintf(stderr, "[online] wav: %.2f s%s\n", pcm.size() / (double)sr,
              fast ? " (fast mode)" : " (real-time paced)");
    }

    auto t0 = std::chrono::steady_clock::now();
    double compute_s = 0.0;
    for (size_t pos = 0; pos < pcm.size() && !g_stop; pos += step) {
      const size_t n = std::min(step, pcm.size() - pos);
      if (!fast) {
        // pace to real time relative to stream start
        const double target = (double)pos / sr;
        for (;;) {
          const double el =
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            t0)
                  .count();
          if (el >= target) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
      }
      const size_t before_tok = st.tokens.size();
      auto c0 = std::chrono::steady_clock::now();
      if (!xasr->EncodeStreamingChunk(pcm.data() + pos, n, st, ctx->sched)) {
        fprintf(stderr, "\n[online] streaming chunk failed\n");
        break;
      }
      compute_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - c0)
              .count();
      if (st.tokens.size() != before_tok) {
        print_partial(xasr->GetTranscription(st));
      }
    }
    auto c0 = std::chrono::steady_clock::now();
    xasr->FinishStream(st, ctx->sched);
    compute_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - c0)
            .count();
    print_final(xasr->GetTranscription(st));
    fprintf(stderr, "[online] audio %.2fs  compute %.2fs  RTF %.3f\n",
            pcm.size() / (double)sr, compute_s,
            compute_s / (pcm.size() / (double)sr));
  } else {
    signal(SIGINT, on_sigint);
    MicRing ring;
    MicCapture mic;
    if (!mic.start(sr, &ring)) {
      rs_free(ctx);
      return 1;
    }
    fprintf(stderr, "[online] listening... (Ctrl-C to stop)\n");    while (!g_stop) {
      std::vector<float> pcm = ring.drain();
      if (pcm.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      const size_t before_tok = st.tokens.size();
      if (!xasr->EncodeStreamingChunk(pcm.data(), pcm.size(), st,
                                      ctx->sched)) {
        fprintf(stderr, "\n[online] streaming chunk failed\n");
        break;
      }
      if (st.tokens.size() != before_tok) {
        print_partial(xasr->GetTranscription(st));
      }
    }
    mic.stop();
    xasr->FinishStream(st, ctx->sched);
    print_final(xasr->GetTranscription(st));
  }

  rs_free(ctx);
  return 0;
}
