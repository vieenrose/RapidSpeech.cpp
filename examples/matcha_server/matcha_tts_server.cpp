// Persistent Matcha-TTS service over a Unix domain socket.
//
// Loads the matcha gguf + ggml-CUDA backend + the built-in zh/en text frontend
// ONCE (1 CPU thread + CUDA), JITs the sm_53 kernels once via a warm-up synth,
// then accepts connections forever. The launch-bound first-graph cost is paid a
// single time, so every request runs at the warm RTF (~0.18 on Jetson Nano gen1).
//
// Protocol (one request per connection — simplest + robust):
//   client connects, writes UTF-8 TEXT, then half-closes (shutdown SHUT_WR);
//   server replies:  4-byte little-endian length N, then N bytes of an 8 kHz mono
//   PCM16 WAV (RIFF). N == 0 means synthesis failed. Then the server closes.
// Mixed zh/en text is handled by the built-in frontend (MATCHA_TOKENS +
// MATCHA_LEXICON + ESPEAK_DATA_PATH); no external phoneme-id step.
//
// Usage:  matcha-tts-server <matcha.gguf> <socket_path>
//   env:  MATCHA_TOKENS, MATCHA_LEXICON, ESPEAK_DATA_PATH, MATCHA_USE_CUDA=1
#include "rapidspeech.h"
#include "core/rs_context.h"
#include "arch/matcha.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <csignal>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

static std::vector<uint8_t> make_wav(const float* x, int n, int sr) {
  std::vector<uint8_t> w; w.reserve(44 + n * 2);
  auto put = [&](const void* p, size_t k) { const uint8_t* b = (const uint8_t*)p; w.insert(w.end(), b, b + k); };
  uint32_t db = (uint32_t)n * 2, ch = 36 + db, br = (uint32_t)sr * 2, scr = (uint32_t)sr, s16 = 16;
  uint16_t ba = 2, bps = 16, fmt = 1, c1 = 1;
  put("RIFF", 4); put(&ch, 4); put("WAVE", 4);
  put("fmt ", 4); put(&s16, 4); put(&fmt, 2); put(&c1, 2); put(&scr, 4); put(&br, 4); put(&ba, 2); put(&bps, 2);
  put("data", 4); put(&db, 4);
  for (int i = 0; i < n; i++) { float v = x[i] < -1 ? -1 : (x[i] > 1 ? 1 : x[i]); int16_t s = (int16_t)lrintf(v * 32767.f); put(&s, 2); }
  return w;
}

static bool send_all(int fd, const void* buf, size_t len) {
  const uint8_t* p = (const uint8_t*)buf;
  while (len) { ssize_t k = write(fd, p, len); if (k <= 0) return false; p += k; len -= (size_t)k; }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: matcha-tts-server <matcha.gguf> <socket_path>\n"); return 1; }
  signal(SIGPIPE, SIG_IGN);
  const char* gguf = argv[1];
  const char* sockpath = argv[2];

  rs_init_params_t p = rs_default_params();
  p.model_path = gguf;
  rs_context_t* ctx = rs_init_from_file(p);   // loads gguf + backend (CUDA if MATCHA_USE_CUDA=1)
  if (!ctx || !ctx->model) { fprintf(stderr, "matcha-tts-server: load failed (%s)\n", gguf); return 1; }

  // Warm-up: one synth through the built-in frontend so the espeak/lexicon load and
  // the sm_53 PTX-JIT / first-graph build are all paid before the first client.
  {
    auto st = ctx->model->CreateState();
    static_cast<MatchaState*>(st.get())->noise_scale = 0.0f;
    ctx->model->PushText(*st, "你好", "zh", nullptr);
  }

  int srv = socket(AF_UNIX, SOCK_STREAM, 0);
  if (srv < 0) { perror("socket"); return 1; }
  struct sockaddr_un addr; memset(&addr, 0, sizeof addr);
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sockpath, sizeof(addr.sun_path) - 1);
  unlink(sockpath);
  if (bind(srv, (struct sockaddr*)&addr, sizeof addr) < 0) { perror("bind"); return 1; }
  chmod(sockpath, 0666);
  if (listen(srv, 8) < 0) { perror("listen"); return 1; }
  fprintf(stderr, "matcha-tts-server: ready (warm) on %s\n", sockpath);

  for (;;) {
    int c = accept(srv, nullptr, nullptr);
    if (c < 0) { if (errno == EINTR) continue; perror("accept"); break; }

    // Read the whole request (UTF-8 text) until the client half-closes.
    std::string text; char buf[4096]; ssize_t k;
    while ((k = read(c, buf, sizeof buf)) > 0) text.append(buf, (size_t)k);
    // trim trailing whitespace/newline
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();

    std::vector<uint8_t> wav;
    if (!text.empty()) {
      auto st = ctx->model->CreateState();
      static_cast<MatchaState*>(st.get())->noise_scale = 0.0f;
      auto t0 = std::chrono::high_resolution_clock::now();
      if (ctx->model->PushText(*st, text.c_str(), "zh", nullptr)) {
        float* data = nullptr; int n = ctx->model->GetAudioOutput(*st, &data);
        if (n > 0 && data) wav = make_wav(data, n, 8000);
        auto t1 = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "matcha-tts-server: \"%.40s\" -> %d samp (%.2fs) in %.0f ms\n",
                text.c_str(), n, n / 8000.0, std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
    }
    uint32_t len = (uint32_t)wav.size();
    if (send_all(c, &len, 4) && len) send_all(c, wav.data(), wav.size());
    close(c);
  }
  close(srv); unlink(sockpath); rs_free(ctx);
  return 0;
}
