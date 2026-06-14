// End-to-end test of the integrated Matcha-TTS arch (matcha.cpp) in RapidSpeech.cpp ggml.
// Loads the matcha gguf, injects the validated phoneme IDs, runs the full pipeline
// (encoder -> length-regulate -> CFM decoder -> Vocos -> iSTFT), writes an 8 kHz wav.
//
// Build: link librapidspeech-core. Usage: matcha_e2e_test matcha8k.gguf out.wav
#include "rapidspeech.h"
#include "core/rs_context.h"
#include "arch/matcha.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>

static void write_wav(const char* path, const float* x, int n, int sr) {
  std::vector<int16_t> pcm(n);
  for (int i = 0; i < n; i++) { float v = x[i] < -1 ? -1 : (x[i] > 1 ? 1 : x[i]); pcm[i] = (int16_t)lrintf(v * 32767.f); }
  uint32_t db = n * 2, ch = 36 + db, br = sr * 2, scr = sr, s16 = 16; uint16_t ba = 2, bps = 16, fmt = 1, c1 = 1;
  FILE* f = fopen(path, "wb");
  fwrite("RIFF", 1, 4, f); fwrite(&ch, 4, 1, f); fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f); fwrite(&s16, 4, 1, f); fwrite(&fmt, 2, 1, f); fwrite(&c1, 2, 1, f); fwrite(&scr, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f); fwrite(&bps, 2, 1, f);
  fwrite("data", 1, 4, f); fwrite(&db, 4, 1, f); fwrite(pcm.data(), 2, n, f); fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: matcha_e2e_test gguf out.wav\n"); return 1; }
  rs_init_params_t p = rs_default_params();
  p.model_path = argv[1];
  rs_context_t* ctx = rs_init_from_file(p);
  if (!ctx || !ctx->model) { printf("load failed\n"); return 1; }
  auto state = ctx->model->CreateState();
  auto* ms = static_cast<MatchaState*>(state.get());
  // the 12 validated phoneme ids (decoder-isolation fixture); deterministic noise_scale=0
  ms->phoneme_ids = { 1, 5, 10, 20, 3, 7, 15, 2, 8, 40, 6, 0 };
  ms->noise_scale = 0.0f;
  ms->length_scale = 1.0f;
  // warm-synth timing: PushText is the full synthesis (encoder+length-reg+decoder+vocos+istft)
  auto clk = [] { return std::chrono::high_resolution_clock::now(); };
  double best = 1e9;
  for (int it = 0; it < 4; it++) {
    auto t0 = clk();
    if (!ctx->model->PushText(*state, "", nullptr, nullptr)) { printf("PushText failed\n"); return 1; }
    double el = std::chrono::duration<double, std::milli>(clk() - t0).count();
    printf("  synth iter%d = %.1f ms\n", it, el);
    if (it > 0 && el < best) best = el;  // skip iter0 (first-touch), keep best warm
    ms->audio_read_cursor = 0;  // reset so GetAudioOutput re-reads
  }
  printf("warm synth (best of 3) = %.1f ms\n", best);
  float* data = nullptr;
  int n = ctx->model->GetAudioOutput(*state, &data);
  printf("audio samples=%d (%.3fs @8k)\n", n, n / 8000.0f);
  if (n > 0) {
    double rms = 0; for (int i = 0; i < n; i++) rms += (double)data[i] * data[i]; rms = std::sqrt(rms / n);
    printf("rms=%.4f peak=%.3f\n", rms, [&]{ float m = 0; for (int i = 0; i < n; i++) m = std::fabs(data[i]) > m ? std::fabs(data[i]) : m; return m; }());
    write_wav(argv[2], data, n, 8000);
  }
  rs_free(ctx);
  return 0;
}
