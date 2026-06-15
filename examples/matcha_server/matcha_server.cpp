// Persistent Matcha-TTS service for RapidSpeech.cpp ggml. (Demo / reference.)
//
// This is a transport-agnostic demo of the warm-persistent pattern; the production
// integration wraps this same engine loop as a LiveKit Agents TTS plugin.
//
// Loads the matcha gguf and (on RS_CUDA builds, with MATCHA_USE_CUDA=1) the CUDA
// backend ONCE, JITs the sm_53 kernels once, then serves synthesis requests in a
// loop — so the launch-bound first-graph cost is paid a single time and every
// subsequent request runs at the warm RTF (on Jetson Nano gen1: ~0.18 vs ~1.0
// cold per-call). This is the deployable shape for matcha on the Maxwell GPU.
//
// Frontend note: the matcha ggml arch has no text->phoneme frontend; it is driven
// by espeak-IPA phoneme IDs (from sherpa-onnx's matcha frontend, --debug=1). This
// service therefore takes phoneme-ID files, not raw text.
//
// Protocol (line-oriented on stdin; one reply line on stdout per request):
//   <ids_file>  <out_wav>  [length_scale]
//     ids_file   : little-endian int32 phoneme IDs
//     out_wav    : 8 kHz mono PCM16 output path
//     length_scale: optional float (default 1.0)
//   -> reply:  OK <out_wav> <nsamples> <synth_ms>   |   ERR <reason>
// EOF on stdin ends the service.
//
// Build: link librapidspeech-core (see examples/matcha_server/CMake or the gen1
// cross-build recipe). Usage: matcha_server <matcha.gguf>
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

static void write_wav(const char* path, const float* x, int n, int sr) {
  std::vector<int16_t> pcm(n);
  for (int i = 0; i < n; i++) { float v = x[i] < -1 ? -1 : (x[i] > 1 ? 1 : x[i]); pcm[i] = (int16_t)lrintf(v * 32767.f); }
  uint32_t db = n * 2, ch = 36 + db, br = sr * 2, scr = sr, s16 = 16; uint16_t ba = 2, bps = 16, fmt = 1, c1 = 1;
  FILE* f = fopen(path, "wb"); if (!f) return;
  fwrite("RIFF",1,4,f); fwrite(&ch,4,1,f); fwrite("WAVE",1,4,f);
  fwrite("fmt ",1,4,f); fwrite(&s16,4,1,f); fwrite(&fmt,2,1,f); fwrite(&c1,2,1,f); fwrite(&scr,4,1,f); fwrite(&br,4,1,f); fwrite(&ba,2,1,f); fwrite(&bps,2,1,f);
  fwrite("data",1,4,f); fwrite(&db,4,1,f); fwrite(pcm.data(),2,n,f); fclose(f);
}

static bool read_ids(const char* path, std::vector<int32_t>& ids) {
  FILE* f = fopen(path, "rb"); if (!f) return false;
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  ids.resize(sz / 4);
  bool ok = ids.size() && fread(ids.data(), 4, ids.size(), f) == ids.size();
  fclose(f); return ok;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: matcha_server <matcha.gguf>\n"); return 1; }
  rs_init_params_t p = rs_default_params();
  p.model_path = argv[1];
  rs_context_t* ctx = rs_init_from_file(p);   // loads gguf + backend (CUDA if MATCHA_USE_CUDA=1)
  if (!ctx || !ctx->model) { fprintf(stderr, "load failed\n"); return 1; }
  auto state = ctx->model->CreateState();
  auto* ms = static_cast<MatchaState*>(state.get());
  ms->noise_scale = 0.0f;   // deterministic; raise for sampling variety

  // Warm the kernels once (one synth on the loaded ids, output discarded) so the
  // first real request isn't charged the PTX-JIT / first-graph cost.
  ms->phoneme_ids = { 1, 5, 10, 20, 3, 7, 15, 2, 8, 40, 6, 0 };
  ctx->model->PushText(*state, "", nullptr, nullptr);
  fprintf(stderr, "matcha_server: ready (warm)\n");
  setvbuf(stdout, nullptr, _IOLBF, 0);

  char line[4096];
  while (fgets(line, sizeof line, stdin)) {
    char idf[2048] = {0}, outf[2048] = {0}; float ls = 1.0f;
    int nf = sscanf(line, "%2047s %2047s %f", idf, outf, &ls);
    if (nf < 2) { printf("ERR bad_request\n"); continue; }
    if (!read_ids(idf, ms->phoneme_ids) || ms->phoneme_ids.empty()) { printf("ERR ids %s\n", idf); continue; }
    ms->length_scale = ls;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!ctx->model->PushText(*state, "", nullptr, nullptr)) { printf("ERR synth\n"); continue; }
    float* data = nullptr; int n = ctx->model->GetAudioOutput(*state, &data);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_w = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (n <= 0 || !data) { printf("ERR empty\n"); continue; }
    write_wav(outf, data, n, 8000);
    printf("OK %s %d %.1f\n", outf, n, ms_w);
  }
  rs_free(ctx);
  return 0;
}
