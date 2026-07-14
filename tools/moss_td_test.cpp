// Minimal direct harness for the MossTD arch: load GGUF, read a 16 kHz mono
// WAV, run Encode + Decode, print the transcription. Bypasses the VAD/segment
// processor so we exercise MOSS's own single-pass long-form path for parity
// against the HF/ONNX reference.
//
// Usage: moss_td_test <moss-td.gguf> <audio.wav> [--gpu] [--prompt "..."]
#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: %s <moss-td.gguf> <audio.wav> [--gpu] [--prompt \"...\"]\n",
           argv[0]);
    return 2;
  }
  const char *model_path = argv[1];
  const char *wav_path   = argv[2];
  bool use_gpu = false;
  std::string prompt;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--gpu") use_gpu = true;
    else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
  }

  rs_init_params_t p{};
  p.model_path = model_path;
  p.n_threads  = 4;
  p.use_gpu    = use_gpu;
  p.task_type  = RS_TASK_ASR_OFFLINE;
  rs_context_t *ctx = rs_init_from_file(p);
  if (!ctx || !ctx->model) { printf("LOAD FAILED\n"); return 1; }
  const RSModelMeta &m = ctx->model->GetMeta();
  printf("LOAD OK: arch=%s sr=%d vocab=%d\n", m.arch_name.c_str(),
         m.audio_sample_rate, m.vocab_size);

  if (!prompt.empty()) ctx->model->SetUserInputPrompt(prompt);

  std::vector<float> pcm;
  if (!load_wav_file_resampled(wav_path, pcm, m.audio_sample_rate)) {
    printf("WAV LOAD FAILED: %s\n", wav_path);
    rs_free(ctx);
    return 1;
  }
  printf("audio: %zu samples (%.2f s)\n", pcm.size(),
         (double)pcm.size() / m.audio_sample_rate);

  auto st = ctx->model->CreateState();
  if (!ctx->model->Encode(pcm, *st, ctx->sched)) {
    printf("ENCODE FAILED\n"); rs_free(ctx); return 1;
  }
  if (!ctx->model->Decode(*st, ctx->sched)) {
    printf("DECODE FAILED\n"); rs_free(ctx); return 1;
  }
  std::string text = ctx->model->GetTranscription(*st);
  printf("\n===== TRANSCRIPTION =====\n%s\n=========================\n",
         text.c_str());
  rs_free(ctx);
  return 0;
}
