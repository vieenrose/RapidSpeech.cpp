// Minimal load-test: init context from a MOSS gguf, confirm the arch resolves
// all tensors (MossTTSNanoModel::Load returns false on any missing tensor).
#include "core/rs_context.h"
#include "rapidspeech.h"
#include <cstdio>
int main(int argc, char **argv) {
  if (argc < 2) { printf("usage: %s model.gguf [--gpu]\n", argv[0]); return 2; }
  rs_init_params_t p{};
  p.model_path = argv[1];
  p.n_threads = 2;
  p.use_gpu = (argc > 2 && std::string(argv[2]) == "--gpu");
  p.task_type = RS_TASK_TTS_ONLINE;
  rs_context_t *ctx = rs_init_from_file(p);
  if (!ctx || !ctx->model) { printf("LOAD FAILED\n"); return 1; }
  const RSModelMeta &m = ctx->model->GetMeta();
  printf("LOAD OK: arch=%s sr=%d vocab=%d\n", m.arch_name.c_str(), m.audio_sample_rate, m.vocab_size);
  auto st = ctx->model->CreateState();
  printf("state created: %s\n", st ? "ok" : "null");
  rs_free(ctx);
  return 0;
}
