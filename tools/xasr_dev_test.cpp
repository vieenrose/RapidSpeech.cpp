// Dev harness for X-ASR zipformer2 parity testing. Loads a GGUF, runs a single
// stage (currently encoder_embed) on a raw feature chunk, writes the result as
// raw float32 for numerical comparison with scripts/xasr/dump_reference.py.
//
//   xasr-dev-test embed <model.gguf> <feats.bin> <T> <feat_dim> <out.bin>
//
// feats.bin: row-major float32 [T, feat_dim] (freq fastest), one chunk.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

bool xasr_debug_embed(const std::map<std::string, struct ggml_tensor *> &w,
                      ggml_backend_t backend, const float *feats, int T,
                      int feat_dim, std::vector<float> &out, int *T_out,
                      int *dim_out);
bool xasr_debug_encoder(const std::map<std::string, struct ggml_tensor *> &w,
                        ggml_backend_t backend, const float *feats, int T,
                        int feat_dim, std::vector<float> &out, int *T_out,
                        int *dim_out);
bool xasr_debug_encoder_stream(const std::map<std::string, struct ggml_tensor *> &w,
                               ggml_backend_t backend, const float *feats, int n_frames,
                               int feat_dim, std::vector<float> &out, int *out_dim, int *n_out);

static std::vector<float> read_bin(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / sizeof(float));
  if (fread(v.data(), 1, n, f) != (size_t)n) { exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char **argv) {
  std::string stage = argc > 1 ? argv[1] : "";
  if (argc < 7 || (stage != "embed" && stage != "encoder" && stage != "stream")) {
    fprintf(stderr, "usage: %s <embed|encoder|stream> <model.gguf> <feats.bin> <T|n_frames> <feat_dim> <out.bin>\n", argv[0]);
    return 1;
  }
  const char *model = argv[2], *featf = argv[3], *outf = argv[6];
  int T = atoi(argv[4]), feat_dim = atoi(argv[5]);

  ggml_context *data_ctx = nullptr;
  gguf_init_params gp{/*no_alloc=*/true, /*ctx=*/&data_ctx};
  gguf_context *gg = gguf_init_from_file(model, gp);
  if (!gg) { fprintf(stderr, "gguf load failed\n"); return 1; }

  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(data_ctx, backend);
  (void)buf;

  FILE *f = fopen(model, "rb");
  size_t data_off = gguf_get_data_offset(gg);
  int n = gguf_get_n_tensors(gg);
  std::vector<char> rb;
  std::map<std::string, ggml_tensor *> w;
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(gg, i);
    ggml_tensor *t = ggml_get_tensor(data_ctx, name);
    if (!t) continue;
    size_t off = gguf_get_tensor_offset(gg, i), sz = ggml_nbytes(t);
    if (rb.size() < sz) rb.resize(sz);
    fseek(f, data_off + off, SEEK_SET);
    if (fread(rb.data(), 1, sz, f) != sz) { fprintf(stderr, "read %s failed\n", name); return 1; }
    ggml_backend_tensor_set(t, rb.data(), 0, sz);
    w[name] = t;
  }
  fclose(f);
  fprintf(stderr, "loaded %zu tensors\n", w.size());
  for (const char *nm : {"encoder_embed.conv.0.weight", "encoder_embed.conv.4.weight",
                         "encoder.encoders.0.layers.0.self_attn1.in_proj.weight",
                         "encoder_proj.weight", "decoder.embedding.weight",
                         "joiner.output_linear.weight"}) {
    ggml_tensor *t = w.count(nm) ? w[nm] : nullptr;
    if (t) fprintf(stderr, "  NE %-52s [%lld,%lld,%lld,%lld] type=%d\n", nm,
                   (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                   (long long)t->ne[3], (int)t->type);
  }
  if (getenv("XASR_NE_ONLY")) return 0;

  std::vector<float> feats = read_bin(featf);
  if ((int)feats.size() != T * feat_dim) {
    fprintf(stderr, "feats size %zu != T*feat_dim %d\n", feats.size(), T * feat_dim);
    return 1;
  }

  std::vector<float> out; int T_out = 0, dim_out = 0;
  bool ok;
  if (stage == "stream")
    ok = xasr_debug_encoder_stream(w, backend, feats.data(), T, feat_dim, out, &dim_out, &T_out);
  else if (stage == "encoder")
    ok = xasr_debug_encoder(w, backend, feats.data(), T, feat_dim, out, &T_out, &dim_out);
  else
    ok = xasr_debug_embed(w, backend, feats.data(), T, feat_dim, out, &T_out, &dim_out);
  if (!ok) { fprintf(stderr, "%s failed\n", stage.c_str()); return 1; }
  fprintf(stderr, "%s out: T_out=%d dim=%d\n", stage.c_str(), T_out, dim_out);
  FILE *o = fopen(outf, "wb");
  fwrite(out.data(), sizeof(float), out.size(), o);
  fclose(o);
  return 0;
}
