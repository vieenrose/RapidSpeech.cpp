// Staged validator for the Matcha-TTS acoustic text-encoder ported to ggml.
// Stage 1 (this file): embedding ×sqrt(hidden) + ConvReluNorm prenet.
//
// Glow-TTS/Matcha text encoder front:
//   x = emb(tokens) * sqrt(192)                       [L,192] -> transpose [192,L]
//   prenet (ConvReluNorm): x_org = x
//     for i in 0..2: x = conv_i(x, k5,p2) -> channelLN(gamma_i,beta_i) -> ReLU
//     x = x_org + proj(x, k1)
//   (x_mask all-ones for unpadded input, so omitted)
//
// channel-LayerNorm (over C): mean/var over channels, then *gamma + beta. The gamma/beta
// are ONNX-folded Constants captured by the converter as
//   model.encoder.prenet.norm_layers.<i>.Mul_1.weight  (gamma)
//   model.encoder.prenet.norm_layers.<i>.Add_1.weight   (beta)
//
// Build (CPU): g++ -std=c++17 matcha_encoder_validate.cpp -I<ggml>/include -L<lib> \
//   -lggml -lggml-base -lggml-cpu -o ev
// Usage: ev matcha8k.gguf enc_tokens.i64 enc_prenet_ref.f32 L
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

static std::vector<float> rf32(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
  fclose(f); return v;
}
static std::vector<int64_t> ri64(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<int64_t> v(n / 8); if (fread(v.data(), 8, v.size(), f) != v.size()) exit(1);
  fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: ev gguf tokens.i64 prenet_ref.f32 L\n"); return 1; }
  const int L = atoi(argv[4]);
  const int H = 192;

  ggml_context* wctx = nullptr;
  gguf_init_params gp{ false, &wctx };
  if (!gguf_init_from_file(argv[1], gp)) { fprintf(stderr, "gguf load fail\n"); return 1; }
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* c = ggml_get_first_tensor(wctx); c; c = ggml_get_next_tensor(wctx, c))
    W[ggml_get_name(c)] = c;
  auto V = [&](const std::string& s) -> ggml_tensor* {
    auto it = W.find(s); if (it == W.end()) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); }
    return it->second;
  };

  ggml_init_params ip{ (size_t)256 * 1024 * 1024, nullptr, false };
  ggml_context* ctx = ggml_init(ip);

  auto f16 = [&](ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(ctx, w, GGML_TYPE_F16) : w; };
  // channel-LayerNorm: x[C,T] (ne0=C). gamma/beta are folded Constants stored as
  // ggml ne=[1,C,1] — reshape to 1-D [C] so they broadcast over ne0.
  auto r1 = [&](ggml_tensor* t) { return ggml_reshape_1d(ctx, t, ggml_nelements(t)); };
  auto cln = [&](ggml_tensor* x, ggml_tensor* g, ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-4f); x = ggml_mul(ctx, x, r1(g)); return ggml_add(ctx, x, r1(b));
  };

  // tokens -> embedding rows. emb.weight ne=[192,2190]; ggml_get_rows(emb, ids[int32]) -> [192,L]
  ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
  ggml_tensor* emb = ggml_get_rows(ctx, V("model.encoder.emb.weight"), ids);  // [192,L]
  emb = ggml_scale(ctx, emb, std::sqrt((float)H));                            // *sqrt(192)
  // prenet works in [T,C] for conv; x_org keeps [C,T] (=emb)
  ggml_tensor* x_org = emb;                                  // [192,L]
  ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, emb)); // [L,192]
  for (int i = 0; i < 3; i++) {
    std::string p = "model.encoder.prenet.";
    std::string n = "model.encoder.prenet.norm_layers." + std::to_string(i) + ".";
    ggml_tensor* w = V(p + "conv_layers." + std::to_string(i) + ".weight");
    ggml_tensor* b = V(p + "conv_layers." + std::to_string(i) + ".bias");
    x = ggml_conv_1d(ctx, f16(w), x, 1, 2, 1);                // [L,192]
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, H));
    x = ggml_cont(ctx, ggml_transpose(ctx, x));              // [192,L]
    x = cln(x, V(n + "Mul_1.weight"), V(n + "Add_1.weight"));
    x = ggml_relu(ctx, x);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));              // [L,192]
  }
  // proj k1
  ggml_tensor* pw = V("model.encoder.prenet.proj.weight"); // [192,192,1]
  x = ggml_conv_1d(ctx, f16(pw), x, 1, 0, 1);                // [L,192]
  x = ggml_add(ctx, x, ggml_reshape_2d(ctx, V("model.encoder.prenet.proj.bias"), 1, H));
  x = ggml_cont(ctx, ggml_transpose(ctx, x));                // [192,L]
  ggml_tensor* out = ggml_add(ctx, x, x_org);                // residual [192,L]

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);

  std::vector<int64_t> tk = ri64(argv[2]);
  std::vector<int32_t> tk32(L); for (int i = 0; i < L; i++) tk32[i] = (int32_t)tk[i];
  memcpy(ids->data, tk32.data(), L * 4);

  ggml_graph_compute_with_ctx(ctx, gf, 4);

  // ref prenet [1,192,L] row-major (c*L+t); out ggml [192,L] ne0=192 (t*192+c)
  std::vector<float> ref = rf32(argv[3]);
  const float* od = (const float*)out->data;
  double err = 0, ss = 0; int cnt = 0;
  for (int c = 0; c < H; c++) for (int t = 0; t < L; t++) {
    float gg = od[t * H + c], rr = ref[c * L + t];
    err += std::fabs(gg - rr); ss += std::fabs(rr); cnt++;
  }
  printf("prenet: mean|err|=%.6f rel=%.5f over %d\n", err / cnt, err / (ss + 1e-9), cnt);
  return 0;
}
