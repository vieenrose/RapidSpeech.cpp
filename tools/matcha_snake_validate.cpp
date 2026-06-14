// Validate the Matcha CFM-decoder SnakeBeta activation in ggml against a numpy reference.
// snakebeta(x) = x + (1/(exp(log_beta)+eps)) * sin^2(exp(log_alpha) * x)
// (the stored alpha/beta are in log space, hence exp; eps=1e-9). Applied per channel.
//
// Build (CPU): g++ -std=c++17 matcha_snake_validate.cpp -I<ggml>/include -L<lib> \
//   -lggml -lggml-base -lggml-cpu -o sn
// Usage: sn gguf snake_x.f32 snake_ref.f32 D T
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

static std::vector<float> rf(const char* p) {
  FILE* f = fopen(p, "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1); fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: sn gguf x.f32 ref.f32 D T\n"); return 1; }
  const int D = atoi(argv[3]), T = atoi(argv[4]);
  ggml_context* wctx = nullptr; gguf_init_params gp{ false, &wctx };
  gguf_init_from_file(argv[1], gp);
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* c = ggml_get_first_tensor(wctx); c; c = ggml_get_next_tensor(wctx, c)) W[ggml_get_name(c)] = c;
  auto V = [&](const std::string& s) { return W.at(s); };

  ggml_init_params ip{ (size_t)64 * 1024 * 1024, nullptr, false }; ggml_context* ctx = ggml_init(ip);
  ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);  // [D,T] ne0=D
  std::string B = "model.decoder.estimator.down_blocks.0.1.0.ff.net.0.";
  // exp(log_alpha/beta) on host (old ggml's elementwise-on-gguf-leaf is unstable), broadcast to [D,T]
  const float* la = (const float*)V(B + "alpha")->data;
  const float* lb = (const float*)V(B + "beta")->data;
  ggml_tensor* alpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
  ggml_tensor* invb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
  for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) {
    ((float*)alpha->data)[t * D + d] = std::exp(la[d]);
    ((float*)invb->data)[t * D + d] = 1.0f / std::exp(lb[d]);   // avoid ggml_div
  }
  // snakebeta = x + sin(alpha*x)^2 * (1/exp(beta))
  ggml_tensor* s = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, alpha)));   // [D,T]
  ggml_tensor* out = ggml_add(ctx, x, ggml_mul(ctx, s, invb));

  std::vector<float> xv = rf(argv[2]);  // [D,T] row-major = ne0=D
  memcpy(x->data, xv.data(), xv.size() * 4);
  ggml_cgraph* gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, out);
  ggml_graph_compute_with_ctx(ctx, gf, 4);

  std::vector<float> ref = rf(argv[3]);
  const float* od = (const float*)out->data;
  double e = 0, s2 = 0; for (size_t i = 0; i < ref.size(); i++) { e += std::fabs(od[i] - ref[i]); s2 += std::fabs(ref[i]); }
  printf("snakebeta: rel=%.6f over %zu\n", e / (s2 + 1e-9), ref.size());
  return 0;
}
