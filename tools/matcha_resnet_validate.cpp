// Validate the Matcha CFM-decoder ResnetBlock1D in ggml against the numpy reference
// (scripts/gen_resnet_ref.py). Core decoder unit:
//   h = Mish(GroupNorm8(Conv1d_k3(x[160,T], block1)))        160->256
//   h = h + Linear(Mish(t_emb), mlp.1)                        time conditioning (broadcast over T)
//   h = Mish(GroupNorm8(Conv1d_k3(h, block2)))               256->256
//   out = h + Conv1d_k1(x, res_conv)                          160->256 skip
// GroupNorm8: x[T,C] -> reshape [T*C/8, 8] -> ggml_norm(ne0) -> reshape -> *gamma[C]+beta[C].
//
// Build: g++ -std=c++17 matcha_resnet_validate.cpp -I<ggml>/include -L<lib> -lggml -lggml-base -lggml-cpu -o rv
// Usage: rv gguf res_x.f32 res_temb.f32 res_ref.f32 T
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
  if (argc < 6) { printf("usage: rv gguf x.f32 temb.f32 ref.f32 T\n"); return 1; }
  const int T = atoi(argv[5]);
  const int CIN = 160, C = 256, G = 8;
  ggml_context* wc = nullptr; gguf_init_params gp{ false, &wc }; gguf_init_from_file(argv[1], gp);
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* t = ggml_get_first_tensor(wc); t; t = ggml_get_next_tensor(wc, t)) W[ggml_get_name(t)] = t;
  std::string P = "model.decoder.estimator.down_blocks.0.0.";
  auto V = [&](const std::string& s) { return W.at(P + s); };

  ggml_init_params ip{ (size_t)256 * 1024 * 1024, nullptr, false }; ggml_context* c = ggml_init(ip);
  auto f16 = [&](ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(c, w, GGML_TYPE_F16) : w; };
  auto r1 = [&](ggml_tensor* t) { return ggml_reshape_1d(c, t, ggml_nelements(t)); };
  // mish(x) = x*tanh(softplus(x)); softplus(x)=log(1+exp(x)) (ggml has no softplus here).
  auto mish = [&](ggml_tensor* x) {
    ggml_tensor* ones = ggml_new_tensor(c, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne);
    ones = ggml_scale(c, ggml_exp(c, x), 0.0f);   // zeros of x's shape
    ones = ggml_add1(c, ones, ggml_new_f32(c, 1.0f));  // -> ones (shape of x)
    ggml_tensor* sp = ggml_log(c, ggml_add(c, ggml_exp(c, x), ones));
    return ggml_mul(c, x, ggml_tanh(c, sp));
  };
  // conv1d: x[T,Cin] -> [T,Cout]
  auto conv = [&](ggml_tensor* x, const std::string& w, const std::string& b, int pad, int cout) {
    ggml_tensor* y = ggml_conv_1d(c, f16(V(w)), x, 1, pad, 1);
    return ggml_add(c, y, ggml_reshape_2d(c, V(b), 1, cout));
  };
  // GroupNorm(8): x[T,256] -> [T*32,8] -> norm -> [T,256] -> *gamma+beta (gamma/beta [256])
  auto gn8 = [&](ggml_tensor* x, const std::string& gw, const std::string& gb) {
    ggml_tensor* xr = ggml_reshape_2d(c, ggml_cont(c, x), T * C / G, G);
    xr = ggml_norm(c, xr, 1e-5f);
    ggml_tensor* xn = ggml_reshape_2d(c, xr, T, C);            // [T,256]
    ggml_tensor* g = ggml_reshape_2d(c, r1(V(gw)), 1, C);      // [1,256]
    ggml_tensor* b = ggml_reshape_2d(c, r1(V(gb)), 1, C);
    return ggml_add(c, ggml_mul(c, xn, g), b);
  };

  ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, CIN);   // [T,160]
  ggml_tensor* temb = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);

  ggml_tensor* h = mish(gn8(conv(x, "block1.block.0.weight", "block1.block.0.bias", 1, C),
                            "block1.block.block.1_2.weight", "block1.block.block.1_2.bias"));
  // time cond: Linear(mish(temb), mlp.1) [256] -> add (broadcast over T)
  ggml_tensor* mw = ggml_reshape_2d(c, V("mlp.1.weight"), 1024, C);  // ne=[1024,256]
  ggml_tensor* tc = ggml_add(c, ggml_mul_mat(c, mw, mish(temb)), r1(V("mlp.1.bias")));  // [256]
  h = ggml_add(c, h, ggml_reshape_2d(c, tc, 1, C));                 // [T,256] + [1,256]
  h = mish(gn8(conv(h, "block2.block.0.weight", "block2.block.0.bias", 1, C),
               "block2.block.block.1_2.weight", "block2.block.block.1_2.bias"));
  ggml_tensor* out = ggml_add(c, h, conv(x, "res_conv.weight", "res_conv.bias", 0, C));  // [T,256]

  ggml_tensor* outT = ggml_cont(c, ggml_transpose(c, out));         // [256,T] to match ref [256,T]
  std::vector<float> xv = rf(argv[2]), tv = rf(argv[3]);
  // x ref is [160,T] (c*T+t); our x is [T,160] (ne0=T, t + T*c) -> same as transpose. fill [T,160]:
  std::vector<float> xb((size_t)T * CIN);
  for (int ci = 0; ci < CIN; ci++) for (int t = 0; t < T; t++) xb[t + (size_t)T * ci] = xv[ci * T + t];
  memcpy(x->data, xb.data(), xb.size() * 4);
  memcpy(temb->data, tv.data(), tv.size() * 4);

  ggml_cgraph* gf = ggml_new_graph(c); ggml_build_forward_expand(gf, outT);
  ggml_graph_compute_with_ctx(c, gf, 4);

  std::vector<float> ref = rf(argv[4]);  // [256,T] (c*T+t)
  const float* od = (const float*)outT->data;  // [256,T] ne0=256 -> (c,t) at t*256+c
  double e = 0, s = 0;
  for (int ci = 0; ci < C; ci++) for (int t = 0; t < T; t++) { e += std::fabs(od[t * C + ci] - ref[ci * T + t]); s += std::fabs(ref[ci * T + t]); }
  printf("resnet block1d: rel=%.5f over %d\n", e / (s + 1e-9), C * T);
  return 0;
}
