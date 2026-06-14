// Staged validator for one Matcha-TTS encoder transformer layer (RoPE attn + FFN).
// Validates against ONNX layer-0 output norm_2.0 (rel 9e-5).
//
// Post-norm transformer layer (n_heads=2, head_dim=96, PARTIAL rotary 48/96):
//   Q/K/V = conv_{q,k,v}(x) (k1 = Linear)              x[192,L] -> [192,L]
//   reshape [96,2,L]; partial-RoPE(Q),(K): rotate first 48 dims (rotate_half 24/24)
//     via baked cos/sin tables, pass through last 48
//   SDPA: softmax(QK^T / sqrt(96)) V                   (openvoice2.cpp pattern)
//   merge heads -> conv_o -> + residual -> norm_1 (channel-LayerNorm folded gamma/beta)
//   FFN: conv_1(192->768,k3) -> ReLU -> conv_2(768->192,k3) -> + residual -> norm_2
//
// rotate_half(r) = concat(-r[24:48], r[0:24]);  r_out = r*cos + rotate_half(r)*sin
//
// Build (CPU): g++ -std=c++17 matcha_attn_validate.cpp -I<ggml>/include -L<lib> \
//   -lggml -lggml-base -lggml-cpu -o av
// Usage: av gguf prenet_in.f32 cos.f32 sin.f32 norm2_ref.f32 L
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

static std::vector<float> rf(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
  fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 7) { printf("usage: av gguf prenet_in.f32 cos.f32 sin.f32 norm1_ref.f32 L\n"); return 1; }
  const int L = atoi(argv[6]);
  // n_heads=2, head_dim=96, PARTIAL rotary: only first ROT=48 dims rotated (rotate_half 24/24).
  const int H = 192, NH = 2, HD = 96, ROT = 48, RH = 24;

  ggml_context* wctx = nullptr;
  gguf_init_params gp{ false, &wctx };
  if (!gguf_init_from_file(argv[1], gp)) { fprintf(stderr, "gguf fail\n"); return 1; }
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* c = ggml_get_first_tensor(wctx); c; c = ggml_get_next_tensor(wctx, c)) W[ggml_get_name(c)] = c;
  auto V = [&](const std::string& s) -> ggml_tensor* {
    auto it = W.find(s); if (it == W.end()) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); } return it->second; };

  ggml_init_params ip{ (size_t)256 * 1024 * 1024, nullptr, false };
  ggml_context* ctx = ggml_init(ip);
  auto r1 = [&](ggml_tensor* t) { return ggml_reshape_1d(ctx, t, ggml_nelements(t)); };

  // inputs
  ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, L);    // prenet_in [192,L] ne0=192
  ggml_tensor* cosT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROT, L); // [48,L]
  ggml_tensor* sinT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROT, L);

  std::string E = "model.encoder.encoder.";
  // Linear via k1-conv weight [1,192,192] (ne=[k,in,out]) -> [in,out] for mul_mat
  auto proj = [&](const std::string& nm) {
    ggml_tensor* w = ggml_reshape_2d(ctx, V(E + nm + ".weight"), H, H);  // [in,out]
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);                            // [out,L]
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, V(E + nm + ".bias"), H, 1));
  };
  ggml_tensor* Q = proj("attn_layers.0.conv_q");  // [192,L]
  ggml_tensor* K = proj("attn_layers.0.conv_k");
  ggml_tensor* Vv = proj("attn_layers.0.conv_v");

  // reshape to heads [96,2,L]
  auto heads = [&](ggml_tensor* t) { return ggml_reshape_3d(ctx, t, HD, NH, L); };
  // partial RoPE: rotate first ROT=48 dims (rotate_half 24/24), pass through last 48.
  // cos/sin tables are [48,L] -> [48,1,L] broadcast over heads.
  ggml_tensor* cos3 = ggml_reshape_3d(ctx, cosT, ROT, 1, L);
  ggml_tensor* sin3 = ggml_reshape_3d(ctx, sinT, ROT, 1, L);
  auto rope = [&](ggml_tensor* t3) {  // t3 [96,2,L]
    ggml_tensor* t = ggml_cont(ctx, t3);
    size_t nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2];
    ggml_tensor* r  = ggml_cont(ctx, ggml_view_3d(ctx, t, ROT, NH, L, nb1, nb2, 0));          // [0:48] rotary
    ggml_tensor* ps = ggml_cont(ctx, ggml_view_3d(ctx, t, HD - ROT, NH, L, nb1, nb2, ROT * nb0)); // [48:96] pass
    size_t rb1 = r->nb[1], rb2 = r->nb[2], rb0 = r->nb[0];
    ggml_tensor* a = ggml_view_3d(ctx, r, RH, NH, L, rb1, rb2, 0);          // [0:24]
    ggml_tensor* b = ggml_view_3d(ctx, r, RH, NH, L, rb1, rb2, RH * rb0);   // [24:48]
    ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, b)), ggml_cont(ctx, a), 0); // [-b,a]
    ggml_tensor* r_out = ggml_add(ctx, ggml_mul(ctx, r, cos3), ggml_mul(ctx, rot, sin3));
    return ggml_concat(ctx, r_out, ps, 0);  // [96,2,L]
  };
  ggml_tensor* Qr = rope(heads(Q));   // [96,2,L]
  ggml_tensor* Kr = rope(heads(K));
  ggml_tensor* V3 = heads(Vv);

  // SDPA (openvoice2 pattern): permute [96,2,L]->[96,L,2]
  ggml_tensor* Qp = ggml_cont(ctx, ggml_permute(ctx, Qr, 0, 2, 1, 3));  // [96,L,2]
  ggml_tensor* Kp = ggml_cont(ctx, ggml_permute(ctx, Kr, 0, 2, 1, 3));
  ggml_tensor* Vp = ggml_cont(ctx, ggml_permute(ctx, V3, 0, 2, 1, 3));
  ggml_tensor* QK = ggml_mul_mat(ctx, Kp, Qp);                          // [L_k, L_q, NH]
  QK = ggml_scale(ctx, QK, 1.0f / std::sqrt((float)HD));
  QK = ggml_soft_max(ctx, QK);                                         // over ne0=L_k
  ggml_tensor* Vt = ggml_cont(ctx, ggml_permute(ctx, Vp, 1, 0, 2, 3)); // [L,96,2]
  ggml_tensor* O = ggml_mul_mat(ctx, Vt, QK);                          // [96, L_q, 2]
  O = ggml_cont(ctx, ggml_permute(ctx, O, 0, 2, 1, 3));               // [96,2,L]
  O = ggml_reshape_2d(ctx, O, H, L);                                  // [192,L]
  // conv_o (k1 Linear)
  ggml_tensor* ow = ggml_reshape_2d(ctx, V(E + "attn_layers.0.conv_o.weight"), H, H);
  O = ggml_mul_mat(ctx, ow, O);
  O = ggml_add(ctx, O, ggml_reshape_2d(ctx, V(E + "attn_layers.0.conv_o.bias"), H, 1));
  // residual + norm_1 (channel-LayerNorm over ne0=192)
  ggml_tensor* res = ggml_add(ctx, x, O);
  ggml_tensor* n = ggml_norm(ctx, res, 1e-4f);
  n = ggml_mul(ctx, n, r1(V(E + "norm_layers_1.0.Mul_1.weight")));
  n = ggml_add(ctx, n, r1(V(E + "norm_layers_1.0.Add_1.weight")));   // post-attn [192,L]

  // FFN: conv_1 (192->768,k3,p1) -> ReLU -> conv_2 (768->192,k3,p1); + residual; norm_2
  auto f16 = [&](ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(ctx, w, GGML_TYPE_F16) : w; };
  ggml_tensor* fx = ggml_cont(ctx, ggml_transpose(ctx, n));          // [L,192]
  ggml_tensor* f1w = V(E + "ffn_layers.0.conv_1.weight");            // [768,192,3]
  fx = ggml_conv_1d(ctx, f16(f1w), fx, 1, 1, 1);                     // [L,768]
  fx = ggml_add(ctx, fx, ggml_reshape_2d(ctx, V(E + "ffn_layers.0.conv_1.bias"), 1, 768));
  fx = ggml_relu(ctx, fx);
  ggml_tensor* f2w = V(E + "ffn_layers.0.conv_2.weight");            // [192,768,3]
  fx = ggml_conv_1d(ctx, f16(f2w), fx, 1, 1, 1);                     // [L,192]
  fx = ggml_add(ctx, fx, ggml_reshape_2d(ctx, V(E + "ffn_layers.0.conv_2.bias"), 1, H));
  fx = ggml_cont(ctx, ggml_transpose(ctx, fx));                     // [192,L]
  ggml_tensor* res2 = ggml_add(ctx, n, fx);
  ggml_tensor* n2 = ggml_norm(ctx, res2, 1e-4f);
  n2 = ggml_mul(ctx, n2, r1(V(E + "norm_layers_2.0.Mul_1.weight")));
  n2 = ggml_add(ctx, n2, r1(V(E + "norm_layers_2.0.Add_1.weight")));  // layer-0 out [192,L]

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, n2);

  // fill inputs: prenet_in ref [1,192,L] (c*L+t) -> x[192,L] ne0=192 (t*192+c)
  std::vector<float> pin = rf(argv[2]);
  std::vector<float> xb((size_t)H * L);
  for (int c = 0; c < H; c++) for (int t = 0; t < L; t++) xb[t * H + c] = pin[c * L + t];
  memcpy(x->data, xb.data(), xb.size() * 4);
  std::vector<float> cb = rf(argv[3]), sb = rf(argv[4]);  // [L,48] row-major = [48,L] ne0=48
  memcpy(cosT->data, cb.data(), cb.size() * 4);
  memcpy(sinT->data, sb.data(), sb.size() * 4);

  ggml_graph_compute_with_ctx(ctx, gf, 4);

  std::vector<float> ref = rf(argv[5]);  // [1,192,L] (c*L+t)
  const float* od = (const float*)n2->data;  // [192,L] ne0=192 (t*192+c)
  double err = 0, ss = 0; int cnt = 0;
  for (int c = 0; c < H; c++) for (int t = 0; t < L; t++) {
    err += std::fabs(od[t * H + c] - ref[c * L + t]); ss += std::fabs(ref[c * L + t]); cnt++;
  }
  printf("layer0 out norm_2: mean|err|=%.6f rel=%.5f over %d\n", err / cnt, err / (ss + 1e-9), cnt);
  return 0;
}
