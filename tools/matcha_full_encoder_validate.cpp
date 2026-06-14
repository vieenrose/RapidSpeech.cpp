// Full Matcha-TTS text encoder validator (ggml): emb -> prenet -> 6x transformer
// -> proj_m (mel mean mu) and proj_w (duration logw). Validates both vs ONNX.
//
// Reuses the validated sub-blocks: ConvReluNorm prenet, RoPE transformer layer
// (n_heads=2, head_dim=96, partial rotary 48/96, post-norm). RoPE cos/sin is shared
// across layers (position-deterministic), reused from the layer-0 baked tables.
//
// Build (CPU): g++ -std=c++17 matcha_full_encoder_validate.cpp -I<ggml>/include -L<lib> \
//   -lggml -lggml-base -lggml-cpu -o fe
// Usage: fe gguf tokens.i64 cos.f32 sin.f32 mu_ref.f32 logw_ref.f32 L
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
static std::vector<int64_t> ri(const char* p) {
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<int64_t> v(n / 8); if (fread(v.data(), 8, v.size(), f) != v.size()) exit(1);
  fclose(f); return v;
}

int main(int argc, char** argv) {
  if (argc < 8) { printf("usage: fe gguf tokens.i64 cos.f32 sin.f32 mu_ref.f32 logw_ref.f32 L\n"); return 1; }
  const int L = atoi(argv[7]);
  const int H = 192, NH = 2, HD = 96, ROT = 48, RH = 24;

  ggml_context* wctx = nullptr;
  gguf_init_params gp{ false, &wctx };
  if (!gguf_init_from_file(argv[1], gp)) { fprintf(stderr, "gguf fail\n"); return 1; }
  std::map<std::string, ggml_tensor*> W;
  for (ggml_tensor* c = ggml_get_first_tensor(wctx); c; c = ggml_get_next_tensor(wctx, c)) W[ggml_get_name(c)] = c;
  auto V = [&](const std::string& s) -> ggml_tensor* {
    auto it = W.find(s); if (it == W.end()) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); } return it->second; };

  ggml_init_params ip{ (size_t)768 * 1024 * 1024, nullptr, false };
  ggml_context* ctx = ggml_init(ip);
  auto r1 = [&](ggml_tensor* t) { return ggml_reshape_1d(ctx, t, ggml_nelements(t)); };
  auto f16 = [&](ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(ctx, w, GGML_TYPE_F16) : w; };
  // channel-LayerNorm over ne0; gamma/beta folded as ne=[1,C,1]
  auto cln = [&](ggml_tensor* x, const std::string& g, const std::string& b) {
    x = ggml_norm(ctx, x, 1e-4f); x = ggml_mul(ctx, x, r1(V(g))); return ggml_add(ctx, x, r1(V(b)));
  };
  // conv1d helper: x[T,Cin] -> [T,Cout]
  auto conv = [&](ggml_tensor* x, const std::string& w, const std::string& b, int pad, int cout) {
    ggml_tensor* y = ggml_conv_1d(ctx, f16(V(w)), x, 1, pad, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, V(b), 1, cout));
  };

  ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
  ggml_tensor* cosT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROT, L);
  ggml_tensor* sinT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ROT, L);
  ggml_tensor* cos3 = ggml_reshape_3d(ctx, cosT, ROT, 1, L);
  ggml_tensor* sin3 = ggml_reshape_3d(ctx, sinT, ROT, 1, L);

  // --- embedding x sqrt(192) ---
  ggml_tensor* emb = ggml_scale(ctx, ggml_get_rows(ctx, V("model.encoder.emb.weight"), ids), std::sqrt((float)H)); // [192,L]
  // --- prenet (ConvReluNorm) --- work in [T,192]
  ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, emb));  // [L,192]
  for (int i = 0; i < 3; i++) {
    std::string p = "model.encoder.prenet.", n = p + "norm_layers." + std::to_string(i) + ".";
    x = conv(x, p + "conv_layers." + std::to_string(i) + ".weight", p + "conv_layers." + std::to_string(i) + ".bias", 2, H);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));   // [192,L]
    x = cln(x, n + "Mul_1.weight", n + "Add_1.weight");
    x = ggml_relu(ctx, x);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));   // [L,192]
  }
  x = conv(x, "model.encoder.prenet.proj.weight", "model.encoder.prenet.proj.bias", 0, H); // [L,192]
  x = ggml_cont(ctx, ggml_transpose(ctx, x));     // [192,L]
  x = ggml_add(ctx, x, emb);                       // residual [192,L]

  // --- 6 transformer layers ---
  auto rope = [&](ggml_tensor* t3) {
    ggml_tensor* t = ggml_cont(ctx, t3);
    size_t nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2];
    ggml_tensor* r  = ggml_cont(ctx, ggml_view_3d(ctx, t, ROT, NH, L, nb1, nb2, 0));
    ggml_tensor* ps = ggml_cont(ctx, ggml_view_3d(ctx, t, HD - ROT, NH, L, nb1, nb2, ROT * nb0));
    size_t rb0 = r->nb[0], rb1 = r->nb[1], rb2 = r->nb[2];
    ggml_tensor* a = ggml_view_3d(ctx, r, RH, NH, L, rb1, rb2, 0);
    ggml_tensor* b = ggml_view_3d(ctx, r, RH, NH, L, rb1, rb2, RH * rb0);
    ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, b)), ggml_cont(ctx, a), 0);
    ggml_tensor* ro = ggml_add(ctx, ggml_mul(ctx, r, cos3), ggml_mul(ctx, rot, sin3));
    return ggml_concat(ctx, ro, ps, 0);
  };
  for (int l = 0; l < 6; l++) {
    std::string E = "model.encoder.encoder.", li = std::to_string(l);
    auto proj = [&](const std::string& nm) {
      ggml_tensor* w = ggml_reshape_2d(ctx, V(E + nm + ".weight"), H, H);
      return ggml_add(ctx, ggml_mul_mat(ctx, w, x), ggml_reshape_2d(ctx, V(E + nm + ".bias"), H, 1));
    };
    ggml_tensor* Q = ggml_reshape_3d(ctx, proj("attn_layers." + li + ".conv_q"), HD, NH, L);
    ggml_tensor* K = ggml_reshape_3d(ctx, proj("attn_layers." + li + ".conv_k"), HD, NH, L);
    ggml_tensor* Vv = ggml_reshape_3d(ctx, proj("attn_layers." + li + ".conv_v"), HD, NH, L);
    ggml_tensor* Qp = ggml_cont(ctx, ggml_permute(ctx, rope(Q), 0, 2, 1, 3));  // [96,L,2]
    ggml_tensor* Kp = ggml_cont(ctx, ggml_permute(ctx, rope(K), 0, 2, 1, 3));
    ggml_tensor* Vp = ggml_cont(ctx, ggml_permute(ctx, Vv, 0, 2, 1, 3));
    ggml_tensor* QK = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, Kp, Qp), 1.0f / std::sqrt((float)HD)));
    ggml_tensor* Vt = ggml_cont(ctx, ggml_permute(ctx, Vp, 1, 0, 2, 3));        // [L,96,2]
    ggml_tensor* O = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, Vt, QK), 0, 2, 1, 3)); // [96,2,L]
    O = ggml_reshape_2d(ctx, O, H, L);
    ggml_tensor* ow = ggml_reshape_2d(ctx, V(E + "attn_layers." + li + ".conv_o.weight"), H, H);
    O = ggml_add(ctx, ggml_mul_mat(ctx, ow, O), ggml_reshape_2d(ctx, V(E + "attn_layers." + li + ".conv_o.bias"), H, 1));
    x = cln(ggml_add(ctx, x, O), E + "norm_layers_1." + li + ".Mul_1.weight", E + "norm_layers_1." + li + ".Add_1.weight");
    // FFN
    ggml_tensor* fx = ggml_cont(ctx, ggml_transpose(ctx, x));   // [L,192]
    fx = ggml_relu(ctx, conv(fx, E + "ffn_layers." + li + ".conv_1.weight", E + "ffn_layers." + li + ".conv_1.bias", 1, 768));
    fx = conv(fx, E + "ffn_layers." + li + ".conv_2.weight", E + "ffn_layers." + li + ".conv_2.bias", 1, H);
    fx = ggml_cont(ctx, ggml_transpose(ctx, fx));               // [192,L]
    x = cln(ggml_add(ctx, x, fx), E + "norm_layers_2." + li + ".Mul_1.weight", E + "norm_layers_2." + li + ".Add_1.weight");
  }
  // --- proj_m (mel mean) : conv k1 192->80 --- x is [192,L]; conv wants [L,192]
  ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));     // [L,192]
  ggml_tensor* mu = conv(xt, "model.encoder.proj_m.weight", "model.encoder.proj_m.bias", 0, 80); // [L,80]
  mu = ggml_cont(ctx, ggml_transpose(ctx, mu));                 // [80,L]
  // --- proj_w (duration) : conv k3 192->256 -> ReLU -> norm -> conv k3 256 -> ReLU -> norm -> proj k1 256->1
  ggml_tensor* dw = ggml_relu(ctx, conv(xt, "model.encoder.proj_w.conv_1.weight", "model.encoder.proj_w.conv_1.bias", 1, 256)); // [L,256]
  dw = ggml_cont(ctx, ggml_transpose(ctx, dw));                 // [256,L]
  dw = cln(dw, "model.encoder.proj_w.norm_1.Mul_1.weight", "model.encoder.proj_w.norm_1.Add_1.weight");
  dw = ggml_cont(ctx, ggml_transpose(ctx, dw));                 // [L,256]
  dw = ggml_relu(ctx, conv(dw, "model.encoder.proj_w.conv_2.weight", "model.encoder.proj_w.conv_2.bias", 1, 256));
  dw = ggml_cont(ctx, ggml_transpose(ctx, dw));                 // [256,L]
  dw = cln(dw, "model.encoder.proj_w.norm_2.Mul_1.weight", "model.encoder.proj_w.norm_2.Add_1.weight");
  dw = ggml_cont(ctx, ggml_transpose(ctx, dw));                 // [L,256]
  ggml_tensor* logw = conv(dw, "model.encoder.proj_w.proj.weight", "model.encoder.proj_w.proj.bias", 0, 1); // [L,1]

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, mu);
  ggml_build_forward_expand(gf, logw);

  std::vector<int64_t> tk = ri(argv[2]);
  std::vector<int32_t> t32(L); for (int i = 0; i < L; i++) t32[i] = (int32_t)tk[i];
  memcpy(ids->data, t32.data(), L * 4);
  std::vector<float> cb = rf(argv[3]), sb = rf(argv[4]);
  memcpy(cosT->data, cb.data(), cb.size() * 4); memcpy(sinT->data, sb.data(), sb.size() * 4);

  ggml_graph_compute_with_ctx(ctx, gf, 4);

  // compare mu [80,L] (ggml ne0=80, t*80+c) vs ref [1,80,L] (c*L+t)
  std::vector<float> rmu = rf(argv[5]);
  const float* md = (const float*)mu->data;
  double e1 = 0, s1 = 0;
  for (int c = 0; c < 80; c++) for (int t = 0; t < L; t++) { e1 += std::fabs(md[t * 80 + c] - rmu[c * L + t]); s1 += std::fabs(rmu[c * L + t]); }
  printf("proj_m (mu):   rel=%.5f\n", e1 / (s1 + 1e-9));
  // compare logw [L,1] (ggml ne0=L? logw is [L,1] ne0=L) vs ref [1,1,L]
  std::vector<float> rlw = rf(argv[6]);
  const float* ld = (const float*)logw->data;  // [L,1] ne0=L -> element t at t
  double e2 = 0, s2 = 0;
  for (int t = 0; t < L; t++) { e2 += std::fabs(ld[t] - rlw[t]); s2 += std::fabs(rlw[t]); }
  printf("proj_w (logw): rel=%.5f\n", e2 / (s2 + 1e-9));
  return 0;
}
