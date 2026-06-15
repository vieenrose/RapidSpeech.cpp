// Matcha-TTS arch for RapidSpeech.cpp — see matcha.h.
//
// FULLY VALIDATED against ONNX, component-by-component (docs/MATCHA_TTS_GGML_PORT.md):
//   text encoder (mu+durations) rel<=1e-4 | length regulator rel 0 | CFM decoder mel corr 0.999993
//   | Vocos+iSTFT corr 1.0.  See tools/matcha_*_validate.cpp for the staged validators.
//
// Pipeline (two-phase, because the decoder length is duration-derived at runtime):
//   PHASE A: encoder graph -> mu[80,L], logw[L].  Compute, read values.
//   length regulator (CPU): durations=ceil(exp(logw))*length_scale; y_len=sum; internal_T=ceil(y_len/4)*4;
//     mu_expanded[80,internal_T] = streaming gather of mu by durations.
//   PHASE B: CFM decoder (3-step Euler ODE over the UNet estimator) -> mel[80,y_len]
//     -> Vocos ConvNeXt -> iSTFT -> waveform.
//
// Layout (ggml ne[] reverse of numpy): running tensor [T,C] (ne0=T) for convs; transformer transposes
// to [C,T]. GroupNorm8 via reshape[T*C/8,8]->norm->reshape. Linear weights: PyTorch/Gemm weights load
// ne=[in,out] (no transpose); folded onnx::MatMul "_2" weights load ne=[out,in] (transpose before mul_mat).
#include "arch/matcha.h"
#include "frontend/matcha_frontend.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <utility>
#include <cstdint>

namespace {
constexpr float PI = 3.14159265358979323846f;
ggml_tensor* f16(ggml_context* c, ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(c, w, GGML_TYPE_F16) : w; }
ggml_tensor* r1(ggml_context* c, ggml_tensor* t) { return ggml_reshape_1d(c, t, ggml_nelements(t)); }
// conv1d data [T,Cin] -> [T',Cout]
ggml_tensor* conv1d(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride, int pad, int cout) {
  ggml_tensor* y = ggml_conv_1d(c, f16(c, w), x, stride, pad, 1);
  return b ? ggml_add(c, y, ggml_reshape_2d(c, b, 1, cout)) : y;
}
// channel-LayerNorm over ne0=C
ggml_tensor* cln(ggml_context* c, ggml_tensor* x, ggml_tensor* g, ggml_tensor* b) {
  x = ggml_norm(c, x, 1e-5f); x = ggml_mul(c, x, r1(c, g)); return ggml_add(c, x, r1(c, b));
}
// mish = x*tanh(softplus(x)), softplus(x)=log(1+exp(x)). `one` is a scalar [1] tensor (a registered
// input — ggml_new_f32 writes data and is illegal in the no_alloc graph-build context).
ggml_tensor* mish(ggml_context* c, ggml_tensor* x, ggml_tensor* one) {
  ggml_tensor* sp = ggml_log(c, ggml_add1(c, ggml_exp(c, x), one));
  return ggml_mul(c, x, ggml_tanh(c, sp));
}
ggml_tensor* msk(ggml_context* c, ggml_tensor* x, ggml_tensor* m) { return m ? ggml_mul(c, x, m) : x; }
}  // namespace

ggml_tensor* MatchaModel::W(const std::string& name) const {
  auto it = w_.find(name);
  if (it == w_.end()) { RS_LOG_ERR("matcha: missing weight %s", name.c_str()); return nullptr; }
  return it->second;
}

// --- backend-sched input helpers: create a no_alloc graph input + queue its host data ---
ggml_tensor* MatchaModel::inp_f32(ggml_context* c, int ne0, int ne1, const float* data) {
  ggml_tensor* t = ne1 > 0 ? ggml_new_tensor_2d(c, GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(c, GGML_TYPE_F32, ne0);
  ggml_set_input(t);
  const uint8_t* p = (const uint8_t*)data;
  pending_.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
  return t;
}
ggml_tensor* MatchaModel::inp_i32(ggml_context* c, int ne0, const int32_t* data) {
  ggml_tensor* t = ggml_new_tensor_1d(c, GGML_TYPE_I32, ne0);
  ggml_set_input(t);
  const uint8_t* p = (const uint8_t*)data;
  pending_.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
  return t;
}
void MatchaModel::flush_inputs() {
  for (auto& pi : pending_) ggml_backend_tensor_set(pi.t, pi.data.data(), 0, pi.data.size());
  pending_.clear();
}

// =====================================================================
// Text encoder: ids -> mu [80,L], logw [L,1]  (validated rel<=1e-4)
// =====================================================================
ggml_tensor* MatchaModel::build_encoder(ggml_context* c, ggml_tensor* ids, int L,
                                        ggml_tensor* cosT, ggml_tensor* sinT, ggml_tensor** logw_out) {
  const int H = hp_.hidden, NH = hp_.n_heads, HD = hp_.head_dim, ROT = hp_.rotary_dim, RH = ROT / 2;
  auto Wt = [&](const std::string& s) { return W(s); };
  ggml_tensor* cos3 = ggml_reshape_3d(c, cosT, ROT, 1, L);
  ggml_tensor* sin3 = ggml_reshape_3d(c, sinT, ROT, 1, L);
  ggml_tensor* emb = ggml_scale(c, ggml_get_rows(c, Wt("model.encoder.emb.weight"), ids), std::sqrt((float)H));
  ggml_tensor* x = ggml_cont(c, ggml_transpose(c, emb));
  for (int i = 0; i < 3; i++) {
    std::string p = "model.encoder.prenet.", n = p + "norm_layers." + std::to_string(i) + ".";
    x = conv1d(c, x, Wt(p + "conv_layers." + std::to_string(i) + ".weight"), Wt(p + "conv_layers." + std::to_string(i) + ".bias"), 1, 2, H);
    x = ggml_cont(c, ggml_transpose(c, x));
    x = cln(c, x, Wt(n + "Mul_1.weight"), Wt(n + "Add_1.weight"));
    x = ggml_relu(c, x);
    x = ggml_cont(c, ggml_transpose(c, x));
  }
  x = conv1d(c, x, Wt("model.encoder.prenet.proj.weight"), Wt("model.encoder.prenet.proj.bias"), 1, 0, H);
  x = ggml_cont(c, ggml_transpose(c, x));
  x = ggml_add(c, x, emb);
  auto rope = [&](ggml_tensor* t3) {
    ggml_tensor* t = ggml_cont(c, t3);
    size_t nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2];
    ggml_tensor* r = ggml_cont(c, ggml_view_3d(c, t, ROT, NH, L, nb1, nb2, 0));
    ggml_tensor* ps = ggml_cont(c, ggml_view_3d(c, t, HD - ROT, NH, L, nb1, nb2, (size_t)ROT * nb0));
    size_t rb0 = r->nb[0], rb1 = r->nb[1], rb2 = r->nb[2];
    ggml_tensor* a = ggml_view_3d(c, r, RH, NH, L, rb1, rb2, 0);
    ggml_tensor* b = ggml_view_3d(c, r, RH, NH, L, rb1, rb2, (size_t)RH * rb0);
    ggml_tensor* rot = ggml_concat(c, ggml_neg(c, ggml_cont(c, b)), ggml_cont(c, a), 0);
    ggml_tensor* ro = ggml_add(c, ggml_mul(c, r, cos3), ggml_mul(c, rot, sin3));
    return ggml_concat(c, ro, ps, 0);
  };
  for (int l = 0; l < hp_.n_layers; l++) {
    std::string E = "model.encoder.encoder.", li = std::to_string(l);
    auto proj = [&](const std::string& nm) {
      ggml_tensor* w = ggml_reshape_2d(c, Wt(E + nm + ".weight"), H, H);
      return ggml_add(c, ggml_mul_mat(c, w, x), ggml_reshape_2d(c, Wt(E + nm + ".bias"), H, 1));
    };
    ggml_tensor* Q = ggml_reshape_3d(c, proj("attn_layers." + li + ".conv_q"), HD, NH, L);
    ggml_tensor* K = ggml_reshape_3d(c, proj("attn_layers." + li + ".conv_k"), HD, NH, L);
    ggml_tensor* Vv = ggml_reshape_3d(c, proj("attn_layers." + li + ".conv_v"), HD, NH, L);
    ggml_tensor* Qp = ggml_cont(c, ggml_permute(c, rope(Q), 0, 2, 1, 3));
    ggml_tensor* Kp = ggml_cont(c, ggml_permute(c, rope(K), 0, 2, 1, 3));
    ggml_tensor* Vp = ggml_cont(c, ggml_permute(c, Vv, 0, 2, 1, 3));
    ggml_tensor* QK = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, Kp, Qp), 1.0f / std::sqrt((float)HD)));
    ggml_tensor* Vt = ggml_cont(c, ggml_permute(c, Vp, 1, 0, 2, 3));
    ggml_tensor* O = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, Vt, QK), 0, 2, 1, 3));
    O = ggml_reshape_2d(c, O, H, L);
    ggml_tensor* ow = ggml_reshape_2d(c, Wt(E + "attn_layers." + li + ".conv_o.weight"), H, H);
    O = ggml_add(c, ggml_mul_mat(c, ow, O), ggml_reshape_2d(c, Wt(E + "attn_layers." + li + ".conv_o.bias"), H, 1));
    x = cln(c, ggml_add(c, x, O), Wt(E + "norm_layers_1." + li + ".Mul_1.weight"), Wt(E + "norm_layers_1." + li + ".Add_1.weight"));
    ggml_tensor* fx = ggml_cont(c, ggml_transpose(c, x));
    fx = ggml_relu(c, conv1d(c, fx, Wt(E + "ffn_layers." + li + ".conv_1.weight"), Wt(E + "ffn_layers." + li + ".conv_1.bias"), 1, 1, hp_.filter));
    fx = conv1d(c, fx, Wt(E + "ffn_layers." + li + ".conv_2.weight"), Wt(E + "ffn_layers." + li + ".conv_2.bias"), 1, 1, H);
    fx = ggml_cont(c, ggml_transpose(c, fx));
    x = cln(c, ggml_add(c, x, fx), Wt(E + "norm_layers_2." + li + ".Mul_1.weight"), Wt(E + "norm_layers_2." + li + ".Add_1.weight"));
  }
  ggml_tensor* xt = ggml_cont(c, ggml_transpose(c, x));
  ggml_tensor* mu = ggml_cont(c, ggml_transpose(c, conv1d(c, xt, Wt("model.encoder.proj_m.weight"), Wt("model.encoder.proj_m.bias"), 1, 0, hp_.n_mels)));
  ggml_tensor* dw = ggml_relu(c, conv1d(c, xt, Wt("model.encoder.proj_w.conv_1.weight"), Wt("model.encoder.proj_w.conv_1.bias"), 1, 1, 256));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = cln(c, dw, Wt("model.encoder.proj_w.norm_1.Mul_1.weight"), Wt("model.encoder.proj_w.norm_1.Add_1.weight"));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = ggml_relu(c, conv1d(c, dw, Wt("model.encoder.proj_w.conv_2.weight"), Wt("model.encoder.proj_w.conv_2.bias"), 1, 1, 256));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = cln(c, dw, Wt("model.encoder.proj_w.norm_2.Mul_1.weight"), Wt("model.encoder.proj_w.norm_2.Add_1.weight"));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  *logw_out = conv1d(c, dw, Wt("model.encoder.proj_w.proj.weight"), Wt("model.encoder.proj_w.proj.bias"), 1, 0, 1);
  return mu;
}

// =====================================================================
// CFM decoder (validated mel corr 0.999993). mu_exp[T,80], T=internal padded len, ylen=valid len.
// Returns mel [ylen,80] (sliced + affine de-normalized).
// =====================================================================
ggml_tensor* MatchaModel::build_cfm(ggml_context* c, ggml_tensor* mu, int T, float noise_scale) {
  (void)noise_scale;  // deterministic x0=0 (noise_scale handled by caller)
  const std::string E = "model.decoder.estimator.";
  auto V = [&](const std::string& s) { return W(E + s); };
  int T2 = (T - 1) / 2 + 1, ylen = cfm_ylen_, YH = (ylen + 1) / 2;
  auto mk = [&](int n, int valid) { std::vector<float> v(n); for (int i = 0; i < n; i++) v[i] = i < valid ? 1.0f : 0.0f; return inp_f32(c, n, 0, v.data()); };
  ggml_tensor* mF = mk(T, ylen), *mH = mk(T2, YH);  // mult masks (registered inputs)
  const float ONEV = 1.0f; ggml_tensor* one = inp_f32(c, 1, 0, &ONEV);  // scalar for mish softplus
  auto gn8 = [&](ggml_tensor* x, int t, const std::string& aff) {
    ggml_tensor* xr = ggml_norm(c, ggml_reshape_2d(c, ggml_cont(c, x), t * 256 / 8, 8), 1e-5f);
    ggml_tensor* xn = ggml_reshape_2d(c, xr, t, 256);
    return ggml_add(c, ggml_mul(c, xn, ggml_reshape_2d(c, r1(c, V(aff + ".weight")), 1, 256)), ggml_reshape_2d(c, r1(c, V(aff + ".bias")), 1, 256));
  };
  auto resnet = [&](ggml_tensor* x, ggml_tensor* te, const std::string& p, int t, ggml_tensor* mm) {
    auto blk = [&](ggml_tensor* in, const std::string& bp) {
      return msk(c, mish(c, gn8(conv1d(c, msk(c, in, mm), V(bp + ".block.0.weight"), V(bp + ".block.0.bias"), 1, 1, 256), t, bp + ".block.block.1_2"), one), mm);
    };
    ggml_tensor* h = blk(x, p + ".block1");
    ggml_tensor* mw = ggml_reshape_2d(c, V(p + ".mlp.1.weight"), 1024, 256);
    ggml_tensor* tc = ggml_add(c, ggml_mul_mat(c, mw, mish(c, te, one)), r1(c, V(p + ".mlp.1.bias")));
    h = ggml_add(c, h, ggml_reshape_2d(c, tc, 1, 256));
    h = blk(h, p + ".block2");
    return ggml_add(c, h, conv1d(c, msk(c, x, mm), V(p + ".res_conv.weight"), V(p + ".res_conv.bias"), 1, 0, 256));
  };
  auto transformer = [&](ggml_tensor* x, int t, const std::string& p) {
    ggml_tensor* xc = ggml_cont(c, ggml_transpose(c, x));
    auto mmT = [&](const std::string& w, ggml_tensor* in) { return ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, V(w))), in); };
    ggml_tensor* h = cln(c, xc, V(p + ".norm1.weight"), V(p + ".norm1.bias"));
    ggml_tensor* q = mmT(p + ".attn1.to_q_2.weight", h), *k = mmT(p + ".attn1.to_k_2.weight", h), *v = mmT(p + ".attn1.to_v_2.weight", h);
    auto hd = [&](ggml_tensor* z) { return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, z, 64, 2, t), 0, 2, 1, 3)); };
    ggml_tensor* qp = hd(q), *kp = hd(k), *vp = hd(v);
    ggml_tensor* qk = ggml_soft_max(c, ggml_scale(c, ggml_mul_mat(c, kp, qp), 1.0f / 8.0f));
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vp, 1, 0, 2, 3));
    ggml_tensor* o = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, vt, qk), 0, 2, 1, 3)), 128, t);
    o = ggml_add(c, mmT(p + ".attn1.to_out.0_2.weight", o), V(p + ".attn1.to_out.0.bias"));
    xc = ggml_add(c, xc, o);
    ggml_tensor* f = cln(c, xc, V(p + ".norm3.weight"), V(p + ".norm3.bias"));
    f = ggml_add(c, mmT(p + ".ff.net.0.proj_2.weight", f), V(p + ".ff.net.0.proj.bias"));
    // SnakeBeta in-graph (backend-safe — no host read of weight ->data):
    //   a = exp(log_alpha);  inv_b = 1/exp(log_beta) = exp(-log_beta)
    ggml_tensor* a = ggml_exp(c, V(p + ".ff.net.0.alpha"));
    ggml_tensor* ib = ggml_exp(c, ggml_neg(c, V(p + ".ff.net.0.beta")));
    f = ggml_add(c, f, ggml_mul(c, ggml_sqr(c, ggml_sin(c, ggml_mul(c, f, a))), ib));
    f = ggml_add(c, mmT(p + ".ff.net.2_2.weight", f), V(p + ".ff.net.2.bias"));
    return ggml_cont(c, ggml_transpose(c, ggml_add(c, xc, f)));
  };
  auto estimator = [&](ggml_tensor* xt, ggml_tensor* te) {
    ggml_tensor* x = msk(c, ggml_concat(c, xt, mu, 1), mF);
    x = resnet(x, te, "down_blocks.0.0", T, mF); x = transformer(x, T, "down_blocks.0.1.0");
    ggml_tensor* skip0 = x;
    x = conv1d(c, msk(c, x, mF), V("down_blocks.0.2.conv.weight"), V("down_blocks.0.2.conv.bias"), 2, 1, 256);
    x = resnet(x, te, "down_blocks.1.0", T2, mH); x = transformer(x, T2, "down_blocks.1.1.0");
    ggml_tensor* skip1 = x;
    x = conv1d(c, msk(c, x, mH), V("down_blocks.1.2.weight"), V("down_blocks.1.2.bias"), 1, 1, 256);
    x = resnet(x, te, "mid_blocks.0.0", T2, mH); x = transformer(x, T2, "mid_blocks.0.1.0");
    x = resnet(x, te, "mid_blocks.1.0", T2, mH); x = transformer(x, T2, "mid_blocks.1.1.0");
    x = ggml_concat(c, x, skip1, 1);
    x = resnet(x, te, "up_blocks.0.0", T2, mH); x = transformer(x, T2, "up_blocks.0.1.0");
    ggml_tensor* up = ggml_add(c, ggml_conv_transpose_1d(c, f16(c, V("up_blocks.0.2.conv.weight")), msk(c, x, mH), 2, 0, 1), ggml_reshape_2d(c, V("up_blocks.0.2.conv.bias"), 1, 256));
    x = ggml_cont(c, ggml_view_2d(c, up, T, 256, up->nb[1], 1 * up->nb[0]));
    x = ggml_concat(c, x, skip0, 1);
    x = resnet(x, te, "up_blocks.1.0", T, mF); x = transformer(x, T, "up_blocks.1.1.0");
    x = conv1d(c, msk(c, x, mF), V("up_blocks.1.2.weight"), V("up_blocks.1.2.bias"), 1, 1, 256);
    x = mish(c, gn8(conv1d(c, msk(c, x, mF), V("final_block.block.0.weight"), V("final_block.block.0.bias"), 1, 1, 256), T, "final_block.block.block.1_2"), one);
    return msk(c, conv1d(c, msk(c, x, mF), V("final_proj.weight"), V("final_proj.bias"), 1, 0, 80), mF);
  };
  // sinusoidal time-emb (host const -> registered input) -> time_mlp; 3-step Euler ODE
  auto temb = [&](float t) {
    std::vector<float> ev(160);
    for (int i = 0; i < 80; i++) { float fr = std::exp(-(float)i * std::log(10000.0f) / 79.0f), an = 1000.0f * t * fr; ev[i] = std::sin(an); ev[80 + i] = std::cos(an); }
    ggml_tensor* e = inp_f32(c, 160, 0, ev.data());
    ggml_tensor* h = ggml_silu(c, ggml_add(c, ggml_mul_mat(c, V("time_mlp.linear_1.weight"), e), r1(c, V("time_mlp.linear_1.bias"))));
    return ggml_add(c, ggml_mul_mat(c, V("time_mlp.linear_2.weight"), h), r1(c, V("time_mlp.linear_2.bias")));
  };
  ggml_tensor* x = ggml_scale(c, mu, 0.0f);  // x0=0 (noise_scale=0)
  const float dt = 1.0f / 3.0f;
  for (int s = 0; s < 3; s++) x = ggml_add(c, x, ggml_scale(c, estimator(x, temb((float)s / 3.0f)), dt));
  ggml_tensor* xl = ggml_cont(c, ggml_view_2d(c, x, ylen, 80, x->nb[1], 0));  // slice to ylen
  const float NEGV = -2.9521978f; ggml_tensor* negsh = inp_f32(c, 1, 0, &NEGV);
  return ggml_add1(c, ggml_scale(c, xl, 5.446792f), negsh);  // [ylen,80] mel
}

// =====================================================================
// Vocos ConvNeXt -> spectral head [514,T]  (validated corr 1.0). mel [T,80].
// =====================================================================
ggml_tensor* MatchaModel::build_vocos(ggml_context* c, ggml_tensor* mel, int T) {
  (void)T;
  const int C = 384, NB = 8;
  auto V = [&](const std::string& s) { return W("voc." + s); };
  auto lin = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int out) {
    ggml_tensor* wt = ggml_cont(c, ggml_transpose(c, w));
    ggml_tensor* y = ggml_mul_mat(c, wt, x);
    return b ? ggml_add(c, y, ggml_reshape_2d(c, b, out, 1)) : y;
  };
  ggml_tensor* h = ggml_add(c, ggml_conv_1d(c, f16(c, V("conv_pre.weight")), mel, 1, 3, 1), ggml_reshape_2d(c, V("conv_pre.bias"), 1, C));
  h = ggml_cont(c, ggml_transpose(c, h));
  h = cln(c, h, V("norm_in.weight"), V("norm_in.bias"));
  h = ggml_cont(c, ggml_transpose(c, h));
  for (int i = 0; i < NB; i++) {
    std::string b = "blocks." + std::to_string(i) + ".";
    ggml_tensor* res = h;
    ggml_tensor* x = ggml_add(c, ggml_conv_1d_dw(c, f16(c, V(b + "dw.weight")), h, 1, 3, 1), ggml_reshape_2d(c, V(b + "dw.bias"), 1, C));
    x = ggml_cont(c, ggml_transpose(c, x));
    x = cln(c, x, V(b + "norm.weight"), V(b + "norm.bias"));
    x = lin(x, V(b + "pw1.weight"), V(b + "pw1.bias"), 1152);
    x = ggml_gelu_erf(c, x);
    x = lin(x, V(b + "pw2.weight"), V(b + "pw2.bias"), C);
    x = ggml_mul(c, x, V(b + "gamma"));
    x = ggml_cont(c, ggml_transpose(c, x));
    h = ggml_add(c, x, res);
  }
  h = ggml_cont(c, ggml_transpose(c, h));
  h = cln(c, h, V("norm_out.weight"), V("norm_out.bias"));
  return lin(h, V("head.weight"), V("head.bias"), 514);
}

namespace {
// in-place iterative radix-2 FFT (n = power of 2). sign=-1 forward, +1 inverse (no 1/N scaling).
void fft_radix2(std::vector<double>& re, std::vector<double>& im, int n, int sign) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = sign * 2.0 * PI / len, wr = std::cos(ang), wi = std::sin(ang);
    for (int i = 0; i < n; i += len) {
      double cr = 1.0, ci = 0.0;
      for (int j = 0; j < len / 2; j++) {
        double a = re[i + j], b = im[i + j];
        double vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
        double vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
        re[i + j] = a + vr; im[i + j] = b + vi;
        re[i + j + len / 2] = a - vr; im[i + j + len / 2] = b - vi;
        double nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
}  // namespace

// iSTFT (validated corr 1.0): head[514,T] -> waveform.
// Per frame: build the conjugate-symmetric 512-pt spectrum from mag/phase, irfft via one radix-2
// IFFT (O(N log N), was O(N^2) DFT — the dominant cost), overlap-add with window-sum normalization.
std::vector<float> MatchaModel::istft(const float* od, int T) const {
  const int NFFT = hp_.n_fft, HOP = hp_.hop_length, BINS = NFFT / 2 + 1;
  std::vector<float> win(NFFT);
  for (int n = 0; n < NFFT; n++) win[n] = 0.5f - 0.5f * std::cos(2.0 * PI * n / NFFT);
  int Lp = (T - 1) * HOP + NFFT;
  std::vector<double> out(Lp, 0.0), wsum(Lp, 0.0);
  std::vector<double> re(NFFT), im(NFFT);
  for (int f = 0; f < T; f++) {
    // build full-length conjugate-symmetric spectrum from the one-sided bins
    for (int k = 0; k < BINS; k++) {
      float mlog = od[f * 514 + k]; if (mlog > 9.0f) mlog = 9.0f;
      double mag = std::exp((double)mlog), ph = od[f * 514 + BINS + k];
      re[k] = mag * std::cos(ph); im[k] = mag * std::sin(ph);
    }
    for (int k = BINS; k < NFFT; k++) { re[k] = re[NFFT - k]; im[k] = -im[NFFT - k]; }  // Hermitian
    fft_radix2(re, im, NFFT, +1);  // inverse transform; real part / N is the time-domain frame
    for (int n = 0; n < NFFT; n++) {
      double v = re[n] / NFFT;
      out[f * HOP + n] += v * win[n]; wsum[f * HOP + n] += (double)win[n] * win[n];
    }
  }
  int wavn = Lp - NFFT, start = NFFT / 2;
  std::vector<float> wav(wavn);
  for (int i = 0; i < wavn; i++) { double ww = wsum[start + i] < 1e-8 ? 1.0 : wsum[start + i]; wav[i] = (float)(out[start + i] / ww); }
  return wav;
}

bool MatchaModel::Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) {
  (void)backend;
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) { RS_LOG_ERR("matcha: bad gguf ctx"); return false; }
  rsctx_ = ctx.get();
  gguf_context* g = ctx->ctx_gguf;
  auto ai = [&](const char* k, int32_t& dst) { int64_t key = gguf_find_key(g, k); if (key != -1) dst = gguf_get_val_u32(g, key); };
  ai("matcha.sample_rate", hp_.sample_rate);
  ai("matcha.num_ode_steps", hp_.num_ode_steps);
  ai("matcha.pad_id", hp_.pad_id);
  ai("matcha.hidden", hp_.hidden);
  ai("matcha.n_vocab", hp_.n_vocab);
  const int n = gguf_get_n_tensors(g);
  for (int i = 0; i < n; i++) {
    const char* name = gguf_get_tensor_name(g, i);
    ggml_tensor* t = ggml_get_tensor(ctx->gguf_data, name);
    if (t) w_[name] = t;
  }
  meta_.arch_name = "matcha-tts";
  meta_.audio_sample_rate = hp_.sample_rate;
  meta_.n_mels = hp_.n_mels;
  RS_LOG_INFO("matcha loaded: %zu tensors, sr=%d ode_steps=%d", w_.size(), hp_.sample_rate, hp_.num_ode_steps);
  return true;
}

std::shared_ptr<RSState> MatchaModel::CreateState() { return std::make_shared<MatchaState>(); }

bool MatchaModel::PushText(RSState& state, const char* text, const char* language, const char* instruct) {
  (void)language; (void)instruct;
  auto& st = static_cast<MatchaState&>(state);
  // If the caller didn't pre-load phoneme_ids (e.g. via MATCHA_IDS in the demo),
  // run the text frontend: tokens.txt + lexicon.txt from MATCHA_TOKENS/MATCHA_LEXICON.
  if (const char* idf = getenv("MATCHA_IDS")) {   // test hook: raw int32-LE ids file
    FILE* f = fopen(idf, "rb");
    if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
             st.phoneme_ids.resize(sz / 4); if (fread(st.phoneme_ids.data(), 4, st.phoneme_ids.size(), f) != st.phoneme_ids.size()) st.phoneme_ids.clear(); fclose(f); }
  }
  if (st.phoneme_ids.empty() && text && *text) {
    const char* tok = getenv("MATCHA_TOKENS");
    const char* lex = getenv("MATCHA_LEXICON");
    if (tok && lex) {
      static MatchaFrontend fe; static bool fe_ready = false, fe_tried = false;
      if (!fe_tried) { fe_tried = true; fe_ready = fe.Load(tok, lex); }
      if (fe_ready) {
        int skipped = 0;
        st.phoneme_ids = fe.TextToIds(text, &skipped);
        RS_LOG_INFO("matcha frontend: \"%s\" -> %d ids (%d skipped)",
                    text, (int)st.phoneme_ids.size(), skipped);
      }
    } else {
      RS_LOG_ERR("matcha: text given but no frontend (set MATCHA_TOKENS + MATCHA_LEXICON, or pre-fill phoneme_ids)");
    }
  }
  if (st.phoneme_ids.empty()) { RS_LOG_ERR("matcha: no phoneme_ids (empty text / frontend produced nothing)"); return false; }

  // Clause-split at punctuation terminator tokens (; : , . ! ?) and synthesize each
  // clause separately, concatenating the audio — matches sherpa-onnx. An English word
  // embedded between Chinese (e.g. a name) renders cleanly in a short clause but gets
  // buried/attenuated inside one long mixed sequence.
  std::vector<int32_t> full_ids = st.phoneme_ids;
  std::vector<std::vector<int32_t>> clauses; {
    std::vector<int32_t> cur;
    auto is_term = [](int32_t id) { return id == 2 || id == 3 || id == 4 || id == 5 || id == 6 || id == 7; };
    for (int32_t id : full_ids) { cur.push_back(id); if (is_term(id)) { clauses.push_back(cur); cur.clear(); } }
    if (!cur.empty()) clauses.push_back(cur);
    if (clauses.empty()) clauses.push_back(full_ids);
  }
  std::vector<float> all_audio;
  const int ROT = hp_.rotary_dim, RH = ROT / 2;
  for (auto& clause_ids : clauses) {
  const int L = (int)clause_ids.size();
  std::vector<float> cosv((size_t)ROT * L), sinv((size_t)ROT * L);
  for (int p = 0; p < L; p++) for (int d = 0; d < RH; d++) {
    float freq = std::pow(10000.0f, -2.0f * d / ROT), ang = p * freq;
    cosv[p * ROT + d] = cosv[p * ROT + d + RH] = std::cos(ang);
    sinv[p * ROT + d] = sinv[p * ROT + d + RH] = std::sin(ang);
  }
#ifdef MATCHA_PROFILE
  auto pf = [] { return std::chrono::high_resolution_clock::now(); };
  auto pa0 = pf();
#endif
  ggml_backend_sched_t sched = rsctx_->sched;

  // PHASE A: encoder -> mu[80,L], logw[L]  (backend-sched path — runs on CPU or CUDA backend)
  ggml_context* cA = nullptr; ggml_cgraph* gA = nullptr;
  if (!init_compute_ctx(&cA, &gA, 8192)) return false;
  pending_.clear();
  ggml_tensor* ids = inp_i32(cA, L, clause_ids.data());
  ggml_tensor* cosT = inp_f32(cA, ROT, L, cosv.data());
  ggml_tensor* sinT = inp_f32(cA, ROT, L, sinv.data());
  ggml_tensor* logw = nullptr;
  ggml_tensor* mu = build_encoder(cA, ids, L, cosT, sinT, &logw);
  ggml_set_output(mu); ggml_set_output(logw);
  ggml_build_forward_expand(gA, mu); ggml_build_forward_expand(gA, logw);
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gA)) { RS_LOG_ERR("matcha: sched alloc A failed"); ggml_free(cA); return false; }
  flush_inputs();
  if (ggml_backend_sched_graph_compute(sched, gA) != GGML_STATUS_SUCCESS) { RS_LOG_ERR("matcha: compute A failed"); ggml_free(cA); return false; }
  std::vector<float> mu_raw((size_t)80 * L), logw_data(L);
  ggml_backend_tensor_get(mu, mu_raw.data(), 0, ggml_nbytes(mu));
  ggml_backend_tensor_get(logw, logw_data.data(), 0, ggml_nbytes(logw));
  ggml_free(cA);
  std::vector<float> mu_data((size_t)80 * L);  // mu_raw is [80,L] ggml (ci+80*t) -> [80,L] numpy (ci*L+t)
  for (int ci = 0; ci < 80; ci++) for (int t = 0; t < L; t++) mu_data[ci * L + t] = mu_raw[(size_t)t * 80 + ci];

  // length regulator (validated rel 0): durations=ceil(exp(logw)); w=durations*length_scale
  std::vector<float> w(L); double ysum = 0;
  for (int i = 0; i < L; i++) { w[i] = std::ceil(std::exp(logw_data[i])) * st.length_scale; ysum += w[i]; }
  int ylen = (int)std::max(ysum, 1.0);
  // DEBUG: force exact durations from a file (to isolate ceil-sensitivity from the decoder chain)
  { const char* fp = getenv("MATCHA_FORCE_DUR"); if (fp) { FILE* f = fopen(fp, "rb"); if (f) { std::vector<int> dd(L); if (fread(dd.data(), 4, L, f) == (size_t)L) { ysum = 0; for (int i = 0; i < L; i++) { w[i] = dd[i] * st.length_scale; ysum += w[i]; } ylen = (int)std::max(ysum, 1.0); } fclose(f); } } }
  { std::string ds; for (int i = 0; i < L; i++) ds += std::to_string((int)w[i]) + ","; RS_LOG_INFO("matcha: durations=[%s] ysum=%.4f ylen=%d", ds.c_str(), ysum, ylen); }
  int T = ((ylen + 3) / 4) * 4;  // internal padded length (mult of 4)
  std::vector<double> cum(L); { double a = 0; for (int i = 0; i < L; i++) { a += w[i]; cum[i] = a; } }
  std::vector<float> muexp((size_t)80 * T, 0.0f);  // [80,T] numpy
  { int i = 0; for (int t = 0; t < T; t++) { while (i < L - 1 && t >= cum[i]) i++; if (t < ylen) for (int ci = 0; ci < 80; ci++) muexp[ci * T + t] = mu_data[ci * L + i]; } }

  // PHASE B: decoder (mu_exp) -> mel -> vocos -> head  (backend-sched path); iSTFT on host
  cfm_ylen_ = ylen;
  std::vector<float> me((size_t)T * 80);  // muexp in [T,80] ggml layout (t + T*ci)
  for (int ci = 0; ci < 80; ci++) for (int t = 0; t < T; t++) me[(size_t)t + (size_t)T * ci] = muexp[(size_t)ci * T + t];
#ifdef MATCHA_PROFILE
  auto pb0 = pf();
#endif
  ggml_context* cB = nullptr; ggml_cgraph* gB = nullptr;
  if (!init_compute_ctx(&cB, &gB, 65536)) return false;
  pending_.clear();
  ggml_tensor* muexp_t = inp_f32(cB, T, 80, me.data());
  ggml_tensor* mel = build_cfm(cB, muexp_t, T, st.noise_scale);   // [ylen,80]
  ggml_tensor* head = build_vocos(cB, mel, ylen);                 // [514,ylen]
  ggml_set_output(head);
  ggml_build_forward_expand(gB, head);
  ggml_backend_sched_reset(sched);
#ifdef MATCHA_PROFILE
  auto p0 = pf();
#endif
  if (!ggml_backend_sched_alloc_graph(sched, gB)) { RS_LOG_ERR("matcha: sched alloc B failed"); ggml_free(cB); return false; }
  flush_inputs();
  if (ggml_backend_sched_graph_compute(sched, gB) != GGML_STATUS_SUCCESS) { RS_LOG_ERR("matcha: compute B failed"); ggml_free(cB); return false; }
#ifdef MATCHA_PROFILE
  auto p1 = pf();
#endif
  std::vector<float> head_raw((size_t)514 * ylen);
  ggml_backend_tensor_get(head, head_raw.data(), 0, ggml_nbytes(head));
  ggml_free(cB);
  { auto a = istft(head_raw.data(), ylen); all_audio.insert(all_audio.end(), a.begin(), a.end()); }
#ifdef MATCHA_PROFILE
  auto p2 = pf();
  RS_LOG_INFO("matcha PROFILE: phaseA(enc)=%.1fms  decBuild=%.1fms  decCompute=%.1fms  istft=%.1fms",
              std::chrono::duration<double, std::milli>(pb0 - pa0).count(),
              std::chrono::duration<double, std::milli>(p0 - pb0).count(),
              std::chrono::duration<double, std::milli>(p1 - p0).count(),
              std::chrono::duration<double, std::milli>(p2 - p1).count());
#endif
  }  // for each clause
  st.phoneme_ids = std::move(full_ids);
  st.audio_output = std::move(all_audio);
  st.audio_read_cursor = 0;
  RS_LOG_INFO("matcha: synthesized %zu samples (%.2fs) over %zu clause(s)",
              st.audio_output.size(), st.audio_output.size() / (float)hp_.sample_rate, clauses.size());
  return true;
}

int MatchaModel::GetAudioOutput(RSState& state, float** out_data) {
  auto& st = static_cast<MatchaState&>(state);
  int remain = (int)st.audio_output.size() - st.audio_read_cursor;
  if (remain <= 0) return 0;
  *out_data = st.audio_output.data() + st.audio_read_cursor;
  st.audio_read_cursor += remain;
  return remain;
}

static bool s_reg = [] {
  rs_register_model_arch("matcha-tts", []() { return std::make_shared<MatchaModel>(); });
  return true;
}();
