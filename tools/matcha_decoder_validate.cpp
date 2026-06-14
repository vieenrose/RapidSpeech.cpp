// Validate the full Matcha CFM decoder in ggml against ONNX (decoder-isolation).
// Input: mu_expanded[80,T] (dec_mu_in.f32, T=internal padded length), x0=0 (noise_scale=0).
// Runs 3-step Euler ODE over the UNet estimator; dumps staged checkpoints + final mel.
// Specs (workflow-verified): transformer 1e-6, time-emb 5e-6, ODE/length-reg/down-up confirmed.
//
// Layout: running tensor x is [T,C] (ne0=T) for convs (my validated convention). Transformer
// transposes to [C,T] (ne0=C) internally. GroupNorm8 via reshape[T*C/8,8]->norm->reshape.
//
// Build: g++ -std=c++17 matcha_decoder_validate.cpp -I<ggml>/include -L<lib> -lggml -lggml-base -lggml-cpu -o dv
// Usage: dv gguf mu_in.f32 T cp_time.f32 cp_res00.f32 cp_tfm00.f32 cp_down0.f32 mel_ref.f32 y_len
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
  FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<float> v(n / 4); if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1); fclose(f); return v;
}
static const float PI = 3.14159265358979323846f;

struct Dec {
  ggml_context* c;
  std::map<std::string, ggml_tensor*> W;
  std::string E = "model.decoder.estimator.";
  ggml_tensor* V(const std::string& s) { auto it = W.find(E + s); if (it == W.end()) { fprintf(stderr, "missing %s\n", s.c_str()); exit(1); } return it->second; }
  ggml_tensor* f16(ggml_tensor* w) { return w->type == GGML_TYPE_F32 ? ggml_cast(c, w, GGML_TYPE_F16) : w; }
  ggml_tensor* r1(ggml_tensor* t) { return ggml_reshape_1d(c, t, ggml_nelements(t)); }
  ggml_tensor* mish(ggml_tensor* x) {  // x*tanh(log(1+exp(x)))
    ggml_tensor* ones = ggml_add1(c, ggml_scale(c, ggml_exp(c, x), 0.0f), ggml_new_f32(c, 1.0f));
    return ggml_mul(c, x, ggml_tanh(c, ggml_log(c, ggml_add(c, ggml_exp(c, x), ones))));
  }
  ggml_tensor* silu(ggml_tensor* x) { return ggml_silu(c, x); }
  // conv1d: x[T,Cin]->[T',Cout]
  ggml_tensor* conv(ggml_tensor* x, const std::string& w, const std::string& b, int stride, int pad, int cout) {
    ggml_tensor* y = ggml_conv_1d(c, f16(V(w)), x, stride, pad, 1);
    return ggml_add(c, y, ggml_reshape_2d(c, V(b), 1, cout));
  }
  // GroupNorm(8): x[T,C]->reshape[T*C/8,8]->norm->reshape->*g+b (g/b [C], _2 affine)
  ggml_tensor* gn8(ggml_tensor* x, int T, int C, const std::string& aff) {
    ggml_tensor* xr = ggml_norm(c, ggml_reshape_2d(c, ggml_cont(c, x), T * C / 8, 8), 1e-5f);
    ggml_tensor* xn = ggml_reshape_2d(c, xr, T, C);
    ggml_tensor* g = ggml_reshape_2d(c, r1(V(aff + ".weight")), 1, C);
    ggml_tensor* bb = ggml_reshape_2d(c, r1(V(aff + ".bias")), 1, C);
    return ggml_add(c, ggml_mul(c, xn, g), bb);
  }
  // channel-LayerNorm over ne0: x[C,T]
  ggml_tensor* lnC(ggml_tensor* x, const std::string& nm) {
    ggml_tensor* y = ggml_norm(c, x, 1e-5f);
    return ggml_add(c, ggml_mul(c, y, V(nm + ".weight")), V(nm + ".bias"));
  }
  ggml_tensor* msk(ggml_tensor* x, ggml_tensor* m) { return m ? ggml_mul(c, x, m) : x; }
  // ResnetBlock1D(x[T,Cin], temb[1024], mult-mask mm[T]) -> [T,256]
  ggml_tensor* resnet(ggml_tensor* x, ggml_tensor* temb, const std::string& p, int T, int cin, ggml_tensor* mm) {
    auto blk = [&](ggml_tensor* in, const std::string& bp) {
      ggml_tensor* h = conv(msk(in, mm), bp + ".block.0.weight", bp + ".block.0.bias", 1, 1, 256);
      h = gn8(h, T, 256, bp + ".block.block.1_2");
      return msk(mish(h), mm);
    };
    ggml_tensor* h = blk(x, p + ".block1");
    ggml_tensor* mw = ggml_reshape_2d(c, V(p + ".mlp.1.weight"), 1024, 256);
    ggml_tensor* tc = ggml_add(c, ggml_mul_mat(c, mw, mish(temb)), r1(V(p + ".mlp.1.bias")));
    h = ggml_add(c, h, ggml_reshape_2d(c, tc, 1, 256));
    h = blk(h, p + ".block2");
    ggml_tensor* skip = conv(msk(x, mm), p + ".res_conv.weight", p + ".res_conv.bias", 1, 0, 256);
    return ggml_add(c, h, skip);
  }
  // TransformerBlock(x[T,256], attn-bias ab[T], mult-mask mm[T]) -> [T,256]  (2 heads x 64)
  ggml_tensor* transformer(ggml_tensor* x, int T, const std::string& p, ggml_tensor* ab, ggml_tensor* mm) {
    ggml_tensor* xc = ggml_cont(c, ggml_transpose(c, x));  // [256,T]
    // _2-suffixed matmul weights are stored ne=[out,in] -> transpose before mul_mat
    auto mmT = [&](const std::string& w, ggml_tensor* in) { return ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, V(w))), in); };
    // attn
    ggml_tensor* h = lnC(xc, p + ".norm1");
    ggml_tensor* q = mmT(p + ".attn1.to_q_2.weight", h);   // [128,T]
    ggml_tensor* k = mmT(p + ".attn1.to_k_2.weight", h);
    ggml_tensor* v = mmT(p + ".attn1.to_v_2.weight", h);
    auto heads = [&](ggml_tensor* t) { return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, t, 64, 2, T), 0, 2, 1, 3)); }; // [64,T,2]
    ggml_tensor* qp = heads(q), *kp = heads(k), *vp = heads(v);
    ggml_tensor* sc = ggml_scale(c, ggml_mul_mat(c, kp, qp), 1.0f / 8.0f);  // [T_k,T_q,2]
    if (ab) sc = ggml_add(c, sc, ab);  // additive key-mask bias [T_k] broadcast over T_q,heads
    ggml_tensor* qk = ggml_soft_max(c, sc);  // [T,T,2]
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vp, 1, 0, 2, 3));  // [T,64,2]
    ggml_tensor* o = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, vt, qk), 0, 2, 1, 3));  // [64,2,T]
    o = ggml_reshape_2d(c, o, 128, T);                                                    // [128,T]
    o = ggml_add(c, mmT(p + ".attn1.to_out.0_2.weight", o), V(p + ".attn1.to_out.0.bias"));  // [256,T]
    xc = ggml_add(c, xc, o);
    // ff
    ggml_tensor* f = lnC(xc, p + ".norm3");
    f = ggml_add(c, mmT(p + ".ff.net.0.proj_2.weight", f), V(p + ".ff.net.0.proj.bias"));  // [1024,T]
    // snakebeta (host-precompute a=exp(alpha), inv_b=1/exp(beta))
    const float* la = (const float*)V(p + ".ff.net.0.alpha")->data;
    const float* lb = (const float*)V(p + ".ff.net.0.beta")->data;
    ggml_tensor* a = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
    ggml_tensor* ib = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
    for (int i = 0; i < 1024; i++) { ((float*)a->data)[i] = std::exp(la[i]); ((float*)ib->data)[i] = 1.0f / std::exp(lb[i]); }
    ggml_tensor* s = ggml_sqr(c, ggml_sin(c, ggml_mul(c, f, a)));
    f = ggml_add(c, f, ggml_mul(c, s, ib));
    f = ggml_add(c, mmT(p + ".ff.net.2_2.weight", f), V(p + ".ff.net.2.bias"));  // [256,T]
    xc = ggml_add(c, xc, f);
    (void)mm;  // transformer does NOT mask its output (only resnets do); padding frames stay nonzero
    return ggml_cont(c, ggml_transpose(c, xc));  // [T,256]
  }
};

int main(int argc, char** argv) {
  if (argc < 10) { printf("usage: dv gguf mu.f32 T cp_time cp_res00 cp_tfm00 cp_down0 mel_ref y_len\n"); return 1; }
  const int T = atoi(argv[3]), YLEN = atoi(argv[9]);
  ggml_context* wc = nullptr; gguf_init_params gp{ false, &wc }; gguf_init_from_file(argv[1], gp);
  Dec d;
  ggml_init_params ip{ (size_t)5 * 1024 * 1024 * 1024ull, nullptr, false }; d.c = ggml_init(ip);
  for (ggml_tensor* t = ggml_get_first_tensor(wc); t; t = ggml_get_next_tensor(wc, t)) d.W[ggml_get_name(t)] = t;
  ggml_context* c = d.c;

  // mu input [80,T] (numpy c*T+t) -> ggml [T,80] (ne0=T, t + T*ci)
  ggml_tensor* mu = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, 80);
  std::vector<float> muv = rf(argv[2]);
  for (int ci = 0; ci < 80; ci++) for (int t = 0; t < T; t++) ((float*)mu->data)[t + (size_t)T * ci] = muv[ci * T + t];

  // time embeddings for t=0,1/3,2/3: sinusoid concat[sin(1000*t*freqs),cos(...)], 80 each; then time_mlp
  auto time_emb_in = [&](float t) {
    ggml_tensor* e = ggml_new_tensor_1d(c, GGML_TYPE_F32, 160);
    for (int i = 0; i < 80; i++) {
      float freq = std::exp(-(float)i * std::log(10000.0f) / 79.0f);
      float ang = 1000.0f * t * freq;
      ((float*)e->data)[i] = std::sin(ang); ((float*)e->data)[80 + i] = std::cos(ang);
    }
    return e;
  };
  auto time_mlp = [&](ggml_tensor* ein) {
    ggml_tensor* h = ggml_add(c, ggml_mul_mat(c, d.V("time_mlp.linear_1.weight"), ein), d.r1(d.V("time_mlp.linear_1.bias")));
    h = d.silu(h);
    return ggml_add(c, ggml_mul_mat(c, d.V("time_mlp.linear_2.weight"), h), d.r1(d.V("time_mlp.linear_2.bias")));
  };
  ggml_tensor* temb[3] = { time_mlp(time_emb_in(0.0f)), time_mlp(time_emb_in(1.0f / 3)), time_mlp(time_emb_in(2.0f / 3)) };

  // masks: full-res T (valid YLEN), half-res T2 (valid ceil(YLEN/2)). mult (1/0) + attn-bias (0/-1e9).
  int T2 = (T - 1) / 2 + 1, YH = (YLEN + 1) / 2;
  auto mk_mult = [&](int n, int valid) { ggml_tensor* m = ggml_new_tensor_1d(c, GGML_TYPE_F32, n); for (int i = 0; i < n; i++) ((float*)m->data)[i] = i < valid ? 1.0f : 0.0f; return m; };
  auto mk_bias = [&](int n, int valid) { ggml_tensor* m = ggml_new_tensor_1d(c, GGML_TYPE_F32, n); for (int i = 0; i < n; i++) ((float*)m->data)[i] = i < valid ? 0.0f : -1e9f; return m; };
  ggml_tensor* mF = mk_mult(T, YLEN), *mH = mk_mult(T2, YH), *aF = mk_bias(T, YLEN), *aH = mk_bias(T2, YH);

  // estimator(x_t[T,80], temb) -> vector field [T,80]
  ggml_tensor* dbg_res00 = nullptr, *dbg_tfm00 = nullptr, *dbg_down0 = nullptr;
  auto estimator = [&](ggml_tensor* xt, ggml_tensor* te, bool dump) {
    ggml_tensor* x = ggml_concat(c, xt, mu, 1);  // [T,160]
    x = d.msk(x, mF);
    // down.0
    x = d.resnet(x, te, "down_blocks.0.0", T, 160, mF);
    if (dump) dbg_res00 = x;
    x = d.transformer(x, T, "down_blocks.0.1.0", nullptr, mF);
    if (dump) dbg_tfm00 = x;
    ggml_tensor* skip0 = x;
    x = d.conv(d.msk(x, mF), "down_blocks.0.2.conv.weight", "down_blocks.0.2.conv.bias", 2, 1, 256);  // [T2,256]
    if (dump) dbg_down0 = x;
    // down.1
    x = d.resnet(x, te, "down_blocks.1.0", T2, 256, mH);
    x = d.transformer(x, T2, "down_blocks.1.1.0", nullptr, mH);
    ggml_tensor* skip1 = x;
    x = d.conv(d.msk(x, mH), "down_blocks.1.2.weight", "down_blocks.1.2.bias", 1, 1, 256);
    // mid
    x = d.resnet(x, te, "mid_blocks.0.0", T2, 256, mH); x = d.transformer(x, T2, "mid_blocks.0.1.0", nullptr, mH);
    x = d.resnet(x, te, "mid_blocks.1.0", T2, 256, mH); x = d.transformer(x, T2, "mid_blocks.1.1.0", nullptr, mH);
    // up.0 (concat skip1)
    x = ggml_concat(c, x, skip1, 1);  // [T2,512]
    x = d.resnet(x, te, "up_blocks.0.0", T2, 512, mH);
    x = d.transformer(x, T2, "up_blocks.0.1.0", nullptr, mH);
    ggml_tensor* up = ggml_conv_transpose_1d(c, d.f16(d.V("up_blocks.0.2.conv.weight")), d.msk(x, mH), 2, 0, 1);
    up = ggml_add(c, up, ggml_reshape_2d(c, d.V("up_blocks.0.2.conv.bias"), 1, 256));
    x = ggml_cont(c, ggml_view_2d(c, up, T, 256, up->nb[1], 1 * up->nb[0]));  // crop [1:1+T]
    // up.1 (concat skip0)
    x = ggml_concat(c, x, skip0, 1);  // [T,512]
    x = d.resnet(x, te, "up_blocks.1.0", T, 512, mF);
    x = d.transformer(x, T, "up_blocks.1.1.0", nullptr, mF);
    x = d.conv(d.msk(x, mF), "up_blocks.1.2.weight", "up_blocks.1.2.bias", 1, 1, 256);
    // tail
    x = d.mish(d.gn8(d.conv(d.msk(x, mF), "final_block.block.0.weight", "final_block.block.0.bias", 1, 1, 256), T, 256, "final_block.block.block.1_2"));
    x = d.conv(d.msk(x, mF), "final_proj.weight", "final_proj.bias", 1, 0, 80);  // [T,80]
    return d.msk(x, mF);
  };

  // 3-step Euler ODE, x0 = 0
  ggml_tensor* x = ggml_scale(c, mu, 0.0f);  // [T,80] zeros (shape of mu)
  const float dt = 1.0f / 3.0f;
  for (int s = 0; s < 3; s++) {
    ggml_tensor* v = estimator(x, temb[s], s == 0);
    x = ggml_add(c, x, ggml_scale(c, v, dt));
  }
  // slice to y_len, affine: mel = x[:YLEN]*5.446792 + (-2.9521978)
  ggml_tensor* xl = ggml_cont(c, ggml_view_2d(c, x, YLEN, 80, x->nb[1], 0));  // [YLEN,80]
  ggml_tensor* mel = ggml_add1(c, ggml_scale(c, xl, 5.446792f), ggml_new_f32(c, -2.9521978f));  // [YLEN,80]

  ggml_cgraph* gf = ggml_new_graph_custom(c, 131072, false);
  ggml_build_forward_expand(gf, mel);
  ggml_build_forward_expand(gf, temb[0]);
  if (dbg_res00) ggml_build_forward_expand(gf, dbg_res00);
  if (dbg_tfm00) ggml_build_forward_expand(gf, dbg_tfm00);
  if (dbg_down0) ggml_build_forward_expand(gf, dbg_down0);
  ggml_graph_compute_with_ctx(c, gf, 4);

  // got is ggml [Tt,C] (ne0=Tt): element (ci,t) at flat t + Tt*ci == ci*Tt + t.
  // ref_CT layout = numpy [C,Tt] (ci*Tt+t)  -> flat-equal.  ref_TC layout = numpy [Tt,C] (t*C+ci).
  auto cmp = [&](const char* nm, ggml_tensor* got, const char* refpath, int C, int Tt, bool ref_TC) {
    std::vector<float> ref = rf(refpath);
    const float* g = (const float*)got->data;
    double e = 0, s = 0;
    for (int ci = 0; ci < C; ci++) for (int t = 0; t < Tt; t++) {
      float gv = g[t + (size_t)Tt * ci];
      float rv = ref_TC ? ref[(size_t)t * C + ci] : ref[(size_t)ci * Tt + t];
      e += std::fabs(gv - rv); s += std::fabs(rv);
    }
    printf("  %-10s rel=%.5f over %d\n", nm, e / (s + 1e-9), C * Tt);
  };
  { std::vector<float> ref = rf(argv[4]); const float* g = (const float*)temb[0]->data; double e=0,s=0; for(int i=0;i<1024;i++){e+=std::fabs(g[i]-ref[i]);s+=std::fabs(ref[i]);} printf("  %-10s rel=%.5f\n","time_mlp",e/(s+1e-9)); }
  if (dbg_res00) cmp("res00", dbg_res00, argv[5], 256, T, false);            // ref [256,T] (C,T)
  if (dbg_tfm00) cmp("tfm00", dbg_tfm00, argv[6], 256, T, true);             // ref [T,256] (T,C)
  if (dbg_down0) cmp("down0", dbg_down0, argv[7], 256, (T - 1) / 2 + 1, false);
  cmp("MEL", mel, argv[8], 80, YLEN, false);                                 // ref [80,YLEN] (C,T)
  return 0;
}
