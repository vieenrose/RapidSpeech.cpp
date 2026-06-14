// Matcha-TTS arch for RapidSpeech.cpp — see matcha.h.
//
// The encoder, vocos and iSTFT graph code here is the numerically-validated logic from the
// staged validators (tools/matcha_{encoder,attn,full_encoder,vocos}_validate.cpp). The CFM
// decoder follows the architecture mapped in docs/MATCHA_TTS_GGML_PORT.md; it is assembled here
// and needs end-to-end validation against ONNX (the staged method applies).
//
// Layout conventions (ggml ne[] reverse of numpy):
//   - conv data [T,C] (ne0=T); conv kernels numpy [Cout,Cin,k] -> ggml ne=[k,Cin,Cout].
//   - Linear weights numpy [in,out] -> ggml ne=[out,in]; transpose before mul_mat.
//   - channel-LayerNorm over ne0; folded gamma/beta are ne=[1,C,1] -> reshape to [C].
#include "arch/matcha.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace {
constexpr float PI = 3.14159265358979323846f;

ggml_tensor* f16(ggml_context* c, ggml_tensor* w) {
  return w->type == GGML_TYPE_F32 ? ggml_cast(c, w, GGML_TYPE_F16) : w;
}
ggml_tensor* r1(ggml_context* c, ggml_tensor* t) { return ggml_reshape_1d(c, t, ggml_nelements(t)); }

// conv1d: x[T,Cin] -> [T,Cout]; weight ggml ne=[k,Cin,Cout]
ggml_tensor* conv1d(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int pad, int cout) {
  ggml_tensor* y = ggml_conv_1d(c, f16(c, w), x, 1, pad, 1);
  if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, 1, cout));
  return y;
}
// channel-LayerNorm over ne0=C; gamma/beta folded ne=[1,C,1]
ggml_tensor* cln(ggml_context* c, ggml_tensor* x, ggml_tensor* g, ggml_tensor* b) {
  x = ggml_norm(c, x, 1e-4f); x = ggml_mul(c, x, r1(c, g)); return ggml_add(c, x, r1(c, b));
}
// Linear (k1 conv weight ne=[1,in,out]) via mul_mat: x[in,L] -> [out,L]
ggml_tensor* linear_k1(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int in, int out) {
  ggml_tensor* w2 = ggml_reshape_2d(c, w, in, out);
  ggml_tensor* y = ggml_mul_mat(c, w2, x);
  if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, out, 1));
  return y;
}
}  // namespace

ggml_tensor* MatchaModel::W(const std::string& name) const {
  auto it = w_.find(name);
  if (it == w_.end()) { RS_LOG_ERR("matcha: missing weight %s", name.c_str()); return nullptr; }
  return it->second;
}

// =====================================================================
// Text encoder: ids -> mu [80,L] (returns), logw [L,1] (out param)
// (validated exact — tools/matcha_full_encoder_validate.cpp)
// =====================================================================
ggml_tensor* MatchaModel::build_encoder(ggml_context* c, ggml_tensor* ids, int L,
                                        ggml_tensor* cosT, ggml_tensor* sinT, ggml_tensor** logw_out) {
  const int H = hp_.hidden, NH = hp_.n_heads, HD = hp_.head_dim, ROT = hp_.rotary_dim, RH = ROT / 2;
  auto Wt = [&](const std::string& s) { return W(s); };
  ggml_tensor* cos3 = ggml_reshape_3d(c, cosT, ROT, 1, L);
  ggml_tensor* sin3 = ggml_reshape_3d(c, sinT, ROT, 1, L);

  ggml_tensor* emb = ggml_scale(c, ggml_get_rows(c, Wt("model.encoder.emb.weight"), ids), std::sqrt((float)H)); // [192,L]
  ggml_tensor* x = ggml_cont(c, ggml_transpose(c, emb));  // [L,192]
  for (int i = 0; i < 3; i++) {
    std::string p = "model.encoder.prenet.", n = p + "norm_layers." + std::to_string(i) + ".";
    x = conv1d(c, x, Wt(p + "conv_layers." + std::to_string(i) + ".weight"), Wt(p + "conv_layers." + std::to_string(i) + ".bias"), 2, H);
    x = ggml_cont(c, ggml_transpose(c, x));
    x = cln(c, x, Wt(n + "Mul_1.weight"), Wt(n + "Add_1.weight"));
    x = ggml_relu(c, x);
    x = ggml_cont(c, ggml_transpose(c, x));
  }
  x = conv1d(c, x, Wt("model.encoder.prenet.proj.weight"), Wt("model.encoder.prenet.proj.bias"), 0, H);
  x = ggml_cont(c, ggml_transpose(c, x));      // [192,L]
  x = ggml_add(c, x, emb);

  auto rope = [&](ggml_tensor* t3) {
    ggml_tensor* t = ggml_cont(c, t3);
    size_t nb0 = t->nb[0], nb1 = t->nb[1], nb2 = t->nb[2];
    ggml_tensor* r  = ggml_cont(c, ggml_view_3d(c, t, ROT, NH, L, nb1, nb2, 0));
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
    fx = ggml_relu(c, conv1d(c, fx, Wt(E + "ffn_layers." + li + ".conv_1.weight"), Wt(E + "ffn_layers." + li + ".conv_1.bias"), 1, hp_.filter));
    fx = conv1d(c, fx, Wt(E + "ffn_layers." + li + ".conv_2.weight"), Wt(E + "ffn_layers." + li + ".conv_2.bias"), 1, H);
    fx = ggml_cont(c, ggml_transpose(c, fx));
    x = cln(c, ggml_add(c, x, fx), Wt(E + "norm_layers_2." + li + ".Mul_1.weight"), Wt(E + "norm_layers_2." + li + ".Add_1.weight"));
  }
  ggml_tensor* xt = ggml_cont(c, ggml_transpose(c, x));    // [L,192]
  ggml_tensor* mu = ggml_cont(c, ggml_transpose(c, conv1d(c, xt, Wt("model.encoder.proj_m.weight"), Wt("model.encoder.proj_m.bias"), 0, hp_.n_mels))); // [80,L]
  // proj_w (duration)
  ggml_tensor* dw = ggml_relu(c, conv1d(c, xt, Wt("model.encoder.proj_w.conv_1.weight"), Wt("model.encoder.proj_w.conv_1.bias"), 1, 256));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = cln(c, dw, Wt("model.encoder.proj_w.norm_1.Mul_1.weight"), Wt("model.encoder.proj_w.norm_1.Add_1.weight"));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = ggml_relu(c, conv1d(c, dw, Wt("model.encoder.proj_w.conv_2.weight"), Wt("model.encoder.proj_w.conv_2.bias"), 1, 256));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  dw = cln(c, dw, Wt("model.encoder.proj_w.norm_2.Mul_1.weight"), Wt("model.encoder.proj_w.norm_2.Add_1.weight"));
  dw = ggml_cont(c, ggml_transpose(c, dw));
  *logw_out = conv1d(c, dw, Wt("model.encoder.proj_w.proj.weight"), Wt("model.encoder.proj_w.proj.bias"), 0, 1); // [L,1]
  return mu;
}

// =====================================================================
// Vocos ConvNeXt -> spectral head [514,T]  (validated exact — matcha_vocos_validate.cpp)
// =====================================================================
ggml_tensor* MatchaModel::build_vocos(ggml_context* c, ggml_tensor* mel, int T) {
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
  return lin(h, V("head.weight"), V("head.bias"), 514);   // [514,T]
}

// iSTFT (validated corr 1.0): head[514,T] -> waveform
std::vector<float> MatchaModel::istft(const float* od, int T) const {
  const int NFFT = hp_.n_fft, HOP = hp_.hop_length, BINS = NFFT / 2 + 1;
  std::vector<float> win(NFFT);
  for (int n = 0; n < NFFT; n++) win[n] = 0.5f - 0.5f * std::cos(2.0 * PI * n / NFFT);
  int Lp = (T - 1) * HOP + NFFT;
  std::vector<double> out(Lp, 0.0), wsum(Lp, 0.0), frame(NFFT);
  const double w = 2.0 * PI / NFFT;
  for (int f = 0; f < T; f++) {
    for (int n = 0; n < NFFT; n++) {
      double acc = 0.0;
      for (int k = 0; k < BINS; k++) {
        float mlog = od[f * 514 + k]; if (mlog > 9.0f) mlog = 9.0f;
        double mag = std::exp(mlog), ph = od[f * 514 + BINS + k];
        double coef = (k == 0 || k == NFFT / 2) ? 1.0 : 2.0, ang = w * k * n;
        acc += coef * (mag * std::cos(ph) * std::cos(ang) - mag * std::sin(ph) * std::sin(ang));
      }
      frame[n] = acc / NFFT;
    }
    for (int n = 0; n < NFFT; n++) { out[f * HOP + n] += frame[n] * win[n]; wsum[f * HOP + n] += (double)win[n] * win[n]; }
  }
  int wavn = Lp - NFFT, start = NFFT / 2;
  std::vector<float> wav(wavn);
  for (int i = 0; i < wavn; i++) { double ww = wsum[start + i] < 1e-8 ? 1.0 : wsum[start + i]; wav[i] = (float)(out[start + i] / ww); }
  return wav;
}

// =====================================================================
// CFM decoder (mapped — docs/MATCHA_TTS_GGML_PORT.md; needs end-to-end validation).
// estimator(x_t, mu, t) -> vector field; 3-step Euler ODE. SnakeBeta confirmed (numpy).
// NOTE: assembled from the mapped structure; see TODO in docs. Returns mel [80,T].
// =====================================================================
ggml_tensor* MatchaModel::build_cfm(ggml_context* c, ggml_tensor* mu, int T, float noise_scale) {
  // With noise_scale handled by the caller's seed (deterministic when 0). The full UNet
  // (ResnetBlock1D + BasicTransformerBlock + down/up + 3-step ODE) is the remaining work;
  // see docs/MATCHA_TTS_GGML_PORT.md "CFM decoder". For now return mu as a placeholder so the
  // arch links and the encoder->vocos path is exercisable end-to-end pending decoder validation.
  (void)noise_scale; (void)T;
  RS_LOG_WARN("matcha: CFM decoder not yet assembled — returning mu (encoder->vocos path only).");
  return mu;
}

// =====================================================================
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
  // load all tensors
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
  auto& st = static_cast<MatchaState&>(state);
  // TEXT FRONTEND INTEGRATION POINT: the Matcha bundle ships tokens.txt + lexicon.txt +
  // espeak-ng-data + rule FSTs (sherpa-onnx style). Producing phoneme IDs for zh-tw/en is a
  // separate frontend effort; here we expect st.phoneme_ids to be pre-populated (e.g. by a
  // caller or a future RapidSpeech Matcha frontend). If empty, nothing to synthesize.
  (void)text; (void)language; (void)instruct;
  if (st.phoneme_ids.empty()) { RS_LOG_ERR("matcha: no phoneme_ids (text frontend not wired)"); return false; }
  const int L = (int)st.phoneme_ids.size();

  // build cos/sin RoPE tables on host (rotary_dim, positions 0..L-1, partial rotary).
  // Matcha bakes these as Constants; recompute standard RoPE (theta=10000) for the encoder.
  const int ROT = hp_.rotary_dim, RH = ROT / 2;
  std::vector<float> cosv((size_t)ROT * L), sinv((size_t)ROT * L);
  for (int p = 0; p < L; p++)
    for (int d = 0; d < RH; d++) {
      float freq = std::pow(10000.0f, -2.0f * d / ROT);
      float ang = p * freq;
      cosv[p * ROT + d] = cosv[p * ROT + d + RH] = std::cos(ang);
      sinv[p * ROT + d] = sinv[p * ROT + d + RH] = std::sin(ang);
    }

  // graph-build context (CPU compute for the validated path; sched-integration is a follow-up)
  size_t mem = (size_t)512 * 1024 * 1024;
  ggml_init_params ip{ mem, nullptr, false };
  ggml_context* c = ggml_init(ip);
  ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, L);
  ggml_tensor* cosT = ggml_new_tensor_2d(c, GGML_TYPE_F32, ROT, L);
  ggml_tensor* sinT = ggml_new_tensor_2d(c, GGML_TYPE_F32, ROT, L);
  memcpy(ids->data, st.phoneme_ids.data(), (size_t)L * 4);
  memcpy(cosT->data, cosv.data(), cosv.size() * 4);
  memcpy(sinT->data, sinv.data(), sinv.size() * 4);

  ggml_tensor* logw = nullptr;
  ggml_tensor* mu = build_encoder(c, ids, L, cosT, sinT, &logw);   // mu [80,L]
  // (length regulator + CFM decoder go here; build_cfm currently passes mu through)
  ggml_tensor* mel = build_cfm(c, mu, L, st.noise_scale);          // [80,T]
  int T = (int)mel->ne[1];
  ggml_tensor* head = build_vocos(c, ggml_cont(c, ggml_transpose(c, mel)) /* [T,80] */, T); // [514,T]

  ggml_cgraph* gf = ggml_new_graph(c);
  ggml_build_forward_expand(gf, head);
  ggml_graph_compute_with_ctx(c, gf, 4);

  st.audio_output = istft((const float*)head->data, T);
  st.audio_read_cursor = 0;
  ggml_free(c);
  RS_LOG_INFO("matcha: synthesized %zu samples (%.2fs)", st.audio_output.size(), st.audio_output.size() / (float)hp_.sample_rate);
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

// registration
static bool s_reg = [] {
  rs_register_model_arch("matcha-tts", []() { return std::make_shared<MatchaModel>(); });
  return true;
}();
