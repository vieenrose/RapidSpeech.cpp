#include "xasr_zipformer2.h"
#include "core/rs_context.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"
#include <cmath>
#include <functional>

// ---------------------------------------------------------------- gguf helpers
static int gguf_i32(gguf_context *g, const char *key, int def = 0) {
  int64_t k = gguf_find_key(g, key);
  return k == -1 ? def : gguf_get_val_i32(g, k);
}

static std::vector<int> gguf_i32_array(gguf_context *g, const char *key) {
  std::vector<int> out;
  int64_t k = gguf_find_key(g, key);
  if (k == -1)
    return out;
  const int32_t *p = (const int32_t *)gguf_get_arr_data(g, k);
  size_t n = gguf_get_arr_n(g, k);
  out.assign(p, p + n);
  return out;
}

// ---------------------------------------------------------------- model
XAsrZipformer2Model::XAsrZipformer2Model() {}
XAsrZipformer2Model::~XAsrZipformer2Model() {}

struct ggml_tensor *XAsrZipformer2Model::W(const std::string &name) const {
  auto it = w_.find(name);
  return it == w_.end() ? nullptr : it->second;
}

bool XAsrZipformer2Model::Load(const std::unique_ptr<rs_context_t> &ctx,
                               ggml_backend_t backend) {
  (void)backend;
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("XAsr: invalid context");
    return false;
  }
  gguf_context *g = ctx->ctx_gguf;

  hp_.n_stacks = gguf_i32(g, "xasr.encoder.n_stacks");
  hp_.num_layers = gguf_i32_array(g, "xasr.encoder.num_layers");
  hp_.dims = gguf_i32_array(g, "xasr.encoder.dims");
  hp_.cnn_kernels = gguf_i32_array(g, "xasr.encoder.cnn_kernels");
  hp_.left_context = gguf_i32_array(g, "xasr.encoder.left_context");
  hp_.query_head_dim = gguf_i32_array(g, "xasr.encoder.query_head_dim");
  hp_.value_head_dim = gguf_i32_array(g, "xasr.encoder.value_head_dim");
  hp_.num_heads = gguf_i32_array(g, "xasr.encoder.num_heads");
  hp_.decode_chunk_len = gguf_i32(g, "xasr.encoder.decode_chunk_len", 96);
  hp_.T = gguf_i32(g, "xasr.encoder.T", 109);
  hp_.feat_dim = gguf_i32(g, "xasr.encoder.feat_dim", 80);
  hp_.out_dim = gguf_i32(g, "xasr.encoder.out_dim", 512);
  hp_.joiner_dim = gguf_i32(g, "xasr.joiner.dim", 512);
  hp_.context_size = gguf_i32(g, "xasr.decoder.context_size", 2);
  hp_.vocab_size = gguf_i32(g, "xasr.vocab_size", 5000);
  hp_.sample_rate = gguf_i32(g, "xasr.sample_rate", 16000);

  if ((int)hp_.num_layers.size() != hp_.n_stacks || hp_.dims.empty()) {
    RS_LOG_ERR("XAsr: bad hparams (n_stacks=%d num_layers=%zu dims=%zu)",
               hp_.n_stacks, hp_.num_layers.size(), hp_.dims.size());
    return false;
  }
  int total_layers = 0;
  for (int n : hp_.num_layers)
    total_layers += n;

  // Vocabulary
  int64_t tok_key = gguf_find_key(g, "tokenizer.ggml.tokens");
  if (tok_key != -1) {
    int n = gguf_get_arr_n(g, tok_key);
    for (int i = 0; i < n; ++i)
      id_to_token_[i] = gguf_get_arr_str(g, tok_key, i);
  }

  // Map every tensor by name.
  const int n_tensors = gguf_get_n_tensors(g);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(g, i);
    struct ggml_tensor *t = ggml_get_tensor(ctx->gguf_data, name);
    if (t)
      w_[name] = t;
  }

  // Sanity: a few must-exist tensors.
  const char *required[] = {
      "encoder_embed.conv.0.weight", "encoder_embed.out.weight",
      "encoder_proj.weight",         "decoder.embedding.weight",
      "decoder_proj.weight",         "joiner.output_linear.weight"};
  for (const char *r : required) {
    if (!W(r)) {
      RS_LOG_ERR("XAsr: missing required tensor '%s'", r);
      return false;
    }
  }

  meta_.arch_name = "XAsrZipformer2";
  meta_.audio_sample_rate = hp_.sample_rate;
  meta_.n_mels = hp_.feat_dim;
  meta_.vocab_size = hp_.vocab_size;
  meta_.use_external_frontend = false;
  meta_.window_type = WindowType::POVEY;

  RS_LOG_INFO("XAsr loaded: %d stacks, %d layers, out_dim=%d, vocab=%d, "
              "%d tensors, %zu tokens",
              hp_.n_stacks, total_layers, hp_.out_dim, hp_.vocab_size,
              n_tensors, id_to_token_.size());
  return true;
}

std::shared_ptr<RSState> XAsrZipformer2Model::CreateState() {
  auto s = std::make_shared<XAsrState>();
  s->hyp.assign(hp_.context_size, GetBlankId());
  s->initialized = false;
  return s;
}

bool XAsrZipformer2Model::Encode(const std::vector<float> &input_frames,
                                 RSState &state, ggml_backend_sched_t sched) {
  (void)input_frames;
  (void)state;
  (void)sched;
  RS_LOG_ERR("XAsr::Encode not yet implemented (scaffold)");
  return false;
}

bool XAsrZipformer2Model::Decode(RSState &state, ggml_backend_sched_t sched) {
  (void)state;
  (void)sched;
  RS_LOG_ERR("XAsr::Decode not yet implemented (scaffold)");
  return false;
}

std::string XAsrZipformer2Model::GetTranscription(RSState &state) {
  auto &st = static_cast<XAsrState &>(state);
  std::string out;
  for (size_t i = hp_.context_size; i < st.hyp.size(); ++i) {
    auto it = id_to_token_.find(st.hyp[i]);
    if (it != id_to_token_.end())
      out += it->second;
  }
  return out;
}

// ================================================================ forward: embed
// Helper for graph-baked scalar constants (CUDA-friendly: plain leaf tensors set
// after allocation, no map_custom).
namespace {
struct Consts {
  ggml_context *ctx;
  std::vector<std::pair<ggml_tensor *, float>> pending;
  std::vector<ggml_tensor *> zpending;             // tensors to zero-fill
  std::vector<std::pair<ggml_tensor *, std::vector<float>>> dpending; // host data
  int n = 0;
  ggml_tensor *operator()(float v) {
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t);
    ggml_set_name(t, ("k" + std::to_string(n++)).c_str());
    pending.emplace_back(t, v);
    return t;
  }
  ggml_tensor *zeros(int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
    ggml_tensor *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
    ggml_set_input(t);
    ggml_set_name(t, ("z" + std::to_string(n++)).c_str());
    zpending.push_back(t);
    return t;
  }
  ggml_tensor *data(std::vector<float> v, int64_t a, int64_t b = 1) {
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b);
    ggml_set_input(t);
    ggml_set_name(t, ("d" + std::to_string(n++)).c_str());
    dpending.emplace_back(t, std::move(v));
    return t;
  }
  void upload() {
    for (auto &p : pending)
      ggml_backend_tensor_set(p.first, &p.second, 0, sizeof(float));
    for (auto *t : zpending) {
      std::vector<float> z(ggml_nelements(t), 0.0f);
      ggml_backend_tensor_set(t, z.data(), 0, ggml_nbytes(t));
    }
    for (auto &p : dpending)
      ggml_backend_tensor_set(p.first, p.second.data(), 0, ggml_nbytes(p.first));
  }
};

// Swoosh activations: softplus(x - shift) - 0.08*x - offset, with the
// numerically stable softplus(z) = relu(z) + log1p(exp(-|z|)) (the naive
// log(1+exp(z)) overflows to inf for large conv pre-activations).
ggml_tensor *swoosh(ggml_context *c, Consts &K, ggml_tensor *x, float shift,
                    float offset) {
  ggml_tensor *z = ggml_add(c, x, K(-shift));
  ggml_tensor *nabs = ggml_neg(c, ggml_abs(c, z));
  ggml_tensor *sp =
      ggml_add(c, ggml_relu(c, z),
               ggml_log(c, ggml_add(c, ggml_exp(c, nabs), K(1.0f))));
  ggml_tensor *lin = ggml_scale(c, x, -0.08f);
  return ggml_add(c, ggml_add(c, sp, lin), K(-offset));
}
#define SWOOSH_R(c, K, x) swoosh(c, K, x, 1.0f, 0.313261687f)
#define SWOOSH_L(c, K, x) swoosh(c, K, x, 4.0f, 0.035f)

// BiasNorm over ne0 (channel-last): out = x / rms(x-bias) * exp(log_scale).
ggml_tensor *bias_norm(ggml_context *c, ggml_tensor *x, ggml_tensor *bias,
                       ggml_tensor *log_scale) {
  int C = x->ne[0];
  ggml_tensor *b = ggml_reshape_2d(c, bias, C, 1);
  ggml_tensor *xb = ggml_sub(c, x, b);
  ggml_tensor *meansq = ggml_scale(c, ggml_sum_rows(c, ggml_sqr(c, xb)),
                                   1.0f / (float)C); // [1, T]
  ggml_tensor *xn = ggml_div(c, x, ggml_sqrt(c, meansq)); // broadcast denom
  return ggml_mul(c, xn, ggml_exp(c, log_scale));         // broadcast scalar
}

// Conv2d + per-out-channel bias. weight ne=[kw,kh,ic,oc], bias ne=[oc].
ggml_tensor *conv2d_b(ggml_context *c, ggml_tensor *w, ggml_tensor *b,
                      ggml_tensor *x, int sf, int st, int pf, int pt) {
  ggml_tensor *y = ggml_conv_2d(c, w, x, sf, st, pf, pt, 1, 1);
  return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}

// Build encoder_embed: feats4 = [freq, T, 1, 1], cache = [outW, 3, 128, 1].
// Returns embed output [out_dim, T_out].
ggml_tensor *build_embed(ggml_context *c, Consts &K,
                         const std::map<std::string, ggml_tensor *> &w,
                         ggml_tensor *feats4, ggml_tensor *cache,
                         std::vector<ggml_tensor *> *dbg = nullptr) {
  auto W = [&](const char *n) { return w.at(n); };
  auto stage = [&](ggml_tensor *t, const char *nm) {
    if (dbg) { ggml_set_output(t); ggml_set_name(t, nm); dbg->push_back(t); }
    return t;
  };
  // conv stack (W=freq, H=time): conv0 k3 pad(f1,t0); conv4 k3 s2; conv7 k3 sf2.
  ggml_tensor *x = conv2d_b(c, W("encoder_embed.conv.0.weight"),
                            W("encoder_embed.conv.0.bias"), feats4, 1, 1, 1, 0);
  stage(x, "conv0"); x = SWOOSH_R(c, K, x);
  x = conv2d_b(c, W("encoder_embed.conv.4.weight"),
               W("encoder_embed.conv.4.bias"), x, 2, 2, 0, 0);
  x = SWOOSH_R(c, K, x);
  x = conv2d_b(c, W("encoder_embed.conv.7.weight"),
               W("encoder_embed.conv.7.bias"), x, 2, 1, 0, 0);
  x = SWOOSH_R(c, K, x); // [outW, T1, 128, 1]
  stage(x, "convstack");
  // ConvNeXt streaming: bypass = x[:, :T1-3], depthwise over [cache; x].
  int T1 = x->ne[1];
  ggml_tensor *bypass =
      ggml_view_4d(c, x, x->ne[0], T1 - 3, x->ne[2], 1, x->nb[1], x->nb[2],
                   x->nb[3], 0);
  ggml_tensor *xc = ggml_concat(c, cache, x, 1); // [outW, T1+3, 128, 1]
  ggml_tensor *d = ggml_conv_2d_dw(c, W("encoder_embed.convnext.depthwise_conv.weight"),
                                   xc, 1, 1, 3, 0, 1, 1);
  d = ggml_add(c, d, ggml_reshape_4d(c, W("encoder_embed.convnext.depthwise_conv.bias"),
                                     1, 1, d->ne[2], 1));
  d = conv2d_b(c, W("encoder_embed.convnext.pointwise_conv1.weight"),
               W("encoder_embed.convnext.pointwise_conv1.bias"), d, 1, 1, 0, 0);
  d = SWOOSH_L(c, K, d);
  d = conv2d_b(c, W("encoder_embed.convnext.pointwise_conv2.weight"),
               W("encoder_embed.convnext.pointwise_conv2.bias"), d, 1, 1, 0, 0);
  x = ggml_add(c, bypass, d); // [outW, T_out, 128, 1]
  stage(x, "convnext");
  // (W=f, T, C=128) -> (f, C, T) -> [f*C, T]
  int f = x->ne[0], T = x->ne[1], ch = x->ne[2];
  x = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)); // [f, C, T, 1]
  x = ggml_reshape_2d(c, x, f * ch, T);             // [2432, T]
  // out Linear -> [out_dim, T]
  x = ggml_mul_mat(c, W("encoder_embed.out.weight"), x);
  x = ggml_add(c, x, ggml_reshape_2d(c, W("encoder_embed.out.bias"),
                                     W("encoder_embed.out.bias")->ne[0], 1));
  stage(x, "linear");
  x = bias_norm(c, x, W("encoder_embed.out_norm.bias"),
                W("encoder_embed.out_norm.log_scale"));
  return x;
}
// ================================================================ encoder layer
using TMap = std::map<std::string, ggml_tensor *>;

// debug dump hook (set by xasr_debug_encoder)
static std::vector<std::pair<std::string, ggml_tensor *>> *g_dbg = nullptr;
static std::string g_dump_prefix;
static void DBG(const std::string &p, const char *nm, ggml_tensor *t) {
  if (g_dbg && p.rfind(g_dump_prefix, 0) == 0) { ggml_set_output(t); g_dbg->push_back({nm, t}); }
}

// Linear: W ne=[in,out], optional bias[out]; x[in,T] -> [out,T].
ggml_tensor *linp(ggml_context *c, ggml_tensor *W, ggml_tensor *b, ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(c, W, x);
  if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, b->ne[0], 1));
  return y;
}

// bypass: src_orig + (src - src_orig) * scale[chan]
ggml_tensor *bypass(ggml_context *c, ggml_tensor *scale, ggml_tensor *orig,
                    ggml_tensor *src) {
  ggml_tensor *d = ggml_sub(c, src, orig);
  d = ggml_mul(c, d, ggml_reshape_2d(c, scale, scale->ne[0], 1));
  return ggml_add(c, orig, d);
}

ggml_tensor *ffn(ggml_context *c, Consts &K, const TMap &w, const std::string &p,
                 ggml_tensor *x) {
  ggml_tensor *h = linp(c, w.at(p + ".in_proj.weight"), w.at(p + ".in_proj.bias"), x);
  h = SWOOSH_L(c, K, h);
  return linp(c, w.at(p + ".out_proj.weight"), w.at(p + ".out_proj.bias"), h);
}

// rel->abs position shift: ps[sl2,T,nh] -> [kt,T,nh], out[t2,t1,h]=ps[(T-1)-t1+t2,t1,h]
ggml_tensor *rel_to_abs(ggml_context *c, ggml_tensor *ps, int T, int kt) {
  ggml_tensor *out = nullptr;
  for (int t1 = 0; t1 < T; ++t1) {
    size_t off = (size_t)((T - 1) - t1) * ps->nb[0] + (size_t)t1 * ps->nb[1];
    ggml_tensor *col = ggml_view_3d(c, ps, kt, 1, ps->ne[2], ps->nb[1], ps->nb[2], off);
    out = out ? ggml_concat(c, out, col, 1) : ggml_cont(c, col);
  }
  return ggml_cont(c, out); // [kt, T, nh]
}

// attention weights [kt,T,nh], softmax over ne0 (kt = lc + T).
ggml_tensor *attn_weights(ggml_context *c, Consts &K, const TMap &w,
                          const std::string &p, ggml_tensor *src,
                          ggml_tensor *pos_emb, int nh, int qhd, int phd, int lc) {
  int T = src->ne[1];
  ggml_tensor *x = linp(c, w.at(p + ".in_proj.weight"), w.at(p + ".in_proj.bias"), src);
  int qd = qhd * nh;
  ggml_tensor *q = ggml_view_2d(c, x, qd, T, x->nb[1], 0);
  ggml_tensor *k = ggml_view_2d(c, x, qd, T, x->nb[1], (size_t)qd * x->nb[0]);
  ggml_tensor *pp = ggml_view_2d(c, x, phd * nh, T, x->nb[1], (size_t)2 * qd * x->nb[0]);
  ggml_tensor *ck = K.zeros(qd, lc);
  ggml_tensor *kf = ggml_concat(c, ck, ggml_cont(c, k), 1); // [qd, kt]
  int kt = lc + T;
  ggml_tensor *qh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, ggml_cont(c, q), qhd, nh, T), 0, 2, 1, 3));
  ggml_tensor *kh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, kf, qhd, nh, kt), 0, 2, 1, 3));
  ggml_tensor *scores = ggml_mul_mat(c, kh, qh); // [kt, T, nh]
  ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
  DBG(p, "scores", scores);
  ggml_tensor *ph = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, ggml_cont(c, pp), phd, nh, T), 0, 2, 1, 3));
  ggml_tensor *pe = linp(c, w.at(p + ".linear_pos.weight"), nullptr, pos_emb); // [phd*nh, sl2]
  int sl2 = pos_emb->ne[1];
  ggml_tensor *peh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, pe, phd, nh, sl2), 0, 2, 1, 3));
  ggml_tensor *pscores = ggml_mul_mat(c, peh, ph); // [sl2, T, nh]
  ggml_mul_mat_set_prec(pscores, GGML_PREC_F32);
  ggml_tensor *pabs = rel_to_abs(c, pscores, T, kt);
  DBG(p, "pabs", pabs);
  ggml_tensor *sc = ggml_add(c, scores, pabs);
  DBG(p, "sc", sc);
  // key-padding mask: left-context columns [0, lc - processed_lens) are zero-filled
  // history and must be masked. processed_lens=0 on the first chunk -> mask all lc.
  int nmask = lc; // TODO: lc - processed_lens once streaming caches are threaded
  if (nmask > 0) {
    std::vector<float> m((size_t)kt, 0.0f);
    for (int j = 0; j < nmask && j < kt; ++j) m[j] = -1e4f;
    sc = ggml_add(c, sc, ggml_reshape_3d(c, K.data(m, kt, 1), kt, 1, 1));
  }
  return ggml_soft_max(c, sc); // over ne0=kt
}

// self attention: attn[kt,T,nh], returns [C,T].
ggml_tensor *self_attn(ggml_context *c, Consts &K, const TMap &w,
                       const std::string &p, ggml_tensor *src,
                       ggml_tensor *attn, int nh, int vhd, int lc) {
  int T = src->ne[1];
  ggml_tensor *v = linp(c, w.at(p + ".in_proj.weight"), w.at(p + ".in_proj.bias"), src); // [vhd*nh, T]
  ggml_tensor *cv = K.zeros(vhd * nh, lc);
  ggml_tensor *vf = ggml_concat(c, cv, v, 1); // [vhd*nh, kt]
  int kt = lc + T;
  // [vhd*nh,kt] -> [vhd,nh,kt] -> [kt,vhd,nh]
  ggml_tensor *vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, vf, vhd, nh, kt), 1, 2, 0, 3));
  // out[d,t1,h] = sum_kt vh[kt,d,h]*attn[kt,t1,h] = mul_mat(vh[kt,vhd,nh], attn[kt,T,nh])
  ggml_tensor *out = ggml_mul_mat(c, vh, attn); // [vhd, T, nh]
  ggml_tensor *merged = ggml_cont(c, ggml_permute(c, out, 0, 2, 1, 3)); // [vhd,nh,T]
  merged = ggml_reshape_2d(c, merged, vhd * nh, T);
  return linp(c, w.at(p + ".out_proj.weight"), w.at(p + ".out_proj.bias"), merged);
}

// nonlin attention: attn0 = head0 [kt,T,1]; returns [C,T].
ggml_tensor *nonlin_attn(ggml_context *c, Consts &K, const TMap &w,
                         const std::string &p, ggml_tensor *src,
                         ggml_tensor *attn0, int nh, int lc) {
  int T = src->ne[1];
  ggml_tensor *x = linp(c, w.at(p + ".in_proj.weight"), w.at(p + ".in_proj.bias"), src); // [3*hidden, T]
  int hidden = x->ne[0] / 3;
  ggml_tensor *s = ggml_cont(c, ggml_view_2d(c, x, hidden, T, x->nb[1], 0));
  ggml_tensor *xx = ggml_cont(c, ggml_view_2d(c, x, hidden, T, x->nb[1], (size_t)hidden * x->nb[0]));
  ggml_tensor *y = ggml_cont(c, ggml_view_2d(c, x, hidden, T, x->nb[1], (size_t)2 * hidden * x->nb[0]));
  ggml_tensor *xs = ggml_mul(c, xx, ggml_tanh(c, s)); // [hidden,T]
  int hd = hidden / nh;
  // [hidden,T]->[hd,nh,T]->[kt,hd,nh] with cache
  ggml_tensor *cx = K.zeros(hidden, lc);
  ggml_tensor *xf = ggml_concat(c, cx, xs, 1); // [hidden, kt]
  int kt = lc + T;
  ggml_tensor *vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, xf, hd, nh, kt), 1, 2, 0, 3)); // [kt,hd,nh]
  // attn0[kt,T,1] (batch 1) must lead so its batch broadcasts over vh's nh.
  ggml_tensor *out = ggml_mul_mat(c, attn0, vh); // -> [T, hd, nh]
  ggml_tensor *merged = ggml_cont(c, ggml_permute(c, out, 2, 0, 1, 3)); // [hd,nh,T]
  merged = ggml_reshape_2d(c, merged, hidden, T);
  merged = ggml_mul(c, merged, y);
  return linp(c, w.at(p + ".out_proj.weight"), w.at(p + ".out_proj.bias"), merged);
}

// convolution module, returns [C,T].
ggml_tensor *conv_module(ggml_context *c, Consts &K, const TMap &w,
                         const std::string &p, ggml_tensor *src, int kernel, int lc) {
  (void)lc;
  int C = src->ne[0], T = src->ne[1];
  ggml_tensor *x = linp(c, w.at(p + ".in_proj.weight"), w.at(p + ".in_proj.bias"), src); // [2C,T]
  ggml_tensor *a = ggml_cont(c, ggml_view_2d(c, x, C, T, x->nb[1], 0));
  ggml_tensor *s = ggml_cont(c, ggml_view_2d(c, x, C, T, x->nb[1], (size_t)C * x->nb[0]));
  ggml_tensor *xs = ggml_mul(c, a, ggml_sigmoid(c, s)); // [C,T]
  int lp = kernel / 2;
  std::string dp = p + ".depthwise_conv";
  // Depthwise conv1d, kernel ne=[K,1,C] (as stored), data [C,W]; returns [C,OL].
  // ggml_conv_1d_dw assumes same-length output, which breaks the shrinking causal
  // conv, so do im2col+mul_mat manually and reshape to the true output length.
  auto dwconv = [&](ggml_tensor *a, ggml_tensor *data_ct, int pad) -> ggml_tensor * {
    ggml_tensor *b = ggml_cont(c, ggml_transpose(c, data_ct)); // [W, C]
    ggml_tensor *na = ggml_reshape_4d(c, a, a->ne[0], 1, a->ne[1], a->ne[2]); // [K,1,1,C]
    ggml_tensor *nb = ggml_reshape_4d(c, b, b->ne[0], 1, b->ne[1], 1);        // [W,1,C,1]
    ggml_tensor *im = ggml_im2col(c, na, nb, 1, 0, pad, 0, 1, 0, false, GGML_TYPE_F16);
    ggml_tensor *r = ggml_mul_mat(c, im, a); // [OL, 1, C]
    r = ggml_reshape_2d(c, r, im->ne[1], C); // [OL, C]
    return ggml_cont(c, ggml_transpose(c, r)); // [C, OL]
  };
  ggml_tensor *cache = K.zeros(C, lp);
  ggml_tensor *xcat = ggml_concat(c, cache, xs, 1); // [C, lp+T]
  ggml_tensor *causal = dwconv(w.at(dp + ".causal_conv.weight"), xcat, 0); // [C,T]
  causal = ggml_add(c, causal, ggml_reshape_2d(c, w.at(dp + ".causal_conv.bias"), C, 1));
  ggml_tensor *chunk = dwconv(w.at(dp + ".chunkwise_conv.weight"), xs, kernel / 2); // [C,T]
  chunk = ggml_add(c, chunk, ggml_reshape_2d(c, w.at(dp + ".chunkwise_conv.bias"), C, 1));
  chunk = ggml_mul(c, chunk, w.at(dp + ".chunk_scale")); // [C,T]
  ggml_tensor *dw = ggml_add(c, causal, chunk); // [C,T]
  ggml_tensor *act = SWOOSH_R(c, K, dw); // out_proj = ActivationDropoutAndLinear(SwooshR)
  return linp(c, w.at(p + ".out_proj.weight"), w.at(p + ".out_proj.bias"), act);
}

// one zipformer2 layer; src [C,T] -> [C,T].
ggml_tensor *zip_layer(ggml_context *c, Consts &K, const TMap &w,
                       const std::string &p, ggml_tensor *src, ggml_tensor *pos_emb,
                       int nh, int qhd, int phd, int vhd, int kernel, int lc) {
  ggml_tensor *orig = src;
  DBG(p, "src_in", src);
  ggml_tensor *aw = attn_weights(c, K, w, p + ".self_attn_weights", src, pos_emb, nh, qhd, phd, lc);
  DBG(p, "aw", aw);
  ggml_tensor *attn0 = ggml_view_3d(c, aw, aw->ne[0], aw->ne[1], 1, aw->nb[1], aw->nb[2], 0);
  ggml_tensor *f1 = ffn(c, K, w, p + ".feed_forward1", src);
  DBG(p, "ffn1", f1); src = ggml_add(c, src, f1);
  ggml_tensor *na = nonlin_attn(c, K, w, p + ".nonlin_attention", src, attn0, nh, lc);
  DBG(p, "na", na); src = ggml_add(c, src, na);
  ggml_tensor *sa1 = self_attn(c, K, w, p + ".self_attn1", src, aw, nh, vhd, lc);
  DBG(p, "sa1", sa1); src = ggml_add(c, src, sa1);
  ggml_tensor *cv1 = conv_module(c, K, w, p + ".conv_module1", src, kernel, lc);
  DBG(p, "cv1", cv1); src = ggml_add(c, src, cv1);
  src = ggml_add(c, src, ffn(c, K, w, p + ".feed_forward2", src));
  src = bypass(c, w.at(p + ".bypass_mid.bypass_scale"), orig, src);
  src = ggml_add(c, src, self_attn(c, K, w, p + ".self_attn2", src, aw, nh, vhd, lc));
  src = ggml_add(c, src, conv_module(c, K, w, p + ".conv_module2", src, kernel, lc));
  src = ggml_add(c, src, ffn(c, K, w, p + ".feed_forward3", src));
  src = bias_norm(c, src, w.at(p + ".norm.bias"), w.at(p + ".norm.log_scale"));
  src = bypass(c, w.at(p + ".bypass.bypass_scale"), orig, src);
  return src;
}

// SimpleDownsample: x[C,T] -> [C, ceil(T/ds)] using softmax weights[ds].
ggml_tensor *simple_downsample(ggml_context *c, Consts &K, ggml_tensor *x,
                               int ds, ggml_tensor *weights) {
  int C = x->ne[0], T = x->ne[1];
  int dseq = (T + ds - 1) / ds, padT = dseq * ds;
  if (padT > T) { // right-pad repeating last column
    ggml_tensor *last = ggml_cont(c, ggml_view_2d(c, x, C, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]));
    ggml_tensor *rep = ggml_repeat(c, last, ggml_new_tensor_2d(c, GGML_TYPE_F32, C, padT - T));
    x = ggml_concat(c, x, rep, 1);
  }
  (void)K;
  // [C, padT] -> [C, ds, dseq] -> [ds, C, dseq]
  ggml_tensor *r = ggml_reshape_3d(c, x, C, ds, dseq);
  r = ggml_cont(c, ggml_permute(c, r, 1, 0, 2, 3)); // [ds, C, dseq]
  r = ggml_mul(c, r, ggml_reshape_3d(c, weights, ds, 1, 1)); // broadcast weights over C,dseq
  ggml_tensor *summed = ggml_sum_rows(c, r); // [1, C, dseq]
  return ggml_reshape_2d(c, summed, C, dseq);
}

// SimpleUpsample: x[C,dseq] -> [C, dseq*ds] (repeat each frame ds times).
ggml_tensor *simple_upsample(ggml_context *c, ggml_tensor *x, int ds) {
  int C = x->ne[0], dseq = x->ne[1];
  ggml_tensor *r = ggml_reshape_3d(c, x, C, 1, dseq);
  r = ggml_repeat(c, r, ggml_new_tensor_3d(c, GGML_TYPE_F32, C, ds, dseq));
  return ggml_reshape_2d(c, r, C, ds * dseq);
}

// convert_num_channels: truncate to nc or zero-pad (channels = ne0).
ggml_tensor *convert_channels(ggml_context *c, Consts &K, ggml_tensor *x, int nc) {
  int C = x->ne[0], T = x->ne[1];
  if (nc <= C) return ggml_cont(c, ggml_view_2d(c, x, nc, T, x->nb[1], 0));
  ggml_tensor *pad = K.zeros(nc - C, T);
  return ggml_concat(c, x, pad, 0);
}

// CompactRelPositionalEncoding (host): -> [pos_dim, seq_len2], seq_len2=2*seq+lc-1.
std::vector<float> compute_pos_emb(int seq, int lc, int pos_dim) {
  int T2 = seq + lc, sl2 = 2 * seq + lc - 1;
  float cl = sqrtf((float)pos_dim);
  float length_scale = (float)pos_dim / 6.283185307179586f; // length_factor=1
  std::vector<float> out((size_t)pos_dim * sl2, 0.0f);
  for (int k = 0; k < sl2; ++k) {
    float pos = -(float)(T2 - 1) + (float)k;
    float sgn = pos > 0 ? 1.f : (pos < 0 ? -1.f : 0.f);
    float xc = cl * sgn * (logf(fabsf(pos) + cl) - logf(cl));
    float xa = atanf(xc / length_scale);
    for (int j = 0; j < pos_dim / 2; ++j) {
      float f = 1.0f + j;
      out[(size_t)k * pos_dim + 2 * j] = cosf(xa * f);
      out[(size_t)k * pos_dim + 2 * j + 1] = sinf(xa * f);
    }
    out[(size_t)k * pos_dim + (pos_dim - 1)] = 1.0f;
  }
  return out;
}

// One stack (Zipformer2Encoder or DownsampledZipformer2Encoder).
ggml_tensor *zip_stack(ggml_context *c, Consts &K, const TMap &w, int stack,
                       int n_layers, ggml_tensor *x, int ds, int nh, int kernel,
                       int pos_dim) {
  int lc = 256 / ds;
  ggml_tensor *orig = x;
  int Torig = x->ne[1];
  if (ds > 1)
    x = simple_downsample(c, K, x, ds,
                          w.at("encoder.encoders." + std::to_string(stack) + ".downsample.weights"));
  int seq = x->ne[1];
  ggml_tensor *pe = K.data(compute_pos_emb(seq, lc, pos_dim), pos_dim, 2 * seq + lc - 1);
  std::string base = "encoder.encoders." + std::to_string(stack) +
                     (ds > 1 ? ".encoder.layers." : ".layers.");
  const char *nl = getenv("XASR_NLAYERS");
  int maxL = nl ? atoi(nl) : n_layers;
  for (int L = 0; L < n_layers && L < maxL; ++L)
    x = zip_layer(c, K, w, base + std::to_string(L), x, pe, nh, 32, 4, 12, kernel, lc);
  if (ds > 1) {
    x = simple_upsample(c, x, ds);
    x = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], Torig, x->nb[1], 0)); // trim
    x = bypass(c, w.at("encoder.encoders." + std::to_string(stack) + ".out_combiner.bypass_scale"), orig, x);
  }
  return x;
}

// Full encoder: embed_out [192,T] -> encoder_out [out_dim, T/2].
ggml_tensor *build_encoder(ggml_context *c, Consts &K, const TMap &w,
                           ggml_tensor *embed_out, const std::vector<int> &dims,
                           const std::vector<int> &nlayers,
                           const std::vector<int> &nheads,
                           const std::vector<int> &kernels,
                           const std::vector<int> &dsf, int pos_dim) {
  ggml_tensor *x = embed_out;
  const char *ns = getenv("XASR_NSTACKS");
  size_t nstacks = ns ? (size_t)atoi(ns) : dims.size();
  std::vector<ggml_tensor *> outs(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    x = convert_channels(c, K, x, dims[i]);
    x = zip_stack(c, K, w, (int)i, nlayers[i], x, dsf[i], nheads[i], kernels[i], pos_dim);
    outs[i] = x;
    if (ns && i + 1 >= nstacks) return x; // debug early-out (only when env set)
  }
  // _get_full_dim_output: cat top-256 slices of stacks 5,4,3 (dims 256,512,768).
  int n = dims.size();
  ggml_tensor *full = outs[n - 1];      // dim 256
  int cur = dims[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    if (dims[i] > cur) {
      ggml_tensor *o = outs[i];
      ggml_tensor *piece = ggml_cont(c, ggml_view_2d(c, o, dims[i] - cur, o->ne[1],
                                                     o->nb[1], (size_t)cur * o->nb[0]));
      full = ggml_concat(c, full, piece, 0);
      cur = dims[i];
    }
  }
  full = simple_downsample(c, K, full, 2, w.at("encoder.downsample_output.weights"));
  return linp(c, w.at("encoder_proj.weight"), w.at("encoder_proj.bias"), full);
}
} // namespace

RS_API bool xasr_debug_embed(const std::map<std::string, ggml_tensor *> &w,
                             ggml_backend_t backend, const float *feats, int T,
                             int feat_dim, std::vector<float> &out, int *T_out,
                             int *dim_out) {
  size_t mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
  ggml_init_params ip{mem, nullptr, true};
  ggml_context *c = ggml_init(ip);
  Consts K{c};

  ggml_tensor *feats4 = ggml_new_tensor_4d(c, GGML_TYPE_F32, feat_dim, T, 1, 1);
  ggml_set_input(feats4);
  ggml_set_name(feats4, "feats");
  int outW = (((feat_dim - 1) / 2) - 1) / 2;
  ggml_tensor *cache = ggml_new_tensor_4d(c, GGML_TYPE_F32, outW, 3, 128, 1);
  ggml_set_input(cache);
  ggml_set_name(cache, "cache");

  std::vector<ggml_tensor *> dbg;
  bool do_dbg = getenv("XASR_EMBED_DBG") != nullptr;
  ggml_tensor *y = build_embed(c, K, w, feats4, cache, do_dbg ? &dbg : nullptr);
  ggml_set_output(y);

  ggml_cgraph *gf = ggml_new_graph(c);
  ggml_build_forward_expand(gf, y);

  ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (!ggml_gallocr_alloc_graph(alloc, gf)) {
    ggml_gallocr_free(alloc); ggml_free(c); return false;
  }
  ggml_backend_tensor_set(feats4, feats, 0, (size_t)feat_dim * T * sizeof(float));
  std::vector<float> zeros((size_t)outW * 3 * 128, 0.0f);
  ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size() * sizeof(float));
  K.upload();

  if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
    ggml_gallocr_free(alloc); ggml_free(c); return false;
  }
  for (ggml_tensor *t : dbg) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    double mn = 1e30, mx = -1e30; int nan = 0;
    for (float f : v) { if (f != f) nan++; else { mn = f < mn ? f : mn; mx = f > mx ? f : mx; } }
    fprintf(stderr, "  STAGE %-10s ne=[%lld,%lld,%lld,%lld] min=%.4f max=%.4f nan=%d\n",
            t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
            (long long)t->ne[3], mn, mx, nan);
  }
  const char *dump = getenv("XASR_DUMP_STAGE");
  if (dump) {
    for (ggml_tensor *t : dbg)
      if (std::string(t->name) == dump) y = t;
  }
  *dim_out = y->ne[0];
  *T_out = y->ne[1];
  out.resize(ggml_nelements(y));
  ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
  ggml_gallocr_free(alloc);
  ggml_free(c);
  return true;
}

RS_API bool xasr_debug_encoder(const std::map<std::string, ggml_tensor *> &w,
                               ggml_backend_t backend, const float *feats, int T,
                               int feat_dim, std::vector<float> &out, int *T_out,
                               int *dim_out) {
  // 960ms config
  std::vector<int> dims{192, 256, 512, 768, 512, 256}, nlayers{2, 2, 4, 5, 4, 2},
      nheads{4, 4, 4, 8, 4, 4}, kernels{31, 31, 15, 15, 15, 31}, dsf{1, 2, 4, 8, 4, 2};
  size_t mem = ggml_tensor_overhead() * 200000 + ggml_graph_overhead_custom(200000, false);
  ggml_init_params ip{mem, nullptr, true};
  ggml_context *c = ggml_init(ip);
  Consts K{c};
  ggml_tensor *feats4 = ggml_new_tensor_4d(c, GGML_TYPE_F32, feat_dim, T, 1, 1);
  ggml_set_input(feats4); ggml_set_name(feats4, "feats");
  int outW = (((feat_dim - 1) / 2) - 1) / 2;
  ggml_tensor *cache = ggml_new_tensor_4d(c, GGML_TYPE_F32, outW, 3, 128, 1);
  ggml_set_input(cache); ggml_set_name(cache, "cache");
  std::vector<std::pair<std::string, ggml_tensor *>> dbg;
  const char *dpfx = getenv("XASR_DUMP");
  if (dpfx) { g_dbg = &dbg; g_dump_prefix = dpfx; }
  ggml_tensor *emb = build_embed(c, K, w, feats4, cache); // [192, T_emb]
  ggml_tensor *y = build_encoder(c, K, w, emb, dims, nlayers, nheads, kernels, dsf, 48);
  ggml_set_output(y);
  ggml_cgraph *gf = ggml_new_graph_custom(c, 200000, false);
  ggml_build_forward_expand(gf, y);
  ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (!ggml_gallocr_alloc_graph(alloc, gf)) { ggml_gallocr_free(alloc); ggml_free(c); return false; }
  ggml_backend_tensor_set(feats4, feats, 0, (size_t)feat_dim * T * sizeof(float));
  std::vector<float> z((size_t)outW * 3 * 128, 0.0f);
  ggml_backend_tensor_set(cache, z.data(), 0, z.size() * sizeof(float));
  K.upload();
  if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
    ggml_gallocr_free(alloc); ggml_free(c); return false;
  }
  for (auto &p : dbg) {
    std::vector<float> v(ggml_nelements(p.second));
    ggml_backend_tensor_get(p.second, v.data(), 0, ggml_nbytes(p.second));
    std::string fn = std::string("/work/models/dump_") + p.first + ".bin";
    FILE *f = fopen(fn.c_str(), "wb");
    if (f) { fwrite(v.data(), sizeof(float), v.size(), f); fclose(f); }
    fprintf(stderr, "  DUMP %s ne=[%lld,%lld,%lld]\n", p.first.c_str(),
            (long long)p.second->ne[0], (long long)p.second->ne[1], (long long)p.second->ne[2]);
  }
  g_dbg = nullptr;
  *dim_out = y->ne[0]; *T_out = y->ne[1];
  out.resize(ggml_nelements(y));
  ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
  ggml_gallocr_free(alloc); ggml_free(c);
  return true;
}

// ---------------------------------------------------------------- registration
extern void
rs_register_model_arch(const std::string &arch,
                       std::function<std::shared_ptr<ISpeechModel>()> creator);

void rs_register_xasr_zipformer2() {
  rs_register_model_arch("XAsrZipformer2", []() {
    return std::make_shared<XAsrZipformer2Model>();
  });
}

static struct XAsrRegistrar {
  XAsrRegistrar() { rs_register_xasr_zipformer2(); }
} _xasr_registrar;
