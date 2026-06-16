#include "xasr_zipformer2.h"
#include "core/rs_context.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"
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
  int n = 0;
  ggml_tensor *operator()(float v) {
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t);
    ggml_set_name(t, ("k" + std::to_string(n++)).c_str());
    pending.emplace_back(t, v);
    return t;
  }
  void upload() {
    for (auto &p : pending)
      ggml_backend_tensor_set(p.first, &p.second, 0, sizeof(float));
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
} // namespace

RS_API bool xasr_debug_embed(const std::map<std::string, ggml_tensor *> &w,
                             ggml_backend_t backend, const float *feats, int T,
                             int feat_dim, std::vector<float> &out, int *T_out,
                             int *dim_out) {
  size_t mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
  ggml_init_params ip{mem, nullptr, true};
  ggml_context *c = ggml_init(ip);
  Consts K{c, {}, 0};

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
