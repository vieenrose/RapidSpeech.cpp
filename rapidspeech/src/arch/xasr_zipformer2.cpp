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
