#include "qwen3_asr.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"
#include "frontend/qwen3_mel.h"   // proven HTK mel frontend (matches the reference engine)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_set>

// Encoder graph node budget — 28 encoder layers + conv stack + projector.
#ifndef QWEN3_ASR_ENC_MAX_NODES
#define QWEN3_ASR_ENC_MAX_NODES 8192
#endif

// ============================================================================
// Construction
// ============================================================================

Qwen3ASRModel::Qwen3ASRModel() : user_input_prompt_("请把这段语音转写为文本。") {}
Qwen3ASRModel::~Qwen3ASRModel() = default;

void Qwen3ASRModel::SetUserInputPrompt(const std::string &prompt) {
  user_input_prompt_ = prompt;
  cached_user_input_prompt_.clear();
  cached_prefix_tokens_.clear();
  cached_suffix_tokens_.clear();
}

std::shared_ptr<RSState> Qwen3ASRModel::CreateState() {
  return std::make_shared<Qwen3ASRState>();
}

// ============================================================================
// Encoder tensor mapping. Names match the conversion script in
// scripts/convert_qwen3_asr_to_gguf.py and the llama.cpp `a.*` / `mm.a.*`
// convention.
// ============================================================================

namespace {

inline struct ggml_tensor *
find_opt(std::map<std::string, struct ggml_tensor *> &tensors,
         const std::string &name) {
  auto it = tensors.find(name);
  return it != tensors.end() ? it->second : nullptr;
}

inline struct ggml_tensor *
find_req(std::map<std::string, struct ggml_tensor *> &tensors,
         const std::string &name) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    RS_LOG_ERR("Qwen3ASR: missing required tensor %s", name.c_str());
    return nullptr;
  }
  return it->second;
}

} // namespace

bool Qwen3ASRModel::MapEncoderTensors(
    std::map<std::string, struct ggml_tensor *> &tensors) {
  // Conv2d stack
  for (int i = 0; i < 3; ++i) {
    const std::string base = "a.conv2d." + std::to_string(i);
    weights_.conv2d_w[i] = find_req(tensors, base + ".weight");
    weights_.conv2d_b[i] = find_opt(tensors, base + ".bias");
    if (!weights_.conv2d_w[i]) return false;
  }

  weights_.conv_out_w   = find_req(tensors, "a.conv_out.weight");
  weights_.conv_out_b   = find_opt(tensors, "a.conv_out.bias");
  // The real Qwen3-ASR audio tower has NO learned position table — positions are
  // computed (per-chunk sinusoidal) at encode time. Keep optional for older dumps.
  weights_.position_embd = find_opt(tensors, "a.position_embd.weight");
  if (!weights_.conv_out_w) return false;

  weights_.post_norm_w = find_opt(tensors, "a.post_norm.weight");
  weights_.post_norm_b = find_opt(tensors, "a.post_norm.bias");

  weights_.mm_1_w = find_req(tensors, "mm.a.mlp.0.weight");
  weights_.mm_1_b = find_opt(tensors, "mm.a.mlp.0.bias");
  weights_.mm_2_w = find_req(tensors, "mm.a.mlp.1.weight");
  weights_.mm_2_b = find_opt(tensors, "mm.a.mlp.1.bias");
  if (!weights_.mm_1_w || !weights_.mm_2_w) return false;

  weights_.layers.resize(hparams_.enc_n_layer);
  for (int il = 0; il < hparams_.enc_n_layer; ++il) {
    const std::string p = "a.blk." + std::to_string(il) + ".";
    auto &L = weights_.layers[il];
    L.attn_norm_w = find_req(tensors, p + "attn_norm.weight");
    L.attn_norm_b = find_opt(tensors, p + "attn_norm.bias");
    L.q_w = find_req(tensors, p + "attn_q.weight");
    L.q_b = find_opt(tensors, p + "attn_q.bias");
    L.k_w = find_req(tensors, p + "attn_k.weight");
    L.k_b = find_opt(tensors, p + "attn_k.bias");
    L.v_w = find_req(tensors, p + "attn_v.weight");
    L.v_b = find_opt(tensors, p + "attn_v.bias");
    L.o_w = find_req(tensors, p + "attn_o.weight");
    L.o_b = find_opt(tensors, p + "attn_o.bias");

    L.q_norm_w = find_opt(tensors, p + "attn_q_norm.weight");
    L.q_norm_b = find_opt(tensors, p + "attn_q_norm.bias");
    L.k_norm_w = find_opt(tensors, p + "attn_k_norm.weight");
    L.k_norm_b = find_opt(tensors, p + "attn_k_norm.bias");

    L.ffn_norm_w = find_req(tensors, p + "ffn_norm.weight");
    L.ffn_norm_b = find_opt(tensors, p + "ffn_norm.bias");
    L.ffn_up_w   = find_req(tensors, p + "ffn_up.weight");
    L.ffn_up_b   = find_opt(tensors, p + "ffn_up.bias");
    L.ffn_dn_w   = find_req(tensors, p + "ffn_down.weight");
    L.ffn_dn_b   = find_opt(tensors, p + "ffn_down.bias");

    if (!L.attn_norm_w || !L.q_w || !L.k_w || !L.v_w || !L.o_w ||
        !L.ffn_norm_w || !L.ffn_up_w || !L.ffn_dn_w) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// LLM loading: identical to FunASRNano's path.
// ============================================================================

bool Qwen3ASRModel::LoadLLM(
    struct gguf_context *ctx_gguf,
    std::map<std::string, struct ggml_tensor *> &tensors,
    ggml_backend_t backend) {
  (void)backend;
  llm_model_ = std::make_shared<llm_model>();
  if (!llm_model_->load_metadata_from_gguf(ctx_gguf)) {
    RS_LOG_ERR("Qwen3ASR: failed to load LLM metadata");
    llm_model_.reset();
    return false;
  }
  if (llm_model_->arch() != LLM_ARCH_QWEN3) {
    RS_LOG_ERR("Qwen3ASR: LLM is not Qwen3 (got %d)",
               (int)llm_model_->arch());
    llm_model_.reset();
    return false;
  }
  if (!llm_model_->map_tensors_qwen3(tensors)) {
    RS_LOG_ERR("Qwen3ASR: failed to map Qwen3 LLM tensors");
    llm_model_.reset();
    return false;
  }
  auto &vocab = const_cast<llm_vocab &>(llm_model_->vocab());
  // Capture the audio delimiter IDs BEFORE load_qwen3_special_tokens(): that call
  // re-labels 151669/151670 as <|startofspeech|>/<|endofspeech|>, and the lazy
  // build_token_map() rebuilds the reverse map from the clobbered text, which
  // would hide the real <|audio_start|>/<|audio_end|> tokens from find_token_id.
  audio_start_id_ = vocab.find_token_id("<|audio_start|>");
  audio_end_id_   = vocab.find_token_id("<|audio_end|>");
  vocab.load_qwen3_special_tokens();
  // Re-register the audio delimiters so prompt tokenize() + the reverse map keep them.
  if (audio_start_id_ >= 0) vocab.add_special_token("<|audio_start|>", audio_start_id_);
  if (audio_end_id_   >= 0) vocab.add_special_token("<|audio_end|>",   audio_end_id_);

  RS_LOG_INFO("Qwen3ASR LLM loaded: layers=%d, embd=%d, heads=%d, vocab=%d, "
              "audio_start=%d, audio_end=%d",
              llm_model_->hparams().n_layer, llm_model_->hparams().n_embd,
              llm_model_->hparams().n_head, llm_model_->hparams().n_vocab,
              audio_start_id_, audio_end_id_);
  return true;
}

// ============================================================================
// Load
// ============================================================================

bool Qwen3ASRModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                        ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Qwen3ASR::Load: invalid context");
    return false;
  }
  gguf_context *ctx_gguf  = ctx->ctx_gguf;
  ggml_context *gguf_data = ctx->gguf_data;
  // CPU-only scheduler for the batch-1 decode loop (null in CPU-only builds —
  // then the decode falls back to the main sched, which is already CPU there).
  decode_sched_ = ctx->sched_cpu;

  auto get_i32 = [&](const char *key, int32_t def) -> int32_t {
    int idx = gguf_find_key(ctx_gguf, key);
    if (idx == -1) return def;
    auto type = gguf_get_kv_type(ctx_gguf, idx);
    if (type == GGUF_TYPE_UINT32) return (int32_t)gguf_get_val_u32(ctx_gguf, idx);
    return gguf_get_val_i32(ctx_gguf, idx);
  };
  auto get_f32 = [&](const char *key, float def) -> float {
    int idx = gguf_find_key(ctx_gguf, key);
    if (idx == -1) return def;
    return gguf_get_val_f32(ctx_gguf, idx);
  };

  // Encoder hparams (qwen3a.encoder.* namespace, see conversion script).
  hparams_.n_mels         = get_i32("qwen3a.encoder.n_mels", 128);
  hparams_.n_fft          = get_i32("qwen3a.encoder.n_fft", 400);
  hparams_.hop_length     = get_i32("qwen3a.encoder.hop_length", 160);
  hparams_.sample_rate    = get_i32("qwen3a.encoder.sample_rate", 16000);
  hparams_.chunk_size     = get_i32("qwen3a.encoder.chunk_size", 100);
  hparams_.enc_d_model    = get_i32("qwen3a.encoder.d_model", 0);
  hparams_.enc_n_head     = get_i32("qwen3a.encoder.n_head", 0);
  hparams_.enc_n_layer    = get_i32("qwen3a.encoder.n_layer", 0);
  hparams_.enc_ffn_dim    = get_i32("qwen3a.encoder.ffn_dim", 0);
  hparams_.enc_max_pos    = get_i32("qwen3a.encoder.max_source_positions", 1500);
  hparams_.enc_norm_eps   = get_f32("qwen3a.encoder.layer_norm_eps", 1e-5f);
  hparams_.mm_in_dim      = get_i32("qwen3a.mm.in_dim", hparams_.enc_d_model);
  hparams_.mm_out_dim     = get_i32("qwen3a.mm.out_dim", 0);

  // LLM hparams: existence of qwen3.block_count signals LLM presence.
  int qwen3_block_count_idx = gguf_find_key(ctx_gguf, "qwen3.block_count");
  hparams_.use_llm = (qwen3_block_count_idx != -1);
  if (hparams_.use_llm) {
    hparams_.n_llm_layer = gguf_get_val_i32(ctx_gguf, qwen3_block_count_idx);
    hparams_.n_llm_embd  = get_i32("qwen3.embedding_length", 0);
    hparams_.n_llm_head  = get_i32("qwen3.attention.head_count", 0);
    hparams_.head_dim    = get_i32("qwen3.attention.key_length", 0);
    hparams_.f_llm_rope_freq_base = get_f32("qwen3.rope.freq_base", 1000000.0f);
  }

  // Sanity / fall-back: encoder d_model not exposed -> infer from pos embd.
  if (hparams_.enc_d_model == 0) {
    auto pe = ggml_get_tensor(gguf_data, "a.position_embd.weight");
    if (pe) hparams_.enc_d_model = (int32_t)pe->ne[0];
  }
  if (hparams_.mm_out_dim == 0) hparams_.mm_out_dim = hparams_.n_llm_embd;

  meta_.arch_name           = "Qwen3ASR";
  meta_.audio_sample_rate   = hparams_.sample_rate;
  meta_.n_mels              = hparams_.n_mels;
  meta_.vocab_size          = get_i32("tokenizer.vocab_size", 0);
  meta_.use_external_frontend = true;

  RS_LOG_INFO("Qwen3ASR hparams: enc d=%d h=%d L=%d ff=%d mels=%d chunk=%d  "
              "LLM L=%d d=%d h=%d",
              hparams_.enc_d_model, hparams_.enc_n_head, hparams_.enc_n_layer,
              hparams_.enc_ffn_dim, hparams_.n_mels, hparams_.chunk_size,
              hparams_.n_llm_layer, hparams_.n_llm_embd, hparams_.n_llm_head);

  // Mel extractor.
  WhisperMelConfig mel_cfg;
  mel_cfg.sample_rate = hparams_.sample_rate;
  mel_cfg.n_fft       = hparams_.n_fft;
  mel_cfg.hop_length  = hparams_.hop_length;
  mel_cfg.n_mels      = hparams_.n_mels;
  mel_cfg.f_min       = 0.0f;
  mel_cfg.f_max       = (float)hparams_.sample_rate / 2.0f;
  mel_cfg.use_htk     = (getenv("RS_MEL_HTK") != nullptr);  // Slaney (matches WhisperFeatureExtractor); RS_MEL_HTK=1 to test HTK
  mel_cfg.chunk_size  = hparams_.chunk_size;
  mel_              = std::make_unique<WhisperMelExtractor>(mel_cfg);

  // Build the global tensor lookup over the shared GGUF context.
  std::map<std::string, struct ggml_tensor *> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  if (!MapEncoderTensors(tensors)) {
    RS_LOG_ERR("Qwen3ASR: encoder tensor mapping failed");
    return false;
  }
  if (hparams_.use_llm) {
    if (!LoadLLM(ctx_gguf, tensors, backend)) return false;
  } else {
    RS_LOG_ERR("Qwen3ASR: LLM section missing in GGUF — qwen3-asr requires "
               "the bundled LLM weights");
    return false;
  }
  return true;
}

// ============================================================================
// Encoder graph + execute
// ============================================================================

namespace {

struct EncGraphHelper {
  ggml_context *ctx;
  const Qwen3ASRHParams &hp;
  const Qwen3ASRWeights &w;

  // Pre-LN bidirectional transformer block.
  ggml_tensor *build_layer(ggml_tensor *cur, int il, int n_tokens) const {
    const auto &L = w.layers[il];
    const int n_head = hp.enc_n_head;
    const int d_model = hp.enc_d_model;
    const int head_dim = d_model / n_head;

    // attention residual branch
    ggml_tensor *residual = cur;
    {
      cur = ggml_norm(ctx, cur, hp.enc_norm_eps);
      cur = ggml_add(ctx, ggml_mul(ctx, cur, L.attn_norm_w), L.attn_norm_b);

      ggml_tensor *Q = ggml_mul_mat(ctx, L.q_w, cur);
      if (L.q_b) Q = ggml_add(ctx, Q, L.q_b);
      ggml_tensor *K = ggml_mul_mat(ctx, L.k_w, cur);
      if (L.k_b) K = ggml_add(ctx, K, L.k_b);
      ggml_tensor *V = ggml_mul_mat(ctx, L.v_w, cur);
      if (L.v_b) V = ggml_add(ctx, V, L.v_b);

      // Reshape for MHA: [head_dim, n_tokens, n_head]  (batch=1)
      Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_tokens);
      K = ggml_reshape_3d(ctx, K, head_dim, n_head, n_tokens);
      V = ggml_reshape_3d(ctx, V, head_dim, n_head, n_tokens);

      // Optional Q/K LayerNorm (Qwen3 audio style — over head_dim).
      if (L.q_norm_w) {
        Q = ggml_norm(ctx, Q, hp.enc_norm_eps);
        Q = ggml_mul(ctx, Q, L.q_norm_w);
        if (L.q_norm_b) Q = ggml_add(ctx, Q, L.q_norm_b);
      }
      if (L.k_norm_w) {
        K = ggml_norm(ctx, K, hp.enc_norm_eps);
        K = ggml_mul(ctx, K, L.k_norm_w);
        if (L.k_norm_b) K = ggml_add(ctx, K, L.k_norm_b);
      }

      // Permute so head is dim 2 (heads slowest of the matmul reductions).
      // current shape: [head_dim, n_head, n_tokens]
      Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
      K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
      V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
      // Now shape: [head_dim, n_tokens, n_head]

      const float scale = 1.0f / std::sqrt((float)head_dim);
      ggml_tensor *scores = ggml_mul_mat(ctx, K, Q);  // [n_tokens, n_tokens, n_head]
      scores = ggml_scale_inplace(ctx, scores, scale);
      ggml_tensor *probs  = ggml_soft_max_inplace(ctx, scores);

      // V^T so we can mul_mat against probs: V was [head_dim, n_tokens, n_head]
      ggml_tensor *Vt = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
      // Vt: [n_tokens, head_dim, n_head]
      ggml_tensor *ctxv = ggml_mul_mat(ctx, Vt, probs);
      // ctxv: [n_tokens(out), head_dim, n_head]  — see funasr-nano reference.
      ctxv = ggml_cont(ctx, ggml_permute(ctx, ctxv, 0, 2, 1, 3));
      ctxv = ggml_reshape_2d(ctx, ctxv, d_model, n_tokens);

      cur = ggml_mul_mat(ctx, L.o_w, ctxv);
      if (L.o_b) cur = ggml_add(ctx, cur, L.o_b);
      cur = ggml_add(ctx, cur, residual);
    }

    // FFN residual branch (Pre-LN, GELU-erf, gateless).
    {
      ggml_tensor *r2 = cur;
      cur = ggml_norm(ctx, cur, hp.enc_norm_eps);
      cur = ggml_add(ctx, ggml_mul(ctx, cur, L.ffn_norm_w), L.ffn_norm_b);

      cur = ggml_mul_mat(ctx, L.ffn_up_w, cur);
      if (L.ffn_up_b) cur = ggml_add(ctx, cur, L.ffn_up_b);
      cur = ggml_gelu(ctx, cur);
      cur = ggml_mul_mat(ctx, L.ffn_dn_w, cur);
      if (L.ffn_dn_b) cur = ggml_add(ctx, cur, L.ffn_dn_b);

      cur = ggml_add(ctx, cur, r2);
    }
    return cur;
  }
};

} // namespace

bool Qwen3ASRModel::RunEncoder(const std::vector<float> &mel_features,
                              int n_frames_padded, RSState &state,
                              ggml_backend_sched_t sched) {
  auto &st = static_cast<Qwen3ASRState &>(state);
  const int chunk_size  = hparams_.chunk_size;
  const int n_chunks    = n_frames_padded / chunk_size;
  const int n_mel       = hparams_.n_mels;
  if (n_chunks <= 0) {
    RS_LOG_ERR("Qwen3ASR: empty audio (n_chunks=0)");
    return false;
  }

  struct ggml_init_params params = {
      QWEN3_ASR_ENC_MAX_NODES * ggml_tensor_overhead() + (1u << 20),
      nullptr,
      true};
  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx0, QWEN3_ASR_ENC_MAX_NODES, false);

  // 1. Input mel buffer as a 2D tensor [n_frames_padded, n_mel]
  //    (frames fast, mels slow — matches WhisperMelExtractor output).
  ggml_tensor *inp_mel =
      ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames_padded, n_mel);
  ggml_set_name(inp_mel, "inp_mel");
  ggml_set_input(inp_mel);

  // 2. View as [chunk_size, n_mel, 1, n_chunks]: chunk frames are contiguous
  //    on ne[0]; ne[1] strides by n_frames_padded; ne[3] strides by chunk_size.
  ggml_tensor *inp = ggml_view_4d(
      ctx0, inp_mel,
      chunk_size, n_mel, 1, n_chunks,
      (size_t)n_frames_padded * sizeof(float),  // nb[1]
      (size_t)chunk_size * sizeof(float),       // nb[2]
      (size_t)chunk_size * sizeof(float),       // nb[3]
      0);
  inp = ggml_cont(ctx0, inp);

  // 3. Three stride-2 conv2d blocks with GELU-erf, kernel 3x3, pad 1x1.
  auto conv_block = [&](ggml_tensor *x, ggml_tensor *kw, ggml_tensor *kb) {
    x = ggml_conv_2d(ctx0, kw, x, 2, 2, 1, 1, 1, 1);
    if (kb) {
      x = ggml_add(ctx0, x,
                   ggml_reshape_4d(ctx0, kb, 1, 1, x->ne[2], 1));
    }
    return ggml_gelu(ctx0, x);
  };
  inp = conv_block(inp, weights_.conv2d_w[0], weights_.conv2d_b[0]);
  inp = conv_block(inp, weights_.conv2d_w[1], weights_.conv2d_b[1]);
  inp = conv_block(inp, weights_.conv2d_w[2], weights_.conv2d_b[2]);
  // shape: [OW, OH, OC, n_chunks] with OW=13, OH=16

  // 4. permute [OW, OH, OC, n_chunks] -> [OH, OC, OW, n_chunks]
  //    then reshape [OH*OC, OW*n_chunks]
  inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 2, 0, 1, 3));
  inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1],
                        inp->ne[2] * inp->ne[3]);

  // 5. project to d_model: [d_model, OW*n_chunks]
  inp = ggml_mul_mat(ctx0, weights_.conv_out_w, inp);
  if (weights_.conv_out_b) inp = ggml_add(ctx0, inp, weights_.conv_out_b);

  const int n_pos = (int)inp->ne[1];                    // OW * n_chunks
  const int tokens_per_chunk = n_pos / n_chunks;        // = OW = 13 expected
  hparams_.tokens_per_chunk = tokens_per_chunk;

  // 6. Per-chunk sinusoidal position embedding. The real model computes it
  //    (no learned table); positions reset every chunk, so we build a
  //    [d_model, tokens_per_chunk] sinusoidal input and repeat it across chunks.
  ggml_tensor *inp_pos = ggml_new_tensor_2d(
      ctx0, GGML_TYPE_F32, hparams_.enc_d_model, tokens_per_chunk);
  ggml_set_name(inp_pos, "inp_pos");
  ggml_set_input(inp_pos);
  {
    ggml_tensor *tgt =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.enc_d_model, n_pos);
    inp = ggml_add(ctx0, inp, ggml_repeat(ctx0, inp_pos, tgt));
  }

  // 7. Transformer encoder.
  EncGraphHelper helper{ctx0, hparams_, weights_};
  ggml_tensor *cur = inp;
  for (int il = 0; il < hparams_.enc_n_layer; ++il) {
    cur = helper.build_layer(cur, il, n_pos);
  }

  // 8. Optional post-encoder LayerNorm.
  if (weights_.post_norm_w) {
    cur = ggml_norm(ctx0, cur, hparams_.enc_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, weights_.post_norm_w),
                   weights_.post_norm_b);
  }

  // 9. 2-layer MLP projector -> LLM hidden dim.
  cur = ggml_mul_mat(ctx0, weights_.mm_1_w, cur);
  if (weights_.mm_1_b) cur = ggml_add(ctx0, cur, weights_.mm_1_b);
  cur = ggml_gelu(ctx0, cur);
  cur = ggml_mul_mat(ctx0, weights_.mm_2_w, cur);
  if (weights_.mm_2_b) cur = ggml_add(ctx0, cur, weights_.mm_2_b);
  // cur: [mm_out_dim, n_pos]
  ggml_set_name(cur, "audio_embeds_out");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("Qwen3ASR: failed to allocate encoder graph");
    ggml_free(ctx0);
    return false;
  }
  ggml_backend_tensor_set(inp_mel, mel_features.data(), 0,
                          mel_features.size() * sizeof(float));
  // Upload the computed per-chunk sinusoidal position embedding (matches the
  // reference compute_sinusoidal_pe: half=d/2, div_term=exp(-ln(1e4)*i/(half-1))).
  {
    const int d = hparams_.enc_d_model, half = d / 2, tpc = tokens_per_chunk;
    std::vector<float> pe((size_t)d * tpc, 0.0f);
    for (int p = 0; p < tpc; ++p) {
      for (int i = 0; i < half; ++i) {
        float div_term = std::exp(-std::log(10000.0f) * i / (half - 1));
        float angle = (float)p * div_term;
        pe[(size_t)p * d + i]        = std::sin(angle);
        pe[(size_t)p * d + half + i] = std::cos(angle);
      }
    }
    ggml_tensor *pe_t = ggml_graph_get_tensor(gf, "inp_pos");
    ggml_backend_tensor_set(pe_t, pe.data(), 0, pe.size() * sizeof(float));
  }
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("Qwen3ASR: encoder graph compute failed");
    ggml_free(ctx0);
    return false;
  }

  const int n_embd = (int)cur->ne[0];
  st.audio_embeds.assign((size_t)n_embd * n_pos, 0.0f);
  st.T_audio = n_pos;
  st.n_embd  = n_embd;
  ggml_backend_tensor_get(cur, st.audio_embeds.data(), 0,
                          (size_t)n_embd * n_pos * sizeof(float));

  if (const char* dp = getenv("QWEN_DUMP_EMB")) {
    FILE* fe = fopen(dp, "wb");
    if (fe) { int32_t d[2]={n_pos,n_embd}; fwrite(d,sizeof(int32_t),2,fe);
      fwrite(st.audio_embeds.data(),sizeof(float),(size_t)n_embd*n_pos,fe); fclose(fe);
      RS_LOG_INFO("dumped audio_embeds [%d,%d] -> %s", n_pos, n_embd, dp); }
  }

  RS_LOG_INFO("Qwen3ASR encoder: n_chunks=%d tokens/chunk=%d T_audio=%d n_embd=%d",
              n_chunks, tokens_per_chunk, n_pos, n_embd);

  ggml_free(ctx0);
  ggml_backend_sched_reset(sched);
  return true;
}

bool Qwen3ASRModel::Encode(const std::vector<float> &input_frames,
                          RSState &state, ggml_backend_sched_t sched) {
  if (!mel_) {
    RS_LOG_ERR("Qwen3ASR::Encode: mel extractor uninitialised");
    return false;
  }
  // input_frames is raw PCM because meta_.use_external_frontend = true.
  // Use the PROVEN HTK mel frontend (qwen3_mel) — the scaffold's whisper_mel STFT
  // did not match torch.stft (~14x scaling + structural divergence). This is the
  // exact mel the model was validated against.
  static MelFilters s_mel_filters; static bool s_mf_init = false;
  if (!s_mf_init) {
    generate_mel_filters(s_mel_filters, hparams_.n_mels, hparams_.n_fft, hparams_.sample_rate);
    s_mf_init = true;
  }
  MelSpectrogram ms;
  if (!log_mel_spectrogram(input_frames.data(), (int)input_frames.size(),
                           s_mel_filters, ms, 4) || ms.n_len <= 0) {
    RS_LOG_WARN("Qwen3ASR::Encode: mel computation failed");
    return false;
  }
  // Pad to a multiple of chunk_size and lay out as [n_mel * n_frames_padded]
  // (frame-fast), matching RunEncoder's inp_mel expectation.
  const int chunk = hparams_.chunk_size;
  const int n_frames_padded = ((ms.n_len + chunk - 1) / chunk) * chunk;
  float floor_v = 1e30f;
  for (float v : ms.data) if (v < floor_v) floor_v = v;
  std::vector<float> mel((size_t)hparams_.n_mels * n_frames_padded, floor_v);
  for (int j = 0; j < hparams_.n_mels; ++j)
    for (int i = 0; i < ms.n_len; ++i)
      mel[(size_t)j * n_frames_padded + i] = ms.data[(size_t)j * ms.n_len + i];
  if (const char* dp = getenv("QWEN_DUMP_MEL")) {
    FILE* fm = fopen(dp, "wb");
    if (fm) { int32_t d[2]={(int32_t)hparams_.n_mels, n_frames_padded};
      fwrite(d,sizeof(int32_t),2,fm); fwrite(mel.data(),sizeof(float),mel.size(),fm); fclose(fm);
      RS_LOG_INFO("dumped mel [%d,%d] (%zu floats) -> %s", d[0], d[1], mel.size(), dp); }
  }
  return RunEncoder(mel, n_frames_padded, state, sched);
}

// ============================================================================
// LLM decode — mirror of FunASRNanoModel::DecodeWithLLM with the audio
// embeddings used as-is (no extra adaptor — the encoder's MLP projector
// already produces LLM-dim vectors).
// ============================================================================

bool Qwen3ASRModel::DecodeWithLLM(RSState &state, ggml_backend_sched_t sched) {
  auto &st = static_cast<Qwen3ASRState &>(state);
  if (st.audio_embeds.empty() || !llm_model_) return false;
  const bool debug_decode = std::getenv("RS_DEBUG_ASR_DECODE") != nullptr;

  const int audio_T = st.T_audio;
  const int n_embd  = st.n_embd;
  if (n_embd != (int)llm_model_->hparams().n_embd) {
    RS_LOG_ERR("Qwen3ASR: encoder output dim (%d) != LLM n_embd (%d)",
               n_embd, (int)llm_model_->hparams().n_embd);
    return false;
  }

  // Rebuild prompt token streams when the user prompt changes.
  if (cached_prefix_tokens_.empty() ||
      cached_user_input_prompt_ != user_input_prompt_) {
    // Minimal Qwen3-ASR chat template — matches the proven reference engine:
    // EMPTY system message, bare audio, and NO instruction text after the audio.
    // (The verbose "You are a speech recognition model" + user prompt is
    // off-distribution for this checkpoint and yields empty/garbled output.)
    const std::string system_msg =
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
    const std::string before_audio =
        (audio_start_id_ >= 0 ? std::string("<|audio_start|>") : std::string());
    const std::string after_audio_with_prompt =
        (audio_end_id_ >= 0 ? std::string("<|audio_end|>") : std::string()) +
        "<|im_end|>\n<|im_start|>assistant\n";

    cached_prefix_tokens_ =
        llm_model_->vocab().tokenize(system_msg + before_audio, false);
    cached_suffix_tokens_ =
        llm_model_->vocab().tokenize(after_audio_with_prompt, false);
    cached_user_input_prompt_ = user_input_prompt_;
  }
  const auto &prefix_tokens = cached_prefix_tokens_;
  const auto &suffix_tokens = cached_suffix_tokens_;
  const int n_prefix = (int)prefix_tokens.size();
  const int n_suffix = (int)suffix_tokens.size();
  const int total_T  = n_prefix + audio_T + n_suffix;

  RS_LOG_INFO("Qwen3ASR DecodeWithLLM: prefix=%d audio=%d suffix=%d total=%d",
              n_prefix, audio_T, n_suffix, total_T);

  // -------- (a) projection graph: build the [n_embd, total_T] LLM input --------
  llm_cparams cparams;
  cparams.n_ctx          = total_T + MAX_DECODE_TOKENS;
  cparams.n_batch        = total_T;
  cparams.n_ubatch       = total_T;
  cparams.n_threads      = 4;
  cparams.n_threads_batch = 4;
  // Jetson Nano gen1 (sm_53 Maxwell) has no flash-attention CUDA kernel, and the
  // batch-1 decode would violate GGML_KQ_MASK_PAD. The manual multi-head path
  // (build_multi_head_attn) is numerically correct on every backend, so disable
  // flash-attn for Qwen3-ASR. Set RS_CUDA_FA=1 to force-enable on Ampere+ desktops.
  cparams.flash_attn = (getenv("RS_CUDA_FA") != nullptr);
  llm_graph_builder_ =
      std::make_unique<llm_build_qwen3>(*llm_model_, cparams, sched);

  host_kv_cache_k_.clear();
  host_kv_cache_v_.clear();
  n_cached_tokens_ = 0;

  struct ggml_init_params proj_params = {
      1024 * ggml_tensor_overhead() + (1u << 16), nullptr, true};
  struct ggml_context *ctx_proj = ggml_init(proj_params);
  struct ggml_cgraph *gf_proj   = ggml_new_graph(ctx_proj);

  ggml_tensor *inp_prefix =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, n_prefix);
  ggml_tensor *inp_suffix =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, n_suffix);
  ggml_set_input(inp_prefix);
  ggml_set_input(inp_suffix);
  ggml_set_name(inp_prefix, "prefix_tokens");
  ggml_set_name(inp_suffix, "suffix_tokens");
  ggml_tensor *prefix_embd =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_prefix);
  ggml_tensor *suffix_embd =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_suffix);

  ggml_tensor *audio_embd_in =
      ggml_new_tensor_2d(ctx_proj, GGML_TYPE_F32, n_embd, audio_T);
  ggml_set_input(audio_embd_in);
  ggml_set_name(audio_embd_in, "audio_embeds_in");

  ggml_tensor *llm_in = ggml_concat(ctx_proj, prefix_embd, audio_embd_in, 1);
  llm_in = ggml_concat(ctx_proj, llm_in, suffix_embd, 1);
  ggml_set_output(llm_in);
  ggml_build_forward_expand(gf_proj, llm_in);

  if (!ggml_backend_sched_alloc_graph(sched, gf_proj)) {
    RS_LOG_ERR("Qwen3ASR: failed to allocate projection graph");
    ggml_free(ctx_proj);
    return false;
  }
  ggml_backend_tensor_set(inp_prefix, prefix_tokens.data(), 0,
                          n_prefix * sizeof(int32_t));
  ggml_backend_tensor_set(inp_suffix, suffix_tokens.data(), 0,
                          n_suffix * sizeof(int32_t));
  ggml_backend_tensor_set(audio_embd_in, st.audio_embeds.data(), 0,
                          (size_t)n_embd * audio_T * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, gf_proj) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("Qwen3ASR: projection graph compute failed");
    ggml_free(ctx_proj);
    return false;
  }

  std::vector<float> llm_in_host((size_t)n_embd * total_T);
  ggml_backend_tensor_get(llm_in, llm_in_host.data(), 0,
                          llm_in_host.size() * sizeof(float));
  ggml_free(ctx_proj);
  ggml_backend_sched_reset(sched);

  // -------- (b) Prefill LLM with the spliced embeddings --------
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) positions[i] = i + 2;  // matches Python ref

  struct ggml_init_params llm_params = {
      512 * ggml_tensor_overhead() + (1u << 16), nullptr, true};
  struct ggml_context *ctx_llm = ggml_init(llm_params);
  ggml_tensor *llm_input =
      ggml_new_tensor_2d(ctx_llm, GGML_TYPE_F32, n_embd, total_T);
  ggml_set_input(llm_input);
  ggml_set_name(llm_input, "llm_input");

  llm_build_opts opts;
  opts.output_mode    = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache    = true;
  opts.causal_mask     = true;

  auto result = llm_graph_builder_->build_graph_from_embeds(
      llm_input, total_T, nullptr, positions.data(), &opts);
  if (!result) {
    RS_LOG_ERR("Qwen3ASR: failed to build LLM prefill graph");
    ggml_free(ctx_llm);
    return false;
  }
  if (!ggml_backend_sched_alloc_graph(sched, result->get_graph())) {
    RS_LOG_ERR("Qwen3ASR: failed to alloc LLM prefill graph");
    ggml_free(ctx_llm);
    return false;
  }
  ggml_backend_tensor_set(llm_input, llm_in_host.data(), 0,
                          llm_in_host.size() * sizeof(float));
  if (auto t = result->get_input_tensor("position_ids")) {
    result->set_position_ids(t, positions.data(), total_T);
  } else if (auto t2 = result->get_input_tensor("position_ids_seq")) {
    result->set_position_ids(t2, positions.data(), total_T);
  }
  if (auto t = result->get_input_tensor("causal_mask")) {
    result->set_causal_mask(t, total_T, 0);
  }
  if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("Qwen3ASR: LLM prefill compute failed");
    ggml_free(ctx_llm);
    return false;
  }

  ggml_tensor *logits = result->get_logits();
  if (!logits) {
    RS_LOG_ERR("Qwen3ASR: no logits from prefill");
    ggml_free(ctx_llm);
    return false;
  }
  const int n_vocab = (int)logits->ne[0];
  std::vector<float> logits_host(n_vocab);
  ggml_backend_tensor_get(logits, logits_host.data(), 0,
                          (size_t)n_vocab * sizeof(float));
  int32_t best_token = 0;
  {
    float best = logits_host[0];
    for (int v = 1; v < n_vocab; ++v) {
      if (logits_host[v] > best) { best = logits_host[v]; best_token = v; }
    }
  }
  st.token_ids.clear();
  st.token_ids.push_back(best_token);

  // Snapshot prefill KV cache.
  const auto &lp = llm_model_->hparams();
  const int n_layer = lp.n_layer;
  const int n_head_kv = lp.n_head_kv > 0 ? lp.n_head_kv : lp.n_head;
  const int head_dim = lp.head_dim;
  const int kv_dim = head_dim * n_head_kv;
  host_kv_cache_k_.assign(n_layer, {});
  host_kv_cache_v_.assign(n_layer, {});
  n_cached_tokens_ = total_T;
  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *k_out = result->get_kv_output_k(il);
    ggml_tensor *v_out = result->get_kv_output_v(il);
    if (!k_out || !v_out) {
      RS_LOG_ERR("Qwen3ASR: prefill KV missing for layer %d", il);
      continue;
    }
    size_t kv_bytes = ggml_nbytes(k_out);
    host_kv_cache_k_[il].resize(kv_bytes / sizeof(float));
    host_kv_cache_v_[il].resize(kv_bytes / sizeof(float));
    ggml_backend_tensor_get(k_out, host_kv_cache_k_[il].data(), 0, kv_bytes);
    ggml_backend_tensor_get(v_out, host_kv_cache_v_[il].data(), 0, kv_bytes);
  }

  ggml_free(ctx_llm);
  ggml_backend_sched_reset(sched);

  // -------- (c) persistent KV buffers & gallocr pre-warm --------
  // The autoregressive decode runs on the CPU-only scheduler when available
  // (batch-1 is ~3.7x faster on the A57 than per-token GPU weight uploads on
  // Maxwell sm_53). Falls back to the main `sched` (already CPU in CPU-only
  // builds). The CPU decode sched can ONLY be used when the LLM weights are
  // CPU-resident — rs_context arranges that unless RS_QWEN3ASR_GPU_WEIGHTS=1,
  // so that flag (and RS_QWEN3ASR_GPU_DECODE=1) must keep decode on the GPU
  // sched, else the CPU sched hits a GPU-resident weight and aborts.
  const bool gpu_decode = getenv("RS_QWEN3ASR_GPU_DECODE") != nullptr ||
                          getenv("RS_QWEN3ASR_GPU_WEIGHTS") != nullptr;
  ggml_backend_sched_t decode_sched =
      (decode_sched_ && !gpu_decode) ? decode_sched_ : sched;
  const int32_t n_kv_max = n_cached_tokens_ + MAX_DECODE_TOKENS;
  std::vector<ggml_tensor *> gpu_kv_k_vec(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v_vec(n_layer, nullptr);
  ggml_backend_buffer_t kv_gpu_buf = nullptr;
  struct ggml_context *ctx_kv = nullptr;
  {
    struct ggml_init_params kv_buf_params = {
        (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16),
        nullptr, true};
    ctx_kv = ggml_init(kv_buf_params);
    for (int il = 0; il < n_layer; ++il) {
      gpu_kv_k_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      gpu_kv_v_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      ggml_set_name(gpu_kv_k_vec[il],
                    ("qwen3asr_kv_k_" + std::to_string(il)).c_str());
      ggml_set_name(gpu_kv_v_vec[il],
                    ("qwen3asr_kv_v_" + std::to_string(il)).c_str());
    }
    // Persistent KV lives on the CPU backend so the CPU decode scheduler can
    // read/write it in place. `sched`'s last backend is CPU (sched invariant),
    // and `decode_sched_` is that same CPU backend — so this is correct for both
    // the split (GPU prefill + CPU decode) and the CPU-only build.
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    ggml_backend_t kv_backend =
        ggml_backend_sched_get_backend(sched, n_backends - 1);
    kv_gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, kv_backend);
    if (!kv_gpu_buf) {
      RS_LOG_ERR("Qwen3ASR: failed to allocate persistent KV buffers");
    }
    for (int il = 0; il < n_layer; ++il) {
      size_t bytes = (size_t)kv_dim * n_cached_tokens_ * sizeof(float);
      ggml_backend_tensor_set(gpu_kv_k_vec[il],
                              host_kv_cache_k_[il].data(), 0, bytes);
      ggml_backend_tensor_set(gpu_kv_v_vec[il],
                              host_kv_cache_v_[il].data(), 0, bytes);
    }
  }
  {
    llm_build_opts warmup_opts;
    warmup_opts.output_mode    = llm_output_mode::OUTPUT_LOGITS;
    warmup_opts.skip_embeddings = false;
    warmup_opts.use_kv_cache    = true;
    warmup_opts.is_decode_step  = true;
    warmup_opts.n_kv_cache      = n_kv_max - 1;
    warmup_opts.n_kv_max        = n_kv_max;
    warmup_opts.causal_mask     = true;
    warmup_opts.gpu_kv_k        = gpu_kv_k_vec.data();
    warmup_opts.gpu_kv_v        = gpu_kv_v_vec.data();
    llm_pos warmup_pos = (llm_pos)(n_kv_max - 1 + 2);
    int32_t dummy = 0;
    auto warmup = llm_graph_builder_->build_graph(
        &dummy, 1, nullptr, &warmup_pos, &warmup_opts);
    if (warmup) ggml_backend_sched_reserve(decode_sched, warmup->get_graph());
  }

  // -------- (d) autoregressive decode (on the CPU scheduler) --------
  const int32_t eos_token_id = llm_model_->vocab().token_eos();
  std::vector<float> kv_stage_k(kv_dim), kv_stage_v(kv_dim);
  int consecutive_same_token = 0, alternating_count = 0;
  constexpr int MAX_CONSECUTIVE_REPEAT = 10;
  constexpr int MAX_ALTERNATING_REPEAT = 10;
  constexpr float REPETITION_PENALTY = 1.10f;
  constexpr int PENALTY_WINDOW = 64;

  for (int step = 0; step < MAX_DECODE_TOKENS; ++step) {
    llm_pos decode_pos = (llm_pos)(n_cached_tokens_ + 2);
    llm_build_opts dec_opts;
    dec_opts.output_mode    = llm_output_mode::OUTPUT_LOGITS;
    dec_opts.skip_embeddings = false;
    dec_opts.use_kv_cache    = true;
    dec_opts.is_decode_step  = true;
    dec_opts.n_kv_cache      = n_cached_tokens_;
    dec_opts.n_kv_max        = n_kv_max;
    dec_opts.causal_mask     = true;
    dec_opts.gpu_kv_k        = gpu_kv_k_vec.data();
    dec_opts.gpu_kv_v        = gpu_kv_v_vec.data();

    auto dec = llm_graph_builder_->build_graph(&best_token, 1, nullptr,
                                              &decode_pos, &dec_opts);
    if (!dec) {
      RS_LOG_ERR("Qwen3ASR: decode build failed at step %d", step);
      break;
    }
    if (!ggml_backend_sched_alloc_graph(decode_sched, dec->get_graph())) {
      RS_LOG_ERR("Qwen3ASR: decode alloc failed at step %d", step);
      break;
    }
    if (auto t = dec->get_input_tensor("inp_tokens")) {
      ggml_backend_tensor_set(t, &best_token, 0, sizeof(int32_t));
    }
    if (auto t = dec->get_input_tensor("position_ids")) {
      dec->set_position_ids(t, &decode_pos, 1);
    } else if (auto t2 = dec->get_input_tensor("position_ids_seq")) {
      dec->set_position_ids(t2, &decode_pos, 1);
    }
    if (auto t = dec->get_input_tensor("causal_mask")) {
      dec->set_causal_mask(t, 1, n_cached_tokens_);
    }
    if (ggml_backend_sched_graph_compute(decode_sched, dec->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("Qwen3ASR: decode compute failed at step %d", step);
      break;
    }
    ggml_tensor *dec_logits = dec->get_logits();
    if (!dec_logits) {
      RS_LOG_ERR("Qwen3ASR: no logits at step %d", step);
      break;
    }
    std::vector<float> dec_logits_host(n_vocab);
    ggml_backend_tensor_get(dec_logits, dec_logits_host.data(), 0,
                            (size_t)n_vocab * sizeof(float));

    // Repetition penalty over recent window.
    {
      int p_start = std::max(0, (int)st.token_ids.size() - PENALTY_WINDOW);
      std::unordered_set<int32_t> recent;
      for (int p = p_start; p < (int)st.token_ids.size(); ++p) {
        recent.insert(st.token_ids[p]);
      }
      for (int32_t tid : recent) {
        if (tid >= 0 && tid < n_vocab) {
          float &v = dec_logits_host[tid];
          if (v > 0.0f) v /= REPETITION_PENALTY;
          else          v *= REPETITION_PENALTY;
        }
      }
    }

    int32_t next_token = 0;
    {
      float best = dec_logits_host[0];
      for (int v = 1; v < n_vocab; ++v) {
        if (dec_logits_host[v] > best) {
          best = dec_logits_host[v];
          next_token = v;
        }
      }
    }

    if (next_token == best_token) consecutive_same_token++;
    else                          consecutive_same_token = 0;
    if (st.token_ids.size() >= 2 &&
        next_token == st.token_ids[st.token_ids.size() - 2] &&
        next_token != st.token_ids.back()) {
      alternating_count++;
    } else {
      alternating_count = 0;
    }

    // Stage new KV column to host + persistent GPU buffer.
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = dec->get_kv_output_k(il);
      ggml_tensor *v_out = dec->get_kv_output_v(il);
      if (!k_out || !v_out) continue;
      const size_t new_bytes = (size_t)kv_dim * sizeof(float);
      const size_t off       = (size_t)n_cached_tokens_ * kv_dim * sizeof(float);
      ggml_backend_tensor_get(k_out, kv_stage_k.data(), off, new_bytes);
      ggml_backend_tensor_get(v_out, kv_stage_v.data(), off, new_bytes);
      ggml_backend_tensor_set(gpu_kv_k_vec[il], kv_stage_k.data(), off, new_bytes);
      ggml_backend_tensor_set(gpu_kv_v_vec[il], kv_stage_v.data(), off, new_bytes);
    }
    n_cached_tokens_++;

    if (next_token == eos_token_id) {
      ggml_backend_sched_reset(decode_sched);
      break;
    }
    if (consecutive_same_token >= MAX_CONSECUTIVE_REPEAT) {
      RS_LOG_WARN("Qwen3ASR: token %d repeated %d times — stopping",
                  next_token, consecutive_same_token);
      ggml_backend_sched_reset(decode_sched);
      break;
    }
    if (alternating_count >= MAX_ALTERNATING_REPEAT) {
      RS_LOG_WARN("Qwen3ASR: alternating pattern at step %d — stopping", step);
      ggml_backend_sched_reset(decode_sched);
      break;
    }
    st.token_ids.push_back(next_token);
    best_token = next_token;
    ggml_backend_sched_reset(decode_sched);
    (void)debug_decode;
  }

  if (kv_gpu_buf) ggml_backend_buffer_free(kv_gpu_buf);
  if (ctx_kv) ggml_free(ctx_kv);

  // Detokenize.
  std::string text = llm_model_->vocab().detokenize(st.token_ids);
  st.tokens.clear();
  st.tokens.push_back(text);
  RS_LOG_INFO("Qwen3ASR: %s", text.c_str());
  return true;
}

bool Qwen3ASRModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  return DecodeWithLLM(state, sched);
}

std::string Qwen3ASRModel::GetTranscription(RSState &state) {
  auto &st = static_cast<Qwen3ASRState &>(state);
  std::string result;
  result.reserve(64);
  for (const auto &s : st.tokens) result += s;
  st.tokens.clear();
  return result;
}

// ============================================================================
// Registration
// ============================================================================

extern void
rs_register_model_arch(const std::string &arch,
                       std::function<std::shared_ptr<ISpeechModel>()> creator);

namespace {
struct Qwen3ASRRegistrar {
  Qwen3ASRRegistrar() {
    rs_register_model_arch(
        "Qwen3ASR", []() { return std::make_shared<Qwen3ASRModel>(); });
  }
} g_qwen3_asr_registrar;
} // namespace
