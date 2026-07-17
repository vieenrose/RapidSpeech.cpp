#include "moss_td.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <unordered_set>

// Encoder graph node budget — 28 encoder layers + conv stack + projector.
#ifndef MOSS_TD_ENC_MAX_NODES
#define MOSS_TD_ENC_MAX_NODES 8192
#endif

// ============================================================================
// Construction
// ============================================================================

// The exact instruction the v5-kl fine-tune was trained/served with — it is
// what triggers the [start][Sxx]text[end] diarization+timestamp output format.
MossTDModel::MossTDModel()
    : user_input_prompt_(
          "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
          "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
          "并在段末标注结束时间戳，以清晰标明该段语音范围。") {}
MossTDModel::~MossTDModel() = default;

void MossTDModel::SetUserInputPrompt(const std::string &prompt) {
  user_input_prompt_ = prompt;
  cached_user_input_prompt_.clear();
  cached_prefix_tokens_.clear();
  cached_suffix_tokens_.clear();
}

std::shared_ptr<RSState> MossTDModel::CreateState() {
  return std::make_shared<MossTDState>();
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

bool MossTDModel::MapEncoderTensors(
    std::map<std::string, struct ggml_tensor *> &tensors) {
  // Whisper conv1d stem.
  weights_.conv1_w = find_req(tensors, "a.conv1.weight");
  weights_.conv1_b = find_opt(tensors, "a.conv1.bias");
  weights_.conv2_w = find_req(tensors, "a.conv2.weight");
  weights_.conv2_b = find_opt(tensors, "a.conv2.bias");
  weights_.position_embd = find_req(tensors, "a.position_embd.weight");
  if (!weights_.conv1_w || !weights_.conv2_w || !weights_.position_embd)
    return false;

  weights_.post_norm_w = find_opt(tensors, "a.post_norm.weight");
  weights_.post_norm_b = find_opt(tensors, "a.post_norm.bias");

  // VQAdaptor: Linear(0) -> SiLU -> Linear(2) -> LayerNorm(norm).
  weights_.mm_1_w = find_req(tensors, "mm.a.0.weight");
  weights_.mm_1_b = find_opt(tensors, "mm.a.0.bias");
  weights_.mm_2_w = find_req(tensors, "mm.a.2.weight");
  weights_.mm_2_b = find_opt(tensors, "mm.a.2.bias");
  weights_.mm_norm_w = find_opt(tensors, "mm.a.norm.weight");
  weights_.mm_norm_b = find_opt(tensors, "mm.a.norm.bias");
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

bool MossTDModel::LoadLLM(
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
  vocab.load_qwen3_special_tokens();

  audio_start_id_ = vocab.find_token_id("<|audio_start|>");
  audio_end_id_   = vocab.find_token_id("<|audio_end|>");

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

bool MossTDModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                        ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Qwen3ASR::Load: invalid context");
    return false;
  }
  gguf_context *ctx_gguf  = ctx->ctx_gguf;
  ggml_context *gguf_data = ctx->gguf_data;

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

  // Encoder hparams (mosstd.encoder.* namespace, see conversion script).
  hparams_.n_mels         = get_i32("mosstd.encoder.n_mels", 80);
  hparams_.n_fft          = get_i32("mosstd.encoder.n_fft", 400);
  hparams_.hop_length     = get_i32("mosstd.encoder.hop_length", 160);
  hparams_.sample_rate    = get_i32("mosstd.encoder.sample_rate", 16000);
  hparams_.chunk_size     = 3000;  // full 30 s Whisper chunk
  hparams_.enc_d_model    = get_i32("mosstd.encoder.d_model", 0);
  hparams_.enc_n_head     = get_i32("mosstd.encoder.n_head", 0);
  hparams_.enc_n_layer    = get_i32("mosstd.encoder.n_layer", 0);
  hparams_.enc_ffn_dim    = get_i32("mosstd.encoder.ffn_dim", 0);
  hparams_.enc_max_pos    = get_i32("mosstd.encoder.max_source_positions", 1500);
  hparams_.enc_norm_eps   = get_f32("mosstd.encoder.layer_norm_eps", 1e-5f);
  hparams_.merge_size     = get_i32("mosstd.mm.merge_size", 4);
  hparams_.mm_in_dim      = get_i32("mosstd.mm.in_dim",
                                    hparams_.enc_d_model * hparams_.merge_size);
  hparams_.mm_out_dim     = get_i32("mosstd.mm.out_dim", 0);
  hparams_.audio_token_id = get_i32("mosstd.mm.audio_token_id", 151671);
  hparams_.mm_norm_eps    = get_f32("mosstd.mm.norm_eps", 1e-6f);

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

  meta_.arch_name           = "MossTD";
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
  const MossTDHParams &hp;
  const MossTDWeights &w;

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
      // Fused flash attention (whisper.cpp style): no materialized
      // [n_tokens, n_tokens, n_head] score matrix, no explicit softmax, and no
      // V^T copy — a large memory-bandwidth + compute win over the naive path.
      // Bidirectional encoder → no mask. Q/K/V are [head_dim, n_tokens, n_head].
      ggml_tensor *ctxv =
          ggml_flash_attn_ext(ctx, Q, K, V, /*mask=*/nullptr, scale,
                              /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
      ggml_flash_attn_ext_set_prec(ctxv, GGML_PREC_F32);
      // flash_attn_ext output: [head_dim, n_head, n_tokens] -> [d_model, n_tokens].
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
      cur = ggml_gelu_erf(ctx, cur);
      cur = ggml_mul_mat(ctx, L.ffn_dn_w, cur);
      if (L.ffn_dn_b) cur = ggml_add(ctx, cur, L.ffn_dn_b);

      cur = ggml_add(ctx, cur, r2);
    }
    return cur;
  }
};

} // namespace

bool MossTDModel::RunEncoder(const std::vector<float> &mel_features,
                              int n_frames_padded, RSState &state,
                              ggml_backend_sched_t sched) {
  auto &st = static_cast<MossTDState &>(state);
  const int chunk_size  = hparams_.chunk_size;
  const int n_mel       = hparams_.n_mels;
  // Encoder cost scales with the number of frames processed. When the audio
  // fits in a single chunk (<=30 s, the common case — the app windows at 28 s)
  // we shrink the chunk to the real 8-aligned length (set by the mel extractor)
  // instead of padding to the full 3000-frame / 30 s chunk. The encoder graph
  // is already variable-length (position embedding is sliced to T), so this
  // just avoids running conv+attention+FFN over silence. For >30 s audio we
  // keep uniform 3000-frame chunks.
  int enc_chunk, n_chunks;
  if (n_frames_padded <= chunk_size) {
    enc_chunk = n_frames_padded;   // 8-aligned (conv2 s2 + merge-4) by Compute
    n_chunks  = 1;
  } else {
    enc_chunk = chunk_size;
    n_chunks  = n_frames_padded / chunk_size;
  }
  if (n_chunks <= 0 || enc_chunk < 8) {
    RS_LOG_ERR("Qwen3ASR: empty audio (n_frames_padded=%d)", n_frames_padded);
    return false;
  }

  const int d_model  = hparams_.enc_d_model;
  const int merge    = hparams_.merge_size;
  const int T_frames = enc_chunk / 2;              // conv2 stride-2
  const int tokens_per_chunk = T_frames / merge;   // /merge (4)
  const int n_embd   = hparams_.mm_out_dim;

  st.audio_embeds.assign((size_t)n_embd * tokens_per_chunk * n_chunks, 0.0f);
  st.T_audio = tokens_per_chunk * n_chunks;
  st.n_embd  = n_embd;

  auto f16 = [&](ggml_context *c, ggml_tensor *t) {
    return ggml_cast(c, t, GGML_TYPE_F16);   // conv_1d im2col wants F16 kernel
  };

  // Each 30 s chunk is encoded independently (bidirectional within-chunk
  // attention, matching HF get_audio_features looping over chunks).
  for (int ci = 0; ci < n_chunks; ++ci) {
    struct ggml_init_params params = {
        MOSS_TD_ENC_MAX_NODES * ggml_tensor_overhead() + (1u << 20),
        nullptr, true};
    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph *gf =
        ggml_new_graph_custom(ctx0, MOSS_TD_ENC_MAX_NODES, false);

    // mel input for this chunk: [enc_chunk(frames), n_mel] (frames fast).
    ggml_tensor *inp_mel =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, enc_chunk, n_mel);
    ggml_set_name(inp_mel, "inp_mel");
    ggml_set_input(inp_mel);

    // Whisper conv1d stem: data [L=frames, IC=mel]; kernel [K,IC,OC].
    ggml_tensor *x = ggml_conv_1d(ctx0, f16(ctx0, weights_.conv1_w),
                                  inp_mel, 1, 1, 1);
    if (weights_.conv1_b)
      x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, weights_.conv1_b, 1, d_model));
    x = ggml_gelu_erf(ctx0, x);                      // [3000, 1024]
    x = ggml_conv_1d(ctx0, f16(ctx0, weights_.conv2_w), x, 2, 1, 1);
    if (weights_.conv2_b)
      x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, weights_.conv2_b, 1, d_model));
    x = ggml_gelu_erf(ctx0, x);                      // [1500, 1024] = [T, d]

    // -> [d_model, T] for the transformer.
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));    // [1024, 1500]

    // + full position embedding [d_model, max_pos] (T == max_pos == 1500).
    // Stored F16; cast to F32 so ggml_add matches the F32 activations.
    {
      ggml_tensor *pe = weights_.position_embd;      // [d_model, max_pos]
      if (pe->ne[1] != x->ne[1])
        pe = ggml_view_2d(ctx0, pe, pe->ne[0], x->ne[1], pe->nb[1], 0);
      if (pe->type != GGML_TYPE_F32) pe = ggml_cast(ctx0, pe, GGML_TYPE_F32);
      x = ggml_add(ctx0, x, pe);
    }

    // Bidirectional pre-LN transformer (reused build_layer).
    EncGraphHelper helper{ctx0, hparams_, weights_};
    ggml_tensor *cur = x;
    const int n_pos = (int)x->ne[1];
    for (int il = 0; il < hparams_.enc_n_layer; ++il)
      cur = helper.build_layer(cur, il, n_pos);

    // Post-encoder LayerNorm (Whisper final layer_norm).
    if (weights_.post_norm_w) {
      cur = ggml_norm(ctx0, cur, hparams_.enc_norm_eps);
      cur = ggml_add(ctx0, ggml_mul(ctx0, cur, weights_.post_norm_w),
                     weights_.post_norm_b);
    }

    // Merge `merge` consecutive time frames: [d_model, T] -> [d_model*merge, T/merge]
    // (contiguous reshape concatenates frames in ascending time order).
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, d_model * merge, tokens_per_chunk);

    // VQAdaptor: Linear -> SiLU -> Linear -> LayerNorm.
    cur = ggml_mul_mat(ctx0, weights_.mm_1_w, cur);
    if (weights_.mm_1_b) cur = ggml_add(ctx0, cur, weights_.mm_1_b);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, weights_.mm_2_w, cur);
    if (weights_.mm_2_b) cur = ggml_add(ctx0, cur, weights_.mm_2_b);
    if (weights_.mm_norm_w) {
      cur = ggml_norm(ctx0, cur, hparams_.mm_norm_eps);
      cur = ggml_add(ctx0, ggml_mul(ctx0, cur, weights_.mm_norm_w),
                     weights_.mm_norm_b);
    }
    ggml_set_name(cur, "audio_embeds_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("MossTD: failed to allocate encoder graph");
      ggml_free(ctx0);
      return false;
    }
    // Upload this chunk's mel. mel_features is [n_frames_padded, n_mel]
    // frames-fast, so a chunk's frames are strided per mel row -> repack.
    {
      std::vector<float> chunk_mel((size_t)enc_chunk * n_mel);
      for (int m = 0; m < n_mel; ++m) {
        const float *src = mel_features.data() +
                           (size_t)m * n_frames_padded + (size_t)ci * enc_chunk;
        std::memcpy(chunk_mel.data() + (size_t)m * enc_chunk, src,
                    (size_t)enc_chunk * sizeof(float));
      }
      ggml_backend_tensor_set(inp_mel, chunk_mel.data(), 0,
                              chunk_mel.size() * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("MossTD: encoder graph compute failed");
      ggml_free(ctx0);
      return false;
    }
    ggml_backend_tensor_get(
        cur, st.audio_embeds.data() + (size_t)n_embd * tokens_per_chunk * ci, 0,
        (size_t)n_embd * tokens_per_chunk * sizeof(float));
    ggml_free(ctx0);
    ggml_backend_sched_reset(sched);
    if (on_phase_) on_phase_("encode", ci + 1, n_chunks);
  }

  RS_LOG_INFO("MossTD encoder: n_chunks=%d tokens/chunk=%d T_audio=%d n_embd=%d",
              n_chunks, tokens_per_chunk, st.T_audio, n_embd);
  if (std::getenv("RS_DEBUG_MOSS")) {
    double s = 0, amax = 0; int nnan = 0;
    for (float v : st.audio_embeds) {
      if (std::isnan(v) || std::isinf(v)) nnan++;
      s += v; if (std::fabs(v) > amax) amax = std::fabs(v);
    }
    fprintf(stderr, "[moss] audio_embeds: n=%zu mean=%.5f absmax=%.5f nan/inf=%d\n",
            st.audio_embeds.size(), s / (double)st.audio_embeds.size(), amax, nnan);
    fprintf(stderr, "[moss] first 8: ");
    for (int i = 0; i < 8 && i < (int)st.audio_embeds.size(); ++i)
      fprintf(stderr, "%.4f ", st.audio_embeds[i]);
    fprintf(stderr, "\n");
  }
  return true;
}

bool MossTDModel::Encode(const std::vector<float> &input_frames,
                          RSState &state, ggml_backend_sched_t sched) {
  if (!mel_) {
    RS_LOG_ERR("Qwen3ASR::Encode: mel extractor uninitialised");
    return false;
  }
  // input_frames is raw PCM because meta_.use_external_frontend = true.
  const bool prof = std::getenv("RS_PROFILE") != nullptr;
  auto _t0 = std::chrono::steady_clock::now();
  std::vector<float> mel;
  int n_frames_padded = mel_->Compute(input_frames, mel);
  if (prof) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _t0).count();
    fprintf(stderr, "[prof] mel(JS-equiv DFT, %d frames): %.1f ms\n",
            n_frames_padded, ms);
  }
  if (n_frames_padded == 0) {
    RS_LOG_WARN("Qwen3ASR::Encode: empty mel features");
    return false;
  }
  // Parity harness: inject a reference mel ([n_mel, n_frames] C-order raw f32)
  // to isolate mel-extractor mismatch from the rest of the encoder.
  if (const char *mf = std::getenv("RS_MEL_FILE")) {
    FILE *f = fopen(mf, "rb");
    if (f) {
      fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
      mel.assign(bytes / sizeof(float), 0.0f);
      size_t rd = fread(mel.data(), sizeof(float), mel.size(), f);
      fclose(f);
      n_frames_padded = (int)(mel.size() / hparams_.n_mels);
      RS_LOG_WARN("MossTD: injected mel from %s (%zu floats -> n_frames=%d)",
                  mf, rd, n_frames_padded);
    }
  }
  if (std::getenv("RS_DEBUG_MOSS")) {
    // mel is [n_frames_padded, n_mel] frames-fast. Print bin 0, frames 0..7
    // and overall stats to compare against HF WhisperFeatureExtractor.
    double s = 0, amax = 0;
    for (float v : mel) { s += v; if (std::fabs(v) > amax) amax = std::fabs(v); }
    fprintf(stderr, "[moss] mel: n=%zu n_frames=%d n_mel=%d mean=%.5f absmax=%.5f\n",
            mel.size(), n_frames_padded, hparams_.n_mels,
            s / (double)mel.size(), amax);
    fprintf(stderr, "[moss] mel bin0 f0..7: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%.5f ", mel[i]);
    fprintf(stderr, "\n");
    FILE *mf = fopen("/tmp/claude-1001/rs_mel.f32", "wb");
    if (mf) { fwrite(mel.data(), sizeof(float), mel.size(), mf); fclose(mf); }
  }
  auto _te = std::chrono::steady_clock::now();
  bool ok = RunEncoder(mel, n_frames_padded, state, sched);
  // HF parity: the reference encodes silence-padded 30 s chunks but SLICES
  // each chunk's output to the real-sample token count ((n-1)//stride + 1,
  // stride = hop*2*merge = 1280 samples/token) before the LLM sees it. The
  // LLM was never trained on trailing-silence tokens; feeding them (partial
  // last chunk, or the single-chunk RS_ENC_MARGIN) corrupts timestamps and
  // destabilizes quantized decodes. Padding is contiguous at the tail, so a
  // global tail truncation equals HF's per-chunk slicing.
  if (ok && !input_frames.empty()) {
    auto &st = static_cast<MossTDState &>(state);
    const int stride = 160 * 2 * hparams_.merge_size;
    const int keep = (int)((input_frames.size() - 1) / stride) + 1;
    if (keep < st.T_audio) {
      st.audio_embeds.resize((size_t)st.n_embd * keep);
      RS_LOG_INFO("MossTD: sliced audio tokens %d -> %d (real-audio length)",
                  st.T_audio, keep);
      st.T_audio = keep;
    }
  }
  if (prof) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _te).count();
    fprintf(stderr, "[prof] encoder(conv+24L transformer+adaptor): %.1f ms\n", ms);
    int nb = ggml_backend_sched_get_n_backends(sched);
    for (int bi = 0; bi < nb; ++bi) {
      ggml_backend_t b = ggml_backend_sched_get_backend(sched, bi);
      fprintf(stderr, "[prof] post-encoder sched buffer[%d] (%s): %.1f MB\n", bi,
              ggml_backend_name(b),
              ggml_backend_sched_get_buffer_size(sched, b) / 1048576.0);
    }
  }
  return ok;
}

bool MossTDModel::EncodeMel(const std::vector<float> &mel_in, int n_frames,
                            RSState &state, ggml_backend_sched_t sched,
                            int keep_tokens) {
  const int chunk = hparams_.chunk_size;   // 3000
  const int n_mel = hparams_.n_mels;
  if (n_frames <= 0 || (int)mel_in.size() < n_mel * n_frames) {
    RS_LOG_ERR("MossTD::EncodeMel: bad mel size %zu for n_mel=%d n_frames=%d",
               mel_in.size(), n_mel, n_frames);
    return false;
  }
  const int n_chunks  = (n_frames + chunk - 1) / chunk;
  const int n_padded  = n_chunks * chunk;
  bool ok;
  if (n_padded == n_frames) {
    ok = RunEncoder(mel_in, n_padded, state, sched);
  } else {
    // Pad frames up to a 30 s chunk multiple ([n_mel, n_frames] bin-slow).
    std::vector<float> mel((size_t)n_mel * n_padded, 0.0f);
    for (int m = 0; m < n_mel; ++m)
      std::memcpy(mel.data() + (size_t)m * n_padded,
                  mel_in.data() + (size_t)m * n_frames,
                  (size_t)n_frames * sizeof(float));
    ok = RunEncoder(mel, n_padded, state, sched);
  }
  if (!ok) return false;

  // Drop trailing silence tokens: keep only the real-audio token count.
  auto &st = static_cast<MossTDState &>(state);
  if (keep_tokens > 0 && keep_tokens < st.T_audio) {
    st.audio_embeds.resize((size_t)st.n_embd * keep_tokens);
    st.T_audio = keep_tokens;
    RS_LOG_INFO("MossTD::EncodeMel: kept %d/%d audio tokens", keep_tokens,
                (int)(st.audio_embeds.size() / std::max(1, st.n_embd)));
  }
  return true;
}

// ============================================================================
// LLM decode — mirror of FunASRNanoModel::DecodeWithLLM with the audio
// embeddings used as-is (no extra adaptor — the encoder's MLP projector
// already produces LLM-dim vectors).
// ============================================================================

bool MossTDModel::DecodeWithLLM(RSState &state, ggml_backend_sched_t sched) {
  auto &st = static_cast<MossTDState &>(state);
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
    const std::string system_msg =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    const std::string before_audio =
        (audio_start_id_ >= 0 ? std::string("<|audio_start|>") : std::string());
    const std::string after_audio_with_prompt =
        (audio_end_id_ >= 0 ? std::string("<|audio_end|>\n") : std::string("\n")) +
        user_input_prompt_ +
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

  // Time markers (HF processor parity): the model is TRAINED with absolute-
  // time digit tokens interleaved into the audio span every 2 s (25 audio
  // tokens per marker at 12.5 tok/s) — e.g. ...25 audio toks..."2"...25..."4".
  // Omitting them (as this port did originally) makes the model infer time
  // without its trained clock. span[i] = -1 marks an audio-embedding slot,
  // otherwise a digit token id. RS_NO_TIME_MARKERS restores the old span.
  std::vector<int32_t> span;
  {
    const bool no_markers = std::getenv("RS_NO_TIME_MARKERS") != nullptr;
    const int every_s = 2, tok_per_marker = 25;   // int(12.5 * every_s)
    static int32_t digit_ids[10];
    static bool digits_ok = false;
    if (!digits_ok) {
      bool ok = true;
      for (int d = 0; d < 10; ++d) {
        auto t = llm_model_->vocab().tokenize(std::string(1, char('0' + d)), false);
        if (t.size() != 1) { ok = false; break; }
        digit_ids[d] = t[0];
      }
      digits_ok = ok;
    }
    if (no_markers || !digits_ok) {
      span.assign(audio_T, -1);
    } else {
      const double duration = audio_T / 12.5;
      int consumed = 0;
      for (int sec = every_s; sec <= (int)duration; sec += every_s) {
        const int pos = (sec / every_s) * tok_per_marker;
        for (int i = consumed; i < pos && i < audio_T; ++i) span.push_back(-1);
        consumed = std::max(consumed, std::min(pos, audio_T));
        for (char c : std::to_string(sec)) span.push_back(digit_ids[c - '0']);
      }
      for (int i = consumed; i < audio_T; ++i) span.push_back(-1);
    }
  }
  const int span_T   = (int)span.size();
  const int total_T  = n_prefix + span_T + n_suffix;

  RS_LOG_INFO("Qwen3ASR DecodeWithLLM: prefix=%d audio=%d suffix=%d total=%d",
              n_prefix, audio_T, n_suffix, total_T);

  // Decode budget scales with audio length: dense Mandarin runs ~7 chars/s
  // (~1 token/char) plus timestamp tokens, so a fixed 512 cap silently
  // truncated any dense window past ~80 s. Budget 1:1 with the audio token
  // count (12.5/s = ~12.5 text tokens/s): verbatim minutes-reading windows
  // exceeded the earlier 2/3 ratio and were cut mid-sentence. Clamped so
  // short clips keep the old budget and pathological inputs can't balloon
  // the KV cache.
  const int max_decode_tokens =
      std::min(3072, std::max(MAX_DECODE_TOKENS, audio_T));

  // -------- (a) projection graph: build the [n_embd, total_T] LLM input --------
  llm_cparams cparams;
  cparams.n_ctx          = total_T + max_decode_tokens;
  cparams.n_batch        = total_T;
  cparams.n_ubatch       = total_T;
  // Decode threads: was hardcoded 4 while the process is pinned to 8 P-cores.
  // RS_THREADS overrides; default = min(8, hw threads).
  const int rs_threads = std::getenv("RS_THREADS")
      ? atoi(std::getenv("RS_THREADS"))
      : std::min(8u, std::thread::hardware_concurrency());
  cparams.n_threads      = rs_threads;
  cparams.n_threads_batch = rs_threads;
  // RS_NO_FLASH=1 falls back to the mul_mat+softmax attention path (A/B).
  cparams.flash_attn = std::getenv("RS_NO_FLASH") == nullptr;
  RS_LOG_INFO("MossTD: attention path = %s",
              cparams.flash_attn ? "flash_attn_ext" : "mulmat+softmax");
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

  // Contiguous prefix+audio+suffix staging buffer (marker-free layout).
  const int stage_T = n_prefix + audio_T + n_suffix;
  std::vector<float> stage_host((size_t)n_embd * stage_T);
  ggml_backend_tensor_get(llm_in, stage_host.data(), 0,
                          stage_host.size() * sizeof(float));
  ggml_free(ctx_proj);
  ggml_backend_sched_reset(sched);

  // Interleave marker digit-token embeddings into the audio span. Digit
  // embeddings are fetched once via a tiny get_rows graph and cached.
  std::vector<float> llm_in_host((size_t)n_embd * total_T);
  {
    if (digit_embd_cache_.empty()) {
      std::vector<int32_t> uniq;
      for (int32_t t : span) if (t >= 0) uniq.push_back(t);
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      if (!uniq.empty()) {
        struct ggml_init_params dp = {
            64 * ggml_tensor_overhead() + (1u << 18), nullptr, true};
        ggml_context *ctx_d = ggml_init(dp);
        ggml_cgraph *gf_d = ggml_new_graph(ctx_d);
        ggml_tensor *ids = ggml_new_tensor_1d(ctx_d, GGML_TYPE_I32, (int)uniq.size());
        ggml_set_input(ids);
        ggml_tensor *emb = ggml_get_rows(ctx_d, llm_model_->tok_embd(), ids);
        ggml_set_output(emb);
        ggml_build_forward_expand(gf_d, emb);
        if (ggml_backend_sched_alloc_graph(sched, gf_d)) {
          ggml_backend_tensor_set(ids, uniq.data(), 0, uniq.size() * sizeof(int32_t));
          if (ggml_backend_sched_graph_compute(sched, gf_d) == GGML_STATUS_SUCCESS) {
            std::vector<float> rows((size_t)n_embd * uniq.size());
            ggml_backend_tensor_get(emb, rows.data(), 0, rows.size() * sizeof(float));
            for (size_t i = 0; i < uniq.size(); ++i)
              digit_embd_cache_[uniq[i]] = std::vector<float>(
                  rows.begin() + i * n_embd, rows.begin() + (i + 1) * n_embd);
          }
        }
        ggml_free(ctx_d);
        ggml_backend_sched_reset(sched);
      }
    }
    const float *stage = stage_host.data();
    float *out = llm_in_host.data();
    memcpy(out, stage, (size_t)n_embd * n_prefix * sizeof(float));
    out += (size_t)n_embd * n_prefix;
    const float *audio_rows = stage + (size_t)n_embd * n_prefix;
    int a = 0;
    for (int32_t t : span) {
      if (t < 0) {
        memcpy(out, audio_rows + (size_t)n_embd * a, n_embd * sizeof(float));
        ++a;
      } else {
        auto it = digit_embd_cache_.find(t);
        if (it != digit_embd_cache_.end())
          memcpy(out, it->second.data(), n_embd * sizeof(float));
        else
          memset(out, 0, n_embd * sizeof(float));
      }
      out += n_embd;
    }
    memcpy(out, stage + (size_t)n_embd * (n_prefix + audio_T),
           (size_t)n_embd * n_suffix * sizeof(float));
  }

  // -------- (b) Prefill LLM with the spliced embeddings --------
  const bool prof = std::getenv("RS_PROFILE") != nullptr;
  auto _tprefill = std::chrono::steady_clock::now();
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) positions[i] = i;  // Qwen3 0-based positions

  struct ggml_init_params llm_params = {
      512 * ggml_tensor_overhead() + (1u << 16), nullptr, true};
  struct ggml_context *ctx_llm = ggml_init(llm_params);
  ggml_tensor *llm_input =
      ggml_new_tensor_2d(ctx_llm, GGML_TYPE_F32, n_embd, total_T);
  ggml_set_input(llm_input);
  ggml_set_name(llm_input, "llm_input");

  // RS_KV_Q8: q8_0 persistent KV cache + q8_0-pinned prefill KV outputs.
  // Checked once here; used by the prefill opts, the snapshot, and the upload.
  const bool use_q8_kv = getenv("RS_KV_Q8") != nullptr;

  llm_build_opts opts;
  opts.output_mode    = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache    = true;
  opts.causal_mask     = true;
  // Pin the prefill's per-layer KV outputs as q8_0 casts: the f32 K/V become
  // reclaimable by gallocr instead of ~885 MB pinned f32 at ~3.8k ctx. The
  // snapshot below reads the q8 bytes directly.
  opts.kv_outputs_q8   = use_q8_kv;

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
  if (on_phase_) on_phase_("prefill", 0, (int)total_T);   // prefill starting
  if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("Qwen3ASR: LLM prefill compute failed");
    ggml_free(ctx_llm);
    return false;
  }
  if (std::getenv("RS_PROFILE")) {
    ggml_backend_t b0 = ggml_backend_sched_get_backend(sched, 0);
    fprintf(stderr, "[prof] post-prefill-compute sched buffer: %.1f MB\n",
            ggml_backend_sched_get_buffer_size(sched, b0) / 1048576.0);
  }
  if (on_phase_) on_phase_("decode", 0, 0);   // prefill done, decoding starts

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
  if (std::getenv("RS_DEBUG_MOSS")) {
    // top-5 tokens at first decode step
    std::vector<int> idx(n_vocab);
    for (int v = 0; v < n_vocab; ++v) idx[v] = v;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b){ return logits_host[a] > logits_host[b]; });
    fprintf(stderr, "[moss] prefix=%d suffix=%d audio=%d total=%d  prefix_ids:",
            n_prefix, n_suffix, audio_T, total_T);
    for (int i = 0; i < n_prefix && i < 12; ++i)
      fprintf(stderr, " %d", prefix_tokens[i]);
    fprintf(stderr, "\n[moss] suffix_ids:");
    for (int i = 0; i < n_suffix && i < 20; ++i)
      fprintf(stderr, " %d", suffix_tokens[i]);
    fprintf(stderr, "\n[moss] top5 first token:");
    for (int i = 0; i < 5; ++i)
      fprintf(stderr, " (%d,%.2f)", idx[i], logits_host[idx[i]]);
    fprintf(stderr, "  (im_end=151645)\n");
  }
  st.token_ids.clear();
  st.token_ids.push_back(best_token);
  if (on_token_) on_token_(llm_model_->vocab().detokenize(st.token_ids));

  // Snapshot prefill KV cache.
  const auto &lp = llm_model_->hparams();
  const int n_layer = lp.n_layer;
  const int n_head_kv = lp.n_head_kv > 0 ? lp.n_head_kv : lp.n_head;
  const int head_dim = lp.head_dim;
  const int kv_dim = head_dim * n_head_kv;
  n_cached_tokens_ = total_T;
  // RS_KV_Q8: quantize each layer's K/V to q8_0 AT EXTRACTION, into a compact
  // q8 host mirror (~34/128 the size), using a single reusable one-layer f32
  // staging buffer. This avoids materializing the all-layer f32 mirror
  // (~885 MB for a 280 s window) between the prefill graph teardown and the
  // persistent-KV upload. The default f32 path is byte-identical to before.
  const size_t q8_row = ggml_row_size(GGML_TYPE_Q8_0, kv_dim);
  const bool use_f16_kv_pref = !use_q8_kv && getenv("RS_KV_F32") == nullptr;
  const size_t f16_row_pref = ggml_row_size(GGML_TYPE_F16, kv_dim);
  std::vector<std::vector<uint8_t>> host_q8_k, host_q8_v;
  std::vector<std::vector<uint8_t>> host_f16_k, host_f16_v;
  if (use_f16_kv_pref) { host_f16_k.assign(n_layer, {}); host_f16_v.assign(n_layer, {}); }
  host_kv_cache_k_.assign(n_layer, {});
  host_kv_cache_v_.assign(n_layer, {});
  {
    std::vector<float> stage;   // one layer's f32 K or V, reused across layers
    if (use_q8_kv) {
      host_q8_k.assign(n_layer, {});
      host_q8_v.assign(n_layer, {});
      stage.resize((size_t)kv_dim * n_cached_tokens_);
    }
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = result->get_kv_output_k(il);
      ggml_tensor *v_out = result->get_kv_output_v(il);
      if (!k_out || !v_out) {
        RS_LOG_ERR("Qwen3ASR: prefill KV missing for layer %d", il);
        continue;
      }
      size_t kv_bytes = ggml_nbytes(k_out);
      if (use_q8_kv) {
        host_q8_k[il].resize(q8_row * (size_t)n_cached_tokens_);
        host_q8_v[il].resize(q8_row * (size_t)n_cached_tokens_);
        if (k_out->type == GGML_TYPE_Q8_0) {
          // kv_outputs_q8 graphs pin q8_0 casts: raw q8 byte readout.
          ggml_backend_tensor_get(k_out, host_q8_k[il].data(), 0, kv_bytes);
          ggml_backend_tensor_get(v_out, host_q8_v[il].data(), 0, kv_bytes);
        } else {
          ggml_backend_tensor_get(k_out, stage.data(), 0, kv_bytes);
          ggml_quantize_chunk(GGML_TYPE_Q8_0, stage.data(), host_q8_k[il].data(),
                              0, n_cached_tokens_, kv_dim, nullptr);
          ggml_backend_tensor_get(v_out, stage.data(), 0, kv_bytes);
          ggml_quantize_chunk(GGML_TYPE_Q8_0, stage.data(), host_q8_v[il].data(),
                              0, n_cached_tokens_, kv_dim, nullptr);
        }
      } else if (use_f16_kv_pref) {
        // f16 snapshot: read f32 layer, cast to f16 into byte staging.
        if (stage.empty()) stage.resize((size_t)kv_dim * n_cached_tokens_);
        host_f16_k[il].resize(f16_row_pref * (size_t)n_cached_tokens_);
        host_f16_v[il].resize(f16_row_pref * (size_t)n_cached_tokens_);
        ggml_backend_tensor_get(k_out, stage.data(), 0, kv_bytes);
        ggml_fp32_to_fp16_row(stage.data(), (ggml_fp16_t *)host_f16_k[il].data(),
                              (size_t)kv_dim * n_cached_tokens_);
        ggml_backend_tensor_get(v_out, stage.data(), 0, kv_bytes);
        ggml_fp32_to_fp16_row(stage.data(), (ggml_fp16_t *)host_f16_v[il].data(),
                              (size_t)kv_dim * n_cached_tokens_);
      } else {
        host_kv_cache_k_[il].resize(kv_bytes / sizeof(float));
        host_kv_cache_v_[il].resize(kv_bytes / sizeof(float));
        ggml_backend_tensor_get(k_out, host_kv_cache_k_[il].data(), 0, kv_bytes);
        ggml_backend_tensor_get(v_out, host_kv_cache_v_[il].data(), 0, kv_bytes);
      }
    }
  }

  ggml_free(ctx_llm);
  ggml_backend_sched_reset(sched);

  // -------- (c) GPU-persistent KV buffers & gallocr pre-warm --------
  // RS_KV_Q8 (MOSS-only, this file is the MOSS decoder): store the persistent
  // KV cache as q8_0 instead of f32 to cut peak memory on long windows. q8_0 is
  // near-lossless for attention. kv_dim (=head_dim*n_head_kv=128*8=1024) is a
  // multiple of QK8_0=32, so q8_0 row quantization is exact-shape. Other archs
  // (qwen3_asr, funasr-nano) don't run this code path and are unaffected.
  // KV storage type: f32 (reference) / f16 (RS_KV_F16 — half memory, plain
  // cast, no block-quant staging cost, timestamps preserved; llama.cpp's
  // default) / q8_0 (RS_KV_Q8 — smallest, but whole-second timestamp snapping
  // + ~3% text drift measured; keep for tight heaps only).
  // f16 is the DEFAULT (half of f32, sub-second timestamps preserved, ~7%
  // decode cost); RS_KV_F32 opts back into the reference format.
  const bool use_f16_kv = !use_q8_kv && getenv("RS_KV_F32") == nullptr;
  const ggml_type kv_type = use_q8_kv ? GGML_TYPE_Q8_0
                          : use_f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;
  if (use_f16_kv) {
    RS_LOG_INFO("MossTD: persistent KV cache stored as f16 (default; "
                "RS_KV_F32/RS_KV_Q8 override)");
  }
  if (use_q8_kv) {
    RS_LOG_INFO("MossTD: RS_KV_Q8 set — persistent KV cache stored as q8_0");
  }
  const int32_t n_kv_max = n_cached_tokens_ + max_decode_tokens;
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
          ggml_new_tensor_2d(ctx_kv, kv_type, kv_dim, n_kv_max);
      gpu_kv_v_vec[il] =
          ggml_new_tensor_2d(ctx_kv, kv_type, kv_dim, n_kv_max);
      ggml_set_name(gpu_kv_k_vec[il],
                    ("qwen3asr_kv_k_" + std::to_string(il)).c_str());
      ggml_set_name(gpu_kv_v_vec[il],
                    ("qwen3asr_kv_v_" + std::to_string(il)).c_str());
    }
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    ggml_backend_t gpu_backend =
        ggml_backend_sched_get_backend(sched, n_backends - 1);
    kv_gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_backend);
    if (!kv_gpu_buf) {
      RS_LOG_ERR("Qwen3ASR: failed to allocate GPU KV buffers");
    } else {
      // Zero the whole buffer: the fixed-shape decode graph attends over all
      // n_kv_max rows with the tail masked to -inf — but uninitialized f16
      // can be NaN, and NaN*0 still poisons the softmax row.
      ggml_backend_buffer_clear(kv_gpu_buf, 0);
    }
    // Upload the prefill KV (n_cached_tokens_ columns). Under RS_KV_Q8 the
    // per-layer snapshot already produced q8_0 bytes (host_q8_k/v) — raw-copy
    // them and free each layer as it lands. For f32 this is the original raw
    // copy from the f32 host mirror (kept resident; default path unchanged).
    for (int il = 0; il < n_layer; ++il) {
      if (use_q8_kv) {
        const size_t qbytes = q8_row * (size_t)n_cached_tokens_;
        if (host_q8_k[il].size() == qbytes) {
          ggml_backend_tensor_set(gpu_kv_k_vec[il], host_q8_k[il].data(), 0, qbytes);
          ggml_backend_tensor_set(gpu_kv_v_vec[il], host_q8_v[il].data(), 0, qbytes);
        }
        std::vector<uint8_t>().swap(host_q8_k[il]);
        std::vector<uint8_t>().swap(host_q8_v[il]);
      } else if (use_f16_kv) {
        const size_t fbytes = ggml_row_size(GGML_TYPE_F16, kv_dim) * (size_t)n_cached_tokens_;
        if (host_f16_k[il].size() == fbytes) {
          ggml_backend_tensor_set(gpu_kv_k_vec[il], host_f16_k[il].data(), 0, fbytes);
          ggml_backend_tensor_set(gpu_kv_v_vec[il], host_f16_v[il].data(), 0, fbytes);
        }
        std::vector<uint8_t>().swap(host_f16_k[il]);
        std::vector<uint8_t>().swap(host_f16_v[il]);
      } else {
        size_t bytes = (size_t)kv_dim * n_cached_tokens_ * sizeof(float);
        ggml_backend_tensor_set(gpu_kv_k_vec[il],
                                host_kv_cache_k_[il].data(), 0, bytes);
        ggml_backend_tensor_set(gpu_kv_v_vec[il],
                                host_kv_cache_v_[il].data(), 0, bytes);
      }
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
    // Pin only the new K/V column as extraction output (see llm_build_opts):
    // the warmup graph shapes the gallocr reserve for the decode loop, so it
    // must match the decode graphs below.
    warmup_opts.kv_outputs_new_col_only = true;
    llm_pos warmup_pos = (llm_pos)(n_kv_max - 1 + 2);
    int32_t dummy = 0;
    auto warmup = llm_graph_builder_->build_graph(
        &dummy, 1, nullptr, &warmup_pos, &warmup_opts);
    if (warmup) ggml_backend_sched_reserve(sched, warmup->get_graph());
  }

  // -------- (d) autoregressive decode --------
  const int32_t eos_token_id = llm_model_->vocab().token_eos();
  std::vector<float> kv_stage_k(kv_dim), kv_stage_v(kv_dim);
  int consecutive_same_token = 0, alternating_count = 0;
  constexpr int MAX_CONSECUTIVE_REPEAT = 10;
  constexpr int MAX_ALTERNATING_REPEAT = 10;
  constexpr int NGRAM = 8;                        // loop-breaker n-gram length
  // hash -> {hits while time stagnant, max_ts_seen at last hit}
  struct NgramSeen { int count; double max_ts; };
  std::unordered_map<uint64_t, NgramSeen> ngram_counts_;
  // The HF reference decodes with plain greedy, but under Q4_K the very first
  // format tokens sit on a knife edge: without a penalty the decode can lock
  // into a degenerate style (timestamps pinned at 0.00, "[00:00:00]" artifacts,
  // dropped speaker tags, bracket spam). A mild 1.10/64 penalty stabilizes
  // that choice; validated clean on 12 s / 80 s / 180 s dense-meeting windows
  // with the QAT q4 model (f16 is clean either way). RS_REP_PENALTY overrides
  // (e.g. =1.0 for f16 parity runs).
  const char *rp_env = std::getenv("RS_REP_PENALTY");
  const float REPETITION_PENALTY = rp_env ? (float)std::atof(rp_env) : 1.10f;
  constexpr int PENALTY_WINDOW = 64;

  if (prof) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _tprefill).count();
    fprintf(stderr, "[prof] decoder prefill (%d ctx tokens, 28L): %.1f ms\n",
            total_T, ms);
    // Memory attribution: per-backend sched compute-buffer reserve + KV buffer.
    int nb = ggml_backend_sched_get_n_backends(sched);
    for (int bi = 0; bi < nb; ++bi) {
      ggml_backend_t b = ggml_backend_sched_get_backend(sched, bi);
      fprintf(stderr, "[prof] sched buffer[%d] (%s): %.1f MB\n", bi,
              ggml_backend_name(b),
              ggml_backend_sched_get_buffer_size(sched, b) / 1048576.0);
    }
    if (kv_gpu_buf)
      fprintf(stderr, "[prof] persistent KV buffer: %.1f MB\n",
              ggml_backend_buffer_get_size(kv_gpu_buf) / 1048576.0);
  }
  // Duration/repeat stop: the decoder must not transcribe past the real audio,
  // and must not RESTART the transcript from the top (a WASM-numerics EOS-boundary
  // divergence re-emits the whole thing → duplicated segments). Each merged audio
  // token spans ~0.08 s. We track the max emitted timestamp; a new timestamp that
  // jumps backward (repeat) or runs well past the audio length ends decoding.
  const double audio_dur_s =
      static_cast<MossTDState &>(state).T_audio * 8.0 * 160.0 / 16000.0;
  double max_ts_seen = 0.0;
  // RS_AUDIO_KV_WINDOW=<seconds>: monotonic-alignment KV eviction. ASR
  // attention is effectively local in time — when emitting text for time T
  // the decoder needs audio KV near T, not minutes back. As max_ts advances,
  // COMPACT the persistent KV: slide the still-needed tail left over audio
  // columns older than (max_ts - W), so per-token attention traffic stays
  // O(window) instead of O(audio). Old K keeps its baked RoPE positions
  // (eviction removes keys, doesn't renumber them); new tokens use logical
  // positions so RoPE stays monotone. Long-range speaker identity is carried
  // by CAM++ linking, not raw audio KV, so W~45s should be accuracy-safe.
  const double kv_window_s =
      std::getenv("RS_AUDIO_KV_WINDOW") ? atof(std::getenv("RS_AUDIO_KV_WINDOW")) : 0.0;
  const int kv_audio_base = n_prefix;                 // first audio col (sinks kept)
  const double tok_per_s = (double)span_T / std::max(1e-9, audio_T / 12.5);
  int kv_audio_lo = kv_audio_base;                    // first NON-evicted audio col
  int kv_audio_end_phys = n_prefix + span_T;          // end of audio region (physical)
  int kv_evicted_total = 0;                           // cols removed so far (absolute->physical shift)
  llm_pos logical_pos = (llm_pos)(n_cached_tokens_ + 2);  // survives compaction
  const size_t kv_row_b = ggml_row_size(kv_type, kv_dim);
  std::vector<uint8_t> kv_move;                       // compaction staging
  auto _tdec = std::chrono::steady_clock::now();
  for (int step = 0; step < max_decode_tokens; ++step) {
    // ---- audio-KV eviction (batched) ----
    if (kv_window_s > 0.0 && max_ts_seen > kv_window_s) {
      // Absolute audio column for (max_ts - W), shifted into PHYSICAL space by
      // what previous compactions already removed. Without the shift, every
      // compaction after the first double-evicts (the window shrinks each
      // round -> model loses in-window audio and stops early).
      int keep_from = kv_audio_base +
          (int)((max_ts_seen - kv_window_s) * tok_per_s) - kv_evicted_total;
      keep_from = std::min(std::max(keep_from, kv_audio_lo), kv_audio_end_phys);
      const int delta = keep_from - kv_audio_lo;
      if (delta >= 256) {                             // amortize the memmove
        const int tail = n_cached_tokens_ - keep_from;
        if (tail > 0) {
          kv_move.resize((size_t)tail * kv_row_b);
          for (int il = 0; il < n_layer; ++il) {
            for (ggml_tensor *kv : {gpu_kv_k_vec[il], gpu_kv_v_vec[il]}) {
              ggml_backend_tensor_get(kv, kv_move.data(),
                                      (size_t)keep_from * kv_row_b,
                                      (size_t)tail * kv_row_b);
              ggml_backend_tensor_set(kv, kv_move.data(),
                                      (size_t)kv_audio_lo * kv_row_b,
                                      (size_t)tail * kv_row_b);
            }
          }
          n_cached_tokens_    -= delta;
          kv_audio_end_phys   -= delta;
          kv_evicted_total    += delta;
          if (prof && step % 64 < 1)
            fprintf(stderr, "[prof] kv-evict: dropped %d cols, n_kv now %d\n",
                    delta, n_cached_tokens_);
        }
      }
    }
    llm_pos decode_pos = logical_pos;
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
    // In-graph KV append (fixed-shape graph + ggml_set_rows, cosyvoice3
    // pattern): the new K/V column is written into the persistent tensors
    // DEVICE-SIDE, eliminating 2*n_layer host get/set round-trips per token
    // (the dominant per-token cost). q8_0 KV keeps the legacy host-staging
    // path (set_rows writes f32/f16); RS_DECODE_LEGACY forces it for A/B.
    const bool ingraph_kv =
        !use_q8_kv && std::getenv("RS_DECODE_LEGACY") == nullptr;
    dec_opts.fixed_kv_cache_shape = ingraph_kv;
    // Legacy path: pin only the new K/V column (~4 KB/layer) instead of the
    // full concat (~34 MB/layer at ~4k ctx) as the extraction output.
    dec_opts.kv_outputs_new_col_only = !ingraph_kv;

    auto _t0 = std::chrono::steady_clock::now();
    auto dec = llm_graph_builder_->build_graph(&best_token, 1, nullptr,
                                              &decode_pos, &dec_opts);
    if (!dec) {
      RS_LOG_ERR("Qwen3ASR: decode build failed at step %d", step);
      break;
    }
    auto _t1 = std::chrono::steady_clock::now();
    if (!ggml_backend_sched_alloc_graph(sched, dec->get_graph())) {
      RS_LOG_ERR("Qwen3ASR: decode alloc failed at step %d", step);
      break;
    }
    auto _t2 = std::chrono::steady_clock::now();
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
    if (ingraph_kv) {
      dec->set_kv_write_indices(n_cached_tokens_, n_layer);
    }
    if (ggml_backend_sched_graph_compute(sched, dec->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("Qwen3ASR: decode compute failed at step %d", step);
      break;
    }
    ggml_tensor *dec_logits = dec->get_logits();
    if (!dec_logits) {
      RS_LOG_ERR("Qwen3ASR: no logits at step %d", step);
      break;
    }
    auto _t3 = std::chrono::steady_clock::now();
    std::vector<float> dec_logits_host(n_vocab);
    ggml_backend_tensor_get(dec_logits, dec_logits_host.data(), 0,
                            (size_t)n_vocab * sizeof(float));
    if (prof && step && step % 200 == 0) {
      auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      fprintf(stderr,
              "[prof] step %d: build %.2f alloc %.2f compute %.2f logits %.2f ms\n",
              step, ms(_t0, _t1), ms(_t1, _t2), ms(_t2, _t3),
              ms(_t3, std::chrono::steady_clock::now()));
    }

    // Repetition penalty over recent window (disabled unless RS_REP_PENALTY).
    if (REPETITION_PENALTY != 1.0f) {
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
    // In-graph path: KV was appended device-side by ggml_set_rows — skip the
    // host staging entirely. Legacy/q8 path below.
    if (!ingraph_kv) {
    // k_out/v_out are the f32 concat outputs [kv_dim, n_cached+1]; the NEW
    // column lives at f32 offset n_cached_tokens_*kv_dim. For q8_0 storage we
    // must quantize that one f32 column to q8_0 and write it at the q8_0 row
    // offset (raw ggml_backend_tensor_set does not quantize).
    const size_t new_bytes_f32 = (size_t)kv_dim * sizeof(float);
    const size_t off_f32       = (size_t)n_cached_tokens_ * kv_dim * sizeof(float);
    const size_t q8_row        = ggml_row_size(GGML_TYPE_Q8_0, kv_dim);
    const size_t off_q8        = (size_t)n_cached_tokens_ * q8_row;
    std::vector<uint8_t> q8_col;
    if (use_q8_kv) q8_col.resize(q8_row);
    const size_t f16_row = ggml_row_size(GGML_TYPE_F16, kv_dim);
    const size_t off_f16 = (size_t)n_cached_tokens_ * f16_row;
    std::vector<uint8_t> f16_col;
    if (use_f16_kv) f16_col.resize(f16_row);
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = dec->get_kv_output_k(il);
      ggml_tensor *v_out = dec->get_kv_output_v(il);
      if (!k_out || !v_out) continue;
      // kv_outputs_new_col_only: the output IS the new column [kv_dim, 1] —
      // read at offset 0. (Legacy full-concat outputs: read at the column's
      // offset.) Detect by width so both contracts stay correct.
      const size_t k_off = (k_out->ne[1] == 1) ? 0 : off_f32;
      ggml_backend_tensor_get(k_out, kv_stage_k.data(), k_off, new_bytes_f32);
      ggml_backend_tensor_get(v_out, kv_stage_v.data(), k_off, new_bytes_f32);
      if (use_q8_kv) {
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kv_stage_k.data(), q8_col.data(),
                            0, 1, kv_dim, nullptr);
        ggml_backend_tensor_set(gpu_kv_k_vec[il], q8_col.data(), off_q8, q8_row);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, kv_stage_v.data(), q8_col.data(),
                            0, 1, kv_dim, nullptr);
        ggml_backend_tensor_set(gpu_kv_v_vec[il], q8_col.data(), off_q8, q8_row);
      } else if (use_f16_kv) {
        ggml_fp32_to_fp16_row(kv_stage_k.data(), (ggml_fp16_t *)f16_col.data(), kv_dim);
        ggml_backend_tensor_set(gpu_kv_k_vec[il], f16_col.data(), off_f16, f16_row);
        ggml_fp32_to_fp16_row(kv_stage_v.data(), (ggml_fp16_t *)f16_col.data(), kv_dim);
        ggml_backend_tensor_set(gpu_kv_v_vec[il], f16_col.data(), off_f16, f16_row);
      } else {
        ggml_backend_tensor_set(gpu_kv_k_vec[il], kv_stage_k.data(), off_f32,
                                new_bytes_f32);
        ggml_backend_tensor_set(gpu_kv_v_vec[il], kv_stage_v.data(), off_f32,
                                new_bytes_f32);
      }
    }
    }  // !ingraph_kv (legacy host staging)
    n_cached_tokens_++;
    logical_pos++;

    if (next_token == eos_token_id) {
      ggml_backend_sched_reset(sched);
      break;
    }
    if (consecutive_same_token >= MAX_CONSECUTIVE_REPEAT) {
      RS_LOG_WARN("Qwen3ASR: token %d repeated %d times — stopping",
                  next_token, consecutive_same_token);
      ggml_backend_sched_reset(sched);
      break;
    }
    if (alternating_count >= MAX_ALTERNATING_REPEAT) {
      RS_LOG_WARN("Qwen3ASR: alternating pattern at step %d — stopping", step);
      ggml_backend_sched_reset(sched);
      break;
    }
    st.token_ids.push_back(next_token);
    const std::string dtext = llm_model_->vocab().detokenize(st.token_ids);
    if (on_token_) on_token_(dtext);
    // Duration/repeat stop: inspect the last complete "[N.NN]" timestamp. A jump
    // FAR backward means the decoder restarted the transcript (WASM
    // EOS-boundary divergence -> duplicated segments); a value well beyond the
    // audio means it's hallucinating past the end. IMPORTANT: overlapping
    // speech in real meetings legitimately yields small backward jumps (the
    // next segment's start precedes the previous segment's end by several
    // seconds), so the backward tolerance must scale with the window — a hard
    // 0.6 s cut truncated real meeting windows. Restart jumps are ~the whole
    // window; overlap jumps are a few seconds.
    if (audio_dur_s > 0.0) {
      const size_t rb = dtext.rfind(']');
      if (rb != std::string::npos && rb > 0) {
        const size_t lb = dtext.rfind('[', rb);
        if (lb != std::string::npos && rb - lb > 1) {
          const std::string num = dtext.substr(lb + 1, rb - lb - 1);
          char *endp = nullptr;
          const double ts = std::strtod(num.c_str(), &endp);
          if (endp != num.c_str() && *endp == '\0') {   // a pure-numeric [ts]
            const double back_tol =
                std::max(5.0, std::min(20.0, 0.5 * audio_dur_s));
            if (ts < max_ts_seen - back_tol || ts > audio_dur_s + 2.0) {
              ggml_backend_sched_reset(sched);
              break;
            }
            if (ts > max_ts_seen) max_ts_seen = ts;
          }
        }
      }
    }

    // Loop breaker: the greedy decoder can cycle on silence-padded / short
    // audio (esp. under WASM SIMD numerics), repeating whole segments. The
    // repeats carry *different* timestamps, so an exact-block check misses
    // them — instead count content n-grams. But real speech repeats too:
    // council meetings re-read motions verbatim, so an n-gram recurring is
    // only a loop if the transcript CLOCK has stalled. Count a recurrence
    // only when max_ts_seen advanced < 2 s since the same n-gram's last hit
    // (a legit re-read arrives tens of seconds later; a stuck loop that DOES
    // advance time runs into the `ts > audio_dur+2` guard above instead).
    {
      const auto &tk = st.token_ids;
      const int nsz = (int)tk.size();
      if (nsz >= NGRAM) {
        uint64_t h = 1469598103934665603ULL;  // FNV-1a over the last NGRAM ids
        for (int j = nsz - NGRAM; j < nsz; ++j) {
          h ^= (uint64_t)(uint32_t)tk[j];
          h *= 1099511628211ULL;
        }
        auto it = ngram_counts_.find(h);
        if (it == ngram_counts_.end()) {
          ngram_counts_.emplace(h, NgramSeen{1, max_ts_seen});
        } else if (max_ts_seen >= it->second.max_ts + 2.0) {
          it->second = NgramSeen{1, max_ts_seen};   // time moved on: not a loop
        } else if (++it->second.count >= 3) {
          RS_LOG_WARN("MossTD: repeated content n-gram x3 with stalled "
                      "timestamps — stopping (loop breaker)");
          ggml_backend_sched_reset(sched);
          break;
        }
      }
    }

    best_token = next_token;
    ggml_backend_sched_reset(sched);
    (void)debug_decode;
  }

  if (kv_gpu_buf) ggml_backend_buffer_free(kv_gpu_buf);
  if (ctx_kv) ggml_free(ctx_kv);

  if (prof) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _tdec).count();
    int n = (int)st.token_ids.size();
    fprintf(stderr, "[prof] decoder generate (%d tokens, incremental 28L): "
            "%.1f ms  (%.1f ms/tok, %.1f tok/s)\n",
            n, ms, n > 0 ? ms / n : 0.0, n > 0 ? 1000.0 * n / ms : 0.0);
  }

  // Detokenize.
  std::string text = llm_model_->vocab().detokenize(st.token_ids);
  st.tokens.clear();
  st.tokens.push_back(text);
  RS_LOG_INFO("Qwen3ASR: %s", text.c_str());
  return true;
}

bool MossTDModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  return DecodeWithLLM(state, sched);
}

std::string MossTDModel::GetTranscription(RSState &state) {
  auto &st = static_cast<MossTDState &>(state);
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

// Explicit registration hook (referenced from the WASM bridge so LTO / dead-
// code elimination cannot strip the static registrar).
void rs_register_moss_td() {
  rs_register_model_arch(
      "MossTD", []() { return std::make_shared<MossTDModel>(); });
}

namespace {
struct MossTDRegistrar {
  MossTDRegistrar() { rs_register_moss_td(); }
} g_moss_td_registrar;
} // namespace
