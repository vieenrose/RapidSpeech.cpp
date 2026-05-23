#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/whisper_mel.h"
#include "llm_graph.h"
#include "llm_model.h"
#include "qwen3.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * Hyperparameters for Qwen3-ASR.
 *
 * Encoder side mirrors the Whisper-family audio tower used in
 * `llama.cpp/tools/mtmd/models/qwen3a.cpp`: three stride-2 conv2d blocks,
 * a linear projection to d_model, per-chunk position embeddings repeated
 * across chunks, then a bidirectional Pre-LN transformer and a 2-layer
 * MLP projector that lands in the LLM hidden dimension.
 *
 * LLM side is a vanilla Qwen3 dense transformer (reuses `llm_build_qwen3`).
 */
struct Qwen3ASRHParams {
  // Audio frontend
  int32_t n_mels         = 128;
  int32_t n_fft          = 400;
  int32_t hop_length     = 160;
  int32_t sample_rate    = 16000;
  int32_t chunk_size     = 100;   // mel frames per chunk
  int32_t tokens_per_chunk = 13;  // after 3 stride-2 conv2d blocks

  // Encoder transformer
  int32_t enc_d_model    = 0;
  int32_t enc_n_head     = 0;
  int32_t enc_n_layer    = 0;
  int32_t enc_ffn_dim    = 0;
  int32_t enc_max_pos    = 0;     // = position_embeddings rows
  float   enc_norm_eps   = 1e-5f;

  // MLP projector dims (encoder d_model -> LLM n_embd).
  int32_t mm_in_dim      = 0;
  int32_t mm_out_dim     = 0;

  // LLM (Qwen3) — mirrored from FunASRNanoHParams for convenience.
  bool    use_llm        = true;
  int32_t n_llm_layer    = 0;
  int32_t n_llm_embd     = 0;
  int32_t n_llm_head     = 0;
  int32_t head_dim       = 0;
  int32_t n_llm_vocab    = 0;
  float   f_llm_rope_freq_base = 1000000.0f;
};

/**
 * Encoder + projector weight bundle. All `ggml_tensor *` are non-owning views
 * into the GGUF context. Optional tensors (q/k norm, post-encoder LN) are
 * nullptr when not present.
 */
struct Qwen3ASRWeights {
  // Conv2d stack (kernel 3x3, stride 2x2, pad 1x1)
  struct ggml_tensor *conv2d_w[3] = {nullptr, nullptr, nullptr};
  struct ggml_tensor *conv2d_b[3] = {nullptr, nullptr, nullptr};

  // Post-conv projection to d_model: [d_model, OH*OC] @ x => [d_model, OW*nch]
  struct ggml_tensor *conv_out_w  = nullptr;
  struct ggml_tensor *conv_out_b  = nullptr;

  // Per-chunk position embedding table (rows = tokens_per_chunk, cols = d_model)
  struct ggml_tensor *position_embd = nullptr;

  struct EncLayer {
    struct ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    struct ggml_tensor *q_w = nullptr, *q_b = nullptr;
    struct ggml_tensor *k_w = nullptr, *k_b = nullptr;
    struct ggml_tensor *v_w = nullptr, *v_b = nullptr;
    struct ggml_tensor *o_w = nullptr, *o_b = nullptr;
    struct ggml_tensor *q_norm_w = nullptr, *q_norm_b = nullptr; // optional
    struct ggml_tensor *k_norm_w = nullptr, *k_norm_b = nullptr; // optional
    struct ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    struct ggml_tensor *ffn_up_w  = nullptr, *ffn_up_b  = nullptr;
    struct ggml_tensor *ffn_dn_w  = nullptr, *ffn_dn_b  = nullptr;
  };
  std::vector<EncLayer> layers;

  // Optional post-encoder LayerNorm
  struct ggml_tensor *post_norm_w = nullptr;
  struct ggml_tensor *post_norm_b = nullptr;

  // 2-layer MLP projector: linear -> GELU(erf) -> linear, output is LLM-dim.
  struct ggml_tensor *mm_1_w = nullptr, *mm_1_b = nullptr;
  struct ggml_tensor *mm_2_w = nullptr, *mm_2_b = nullptr;
};

/**
 * Qwen3-ASR ASR model.
 *
 * Encode():
 *   - Treats `input_frames` as raw mono PCM @ 16 kHz (because
 *     `meta_.use_external_frontend = true` makes RSProcessor skip its Kaldi
 *     fbank and pass PCM through verbatim).
 *   - Runs the Whisper-style mel extractor, then a single graph through the
 *     conv2d stack + ViT encoder + projector to produce LLM-dim audio
 *     embeddings.
 *
 * Decode():
 *   - Splices [<|im_start|>system ... user prompt|], audio embeddings, and
 *     the assistant kick-off tokens into one prefill batch, then runs an
 *     autoregressive greedy loop (with repetition penalty) until EOS.
 */
class Qwen3ASRModel : public ISpeechModel {
public:
  Qwen3ASRModel();
  ~Qwen3ASRModel() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &state) override;

  const RSModelMeta &GetMeta() const override { return meta_; }

  void SetUseLLM(bool use) override { (void)use; /* always uses LLM */ }
  bool SupportsTwoPass() const override { return false; }
  void SetCTCPrecheck(bool enable) override { (void)enable; /* no CTC */ }
  void SetUserInputPrompt(const std::string &prompt) override;

private:
  RSModelMeta             meta_;
  Qwen3ASRHParams         hparams_;
  Qwen3ASRWeights         weights_;
  std::unique_ptr<WhisperMelExtractor> mel_;

  // LLM components
  llm_model_ptr                       llm_model_;
  std::unique_ptr<llm_build_qwen3>    llm_graph_builder_;

  // Persistent host KV mirror (mirrors FunASRNano's optimised cache).
  std::vector<std::vector<float>>     host_kv_cache_k_;
  std::vector<std::vector<float>>     host_kv_cache_v_;
  uint32_t                            n_cached_tokens_ = 0;
  static constexpr int MAX_DECODE_TOKENS = 512;

  // Prompt and audio-token IDs (cached so we don't re-tokenize per call).
  std::string user_input_prompt_;
  std::string cached_user_input_prompt_;
  std::vector<int32_t> cached_prefix_tokens_;
  std::vector<int32_t> cached_suffix_tokens_;
  int32_t audio_start_id_ = -1;
  int32_t audio_end_id_   = -1;

  bool MapEncoderTensors(std::map<std::string, struct ggml_tensor *> &tensors);
  bool LoadLLM(struct gguf_context *ctx_gguf,
               std::map<std::string, struct ggml_tensor *> &tensors,
               ggml_backend_t backend);

  // Encoder graph + execute, fills state.audio_embeds.
  bool RunEncoder(const std::vector<float> &mel_features,
                  int n_frames_padded, RSState &state,
                  ggml_backend_sched_t sched);

  bool DecodeWithLLM(RSState &state, ggml_backend_sched_t sched);
};

/**
 * Per-request state. Holds the audio-token embeddings produced by the
 * encoder (CPU side, ready to upload into the LLM prefill graph), the
 * decoded token ID stream, and the final detokenized text.
 */
struct Qwen3ASRState : public RSState {
  // [n_embd, T_audio] in row-major (T_audio fast over n_embd)? No — laid out
  // exactly the way the encoder's host buffer is contiguous so we can feed
  // it into a ggml_tensor_set() on a [n_embd, T_audio] tensor: n_embd is fast.
  std::vector<float>   audio_embeds;
  int32_t              T_audio = 0;
  int32_t              n_embd  = 0;
  std::vector<int32_t> token_ids;
  std::vector<std::string> tokens;
};
