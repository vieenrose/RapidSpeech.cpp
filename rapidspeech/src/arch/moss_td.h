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
#include <functional>
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
struct MossTDHParams {
  // Audio frontend (Whisper-medium: 80 mel, 30 s = 3000 frames per chunk)
  int32_t n_mels         = 80;
  int32_t n_fft          = 400;
  int32_t hop_length     = 160;
  int32_t sample_rate    = 16000;
  int32_t chunk_size     = 3000;  // mel frames per 30 s chunk (whole encoder)

  // Encoder transformer
  int32_t enc_d_model    = 0;
  int32_t enc_n_head     = 0;
  int32_t enc_n_layer    = 0;
  int32_t enc_ffn_dim    = 0;
  int32_t enc_max_pos    = 1500;  // = position_embeddings rows (after conv s2)
  float   enc_norm_eps   = 1e-5f;

  // VQAdaptor: merge `merge_size` consecutive encoder frames (d_model each)
  // -> mm_in_dim (= d_model*merge_size), then project to mm_out_dim (LLM dim).
  int32_t merge_size     = 4;
  int32_t mm_in_dim      = 0;
  int32_t mm_out_dim     = 0;
  int32_t audio_token_id = 151671;
  float   mm_norm_eps    = 1e-6f;

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
struct MossTDWeights {
  // Whisper conv1d stem: conv1 (n_mels->d_model, k3, s1, p1) + GELU,
  //                      conv2 (d_model->d_model, k3, s2, p1) + GELU.
  struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
  struct ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;

  // Full learned/sinusoidal position embedding table [d_model, max_pos(1500)],
  // added to the whole sequence (not per-chunk).
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

  // Post-encoder LayerNorm (Whisper final layer_norm)
  struct ggml_tensor *post_norm_w = nullptr;
  struct ggml_tensor *post_norm_b = nullptr;

  // VQAdaptor over merge-4 features: Linear(in_dim->d) [mm_1] -> SiLU ->
  // Linear(d->d) [mm_2] -> LayerNorm(d) [mm_norm].  Output is LLM-dim.
  struct ggml_tensor *mm_1_w = nullptr, *mm_1_b = nullptr;
  struct ggml_tensor *mm_2_w = nullptr, *mm_2_b = nullptr;
  struct ggml_tensor *mm_norm_w = nullptr, *mm_norm_b = nullptr;
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
class MossTDModel : public ISpeechModel {
public:
  MossTDModel();
  ~MossTDModel() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;

  // Run the encoder directly from externally-computed log-mel features
  // (layout [n_mel, n_frames] C-order = mel-bin slow / frame fast), bypassing
  // the internal WhisperMelExtractor. Used by the WASM/web path, which computes
  // an HF-exact mel in JS. `mel` length must be a multiple of n_mels; the frame
  // count is padded up to a 30 s (3000-frame) chunk multiple internally by the
  // caller, or here if needed.
  // keep_tokens > 0 truncates the produced audio tokens to that count (the
  // real-audio token length = (n_samples-1)/1280 + 1), dropping the trailing
  // silence tokens that a 30 s-padded window would otherwise emit (which make
  // the decoder loop/hallucinate past the real audio). 0 keeps all tokens.
  bool EncodeMel(const std::vector<float> &mel, int n_frames, RSState &state,
                 ggml_backend_sched_t sched, int keep_tokens = 0);
  std::string GetTranscription(RSState &state) override;

  // Per-token streaming hook: invoked after every generated token with the
  // full decoded-so-far text of the current window (empty to disable). Used by
  // the WASM worker to postMessage a live transcript to the main thread.
  void SetOnToken(std::function<void(const std::string &)> cb) {
    on_token_ = std::move(cb);
  }

  // Phase-progress hook: invoked at pipeline milestones so slow phases
  // (multi-chunk encoder, long-window prefill) can show live progress instead
  // of a silent multi-minute gap. phase is "mel" | "encode" | "prefill" |
  // "decode"; cur/total are phase-specific (encode: chunk index/count;
  // prefill: 0/ctx_tokens at start, total/total at end).
  void SetOnPhase(std::function<void(const char *, int, int)> cb) {
    on_phase_ = std::move(cb);
  }

  const RSModelMeta &GetMeta() const override { return meta_; }

  void SetUseLLM(bool use) override { (void)use; /* always uses LLM */ }
  bool SupportsTwoPass() const override { return false; }
  void SetCTCPrecheck(bool enable) override { (void)enable; /* no CTC */ }
  void SetUserInputPrompt(const std::string &prompt) override;

private:
  RSModelMeta             meta_;
  MossTDHParams         hparams_;
  MossTDWeights         weights_;
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
  std::function<void(const std::string &)> on_token_;  // streaming hook
  std::function<void(const char *, int, int)> on_phase_;  // phase progress
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
struct MossTDState : public RSState {
  // [n_embd, T_audio] in row-major (T_audio fast over n_embd)? No — laid out
  // exactly the way the encoder's host buffer is contiguous so we can feed
  // it into a ggml_tensor_set() on a [n_embd, T_audio] tensor: n_embd is fast.
  std::vector<float>   audio_embeds;
  int32_t              T_audio = 0;
  int32_t              n_embd  = 0;
  std::vector<int32_t> token_ids;
  std::vector<std::string> tokens;
};
