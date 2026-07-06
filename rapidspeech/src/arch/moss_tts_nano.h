#pragma once
// MOSS-TTS-Nano-100M on ggml (Jetson Nano gen1 / ggml-CUDA).
// GPT-2 global (12L, RoPE-NEOX, gelu_new) + 1-layer local decoder over 16 RVQ
// codebooks + MOSS-Audio-Tokenizer-Nano decode (4-stage transformer, 12.5Hz->48kHz).
// Weights Q8_0 (bandwidth-bound mul_mat_vec; ~lossless, 2.4x faster on Maxwell).
#include "core/rs_model.h"
#include "ggml.h"
#include <vector>

struct rs_context_t;

struct MossHParams {
  uint32_t n_layer = 0, n_local_layer = 0, n_embd = 0, n_head = 0, n_ff = 0,
           head_dim = 0, vocab_size = 0, n_codebooks = 0, codebook_size = 0,
           sample_rate = 48000;
  float rope_base = 10000.0f;
  int audio_start = 0, audio_end = 0, audio_pad = 0, audio_user_slot = 0,
      audio_assistant_slot = 0;
};

// one GPT-2 transformer block (global or local)
struct MossBlock {
  ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
  ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;
  ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
  ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
  ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
  ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

// codec decoder stage (transformer @ hidden 256)
struct MossCodecLayer {
  ggml_tensor *in_proj_w = nullptr, *out_proj_w = nullptr;      // self-attn
  ggml_tensor *ffn0_w = nullptr, *ffn1_w = nullptr;
  ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr, *norm2_w = nullptr, *norm2_b = nullptr;
  ggml_tensor *ls1 = nullptr, *ls2 = nullptr;                    // layer scales
};
struct MossCodecStage {
  ggml_tensor *input_proj_w = nullptr, *output_proj_w = nullptr;
  std::vector<MossCodecLayer> layers;
};
struct MossRVQ {
  ggml_tensor *codebook = nullptr;                              // [code_dim, 1024]
  ggml_tensor *in_proj_w = nullptr, *in_proj_b = nullptr;
  ggml_tensor *out_proj_w = nullptr, *out_proj_b = nullptr;
};

class MossTTSNanoModel : public ISpeechModel {
public:
  MossTTSNanoModel();
  ~MossTTSNanoModel() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx, ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  // ASR-side no-ops (this is a TTS model)
  bool Encode(const std::vector<float> &, RSState &, ggml_backend_sched_t) override { return false; }
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &) override { return ""; }
  // TTS interface
  bool PushText(RSState &state, const char *text, const char *language,
                const char *instruct) override;
  bool PushReferenceAudio(RSState &state, const float *samples, int n_samples,
                          int sample_rate, ggml_backend_sched_t sched) override;
  bool PushReferenceText(RSState &state, const char *ref_text) override;
  int GetAudioOutput(RSState &state, float **out_data) override;
  void SetSeed(uint64_t seed) override { seed_ = seed; }
  const RSModelMeta &GetMeta() const override { return meta_; }

private:
  RSModelMeta meta_;
  MossHParams hp_;
  ggml_backend_t backend_ = nullptr;
  uint64_t seed_ = 0;

  // AR model tensors
  ggml_tensor *token_embd_ = nullptr, *output_norm_w_ = nullptr, *output_norm_b_ = nullptr;
  std::vector<MossBlock> blocks_;                 // global (n_layer)
  std::vector<MossBlock> local_blocks_;           // local (n_local_layer)
  ggml_tensor *local_out_norm_w_ = nullptr, *local_out_norm_b_ = nullptr;
  std::vector<ggml_tensor *> audio_embd_;         // [n_codebooks] (tied heads)
  // codec tensors
  bool codec_loaded_ = false;
  std::vector<MossCodecStage> codec_stages_;
  ggml_tensor *codec_q_in_w_ = nullptr, *codec_q_in_b_ = nullptr;
  ggml_tensor *codec_q_out_w_ = nullptr, *codec_q_out_b_ = nullptr;
  std::vector<MossRVQ> codec_rvq_;

  bool LoadHParams(const rs_context_t &ctx);
  bool LoadTensors(const rs_context_t &ctx);
  bool LoadCodec(const rs_context_t &ctx);
};
