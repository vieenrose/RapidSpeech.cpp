#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * CosyVoice3 speech_tokenizer_v3 — 16 kHz wav → 6561-vocab speech tokens.
 *
 * Architecture (matches the upstream FunAudioLLM S3 V3 encoder):
 *   16 kHz mono PCM
 *     → 128-mel log spectrogram (n_fft=400, hop=160, Slaney, log10 + global
 *        clip max-8 + (x+4)/4, drop last frame — same as Whisper / S3 V2)
 *     → encoder.conv1 (128→1280, k=3, s=2, p=1) + bias + GELU
 *     → encoder.conv2 (1280→1280, k=3, s=2, p=1) + bias + GELU       → (T/4, 1280)
 *     → 12 × FSMN-attn block:
 *         attn_ln (gamma, beta; eps=1e-6)
 *         Q/K/V projections (K has no bias)
 *         FSMN: Transpose → Pad sym 15/15 → DW Conv1d(k=31, g=1280)
 *               → Transpose back, add V residual
 *         (NO RoPE)
 *         MultiHeadAttn (heads=20, head_dim=64, scale=1/sqrt(64))
 *         out_proj + bias  → + FSMN → residual
 *         mlp_ln (gamma, beta; eps=1e-5)
 *         mlp.0 (1280→5120) + GELU → mlp.2 (5120→1280) → residual
 *     → quant.pd (1280→8)
 *     → FSQ: h = tanh(h)·0.999 → round → +1 → token = Σᵢ 3ⁱ · hᵢ
 *
 * Output: T_tok = ⌈T_mel/4⌉ int32 tokens in [0, 6561).
 */
class CosyVoice3SpeechTokenizer {
public:
  CosyVoice3SpeechTokenizer() = default;
  ~CosyVoice3SpeechTokenizer();

  bool Load(gguf_context *ctx_gguf, ggml_context *gguf_data,
            ggml_backend_t backend);

  // Returns 0 on success (tokens_out populated), -1 on failure.
  int Tokenize(const float *pcm_16k, int n_samples,
               std::vector<int32_t> &tokens_out,
               ggml_backend_sched_t sched);

  bool loaded() const { return loaded_; }
  int  dim()    const { return dim_; }
  int  depth()  const { return depth_; }

private:
  // -------------------------------------------------------------------------
  // Hparams (read from cosyvoice3.tokenizer.* KVs).
  // -------------------------------------------------------------------------
  bool loaded_ = false;
  int  dim_           = 1280;
  int  depth_         = 12;
  int  n_heads_       = 20;
  int  head_dim_      = 64;
  int  ff_dim_        = 5120;
  int  fsmn_kernel_   = 31;
  int  fsmn_pad_l_    = 15;
  int  fsmn_pad_r_    = 15;
  int  codebook_      = 6561;
  int  proj_dim_      = 8;
  int  n_mels_        = 128;
  bool use_rope_      = true;
  float rope_theta_   = 10000.0f;
  int   rope_max_pos_ = 4096;
  float fsq_gain_     = 0.9990000128746033f;
  float attn_ln_eps_  = 1e-6f;
  float mlp_ln_eps_   = 1e-5f;

  // -------------------------------------------------------------------------
  // Bound weight tensors.
  // -------------------------------------------------------------------------
  ggml_tensor *conv1_w_ = nullptr, *conv1_b_ = nullptr;  // (3, 128, 1280)
  ggml_tensor *conv2_w_ = nullptr, *conv2_b_ = nullptr;  // (3, 1280, 1280)
  ggml_tensor *pd_w_    = nullptr, *pd_b_    = nullptr;  // (1280, 8)

  struct Block {
    ggml_tensor *attn_ln_w = nullptr, *attn_ln_b = nullptr;
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr;                          // no K bias
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *fsmn_w = nullptr;                       // (31, 1, 1280)
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *mlp_ln_w = nullptr, *mlp_ln_b = nullptr;
    ggml_tensor *mlp0_w = nullptr, *mlp0_b = nullptr;    // 1280 → 5120
    ggml_tensor *mlp2_w = nullptr, *mlp2_b = nullptr;    // 5120 → 1280
  };
  std::vector<Block> blocks_;

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  bool LoadHparams(gguf_context *ctx_gguf);
  bool LoadTensors(const std::map<std::string, ggml_tensor *> &tensors);

  // Build encoder graph: conv front-end + 12 FSMN-attn blocks + project_down.
  // Returns the (proj_dim=8, T_tok) tensor — caller does FSQ on the host.
  ggml_tensor *BuildEncoderGraph(ggml_context *ctx, ggml_cgraph *gf,
                                 ggml_tensor *mel_in, int T_mel) const;
};
