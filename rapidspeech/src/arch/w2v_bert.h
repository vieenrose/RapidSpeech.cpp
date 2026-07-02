#pragma once

// W2V-BERT 2.0 — 18-layer Conformer encoder (hidden=1024, heads=16).
//
// Forward: SeamlessM4T fbank-160 input_features [T, 160]
//        → feature_projection (LN + linear 160→1024)
//        → 17× Macaron Conformer blocks (encoder layers 0..16)
//        → hidden17 [T, 1024]
//
// The 18-layer slice mirrors upstream IndexTTS-2:
//   hidden_states = model.forward(..., output_hidden_states=True).hidden_states
//   sc_feat = hidden_states[17]   # index 0 is feature_projection; this is
//                                 # encoder layer 16 output
//
// Layers 0..17 may be shipped in the GGUF, but the runtime stops at 0..16.
//
// Transformer block ordering (HF Wav2Vec2BertEncoderLayer):
//   FFN1 (half residual) → MHSA (relative_key) → ConvModule (GLU+dw+SiLU)
//   → FFN2 (half residual) → final_LN
//
// Reference: CrispASR conformer_block (indextts2_gpt.cpp:434-581) and
//            HF transformers Wav2Vec2BertEncoderLayer source.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace w2v_bert {

// ---------------------------------------------------------------------------
// Hyperparameters (embedded in GGUF metadata by converter)
// ---------------------------------------------------------------------------
struct HParams {
    int  n_layers      = 18;
    int  hidden        = 1024;
    int  n_heads       = 16;
    int  head_dim      = 64;    // = hidden / n_heads
    int  intermediate  = 4096;
    int  conv_kernel   = 31;
    int  left_max_pos  = 64;
    int  right_max_pos = 8;
    int  fp_input_dim  = 160;
    int  dist_emb_size = 73;    // left_max + right_max + 1
};

// ---------------------------------------------------------------------------
// One Conformer layer (all weight pointers; nullptr if not present in GGUF)
// ---------------------------------------------------------------------------
struct ConformerLayer {
    // FFN1  (Macaron half-residual)
    ggml_tensor *ffn1_ln_w   = nullptr, *ffn1_ln_b   = nullptr;
    ggml_tensor *ffn1_fc1_w  = nullptr, *ffn1_fc1_b  = nullptr;
    ggml_tensor *ffn1_fc2_w  = nullptr, *ffn1_fc2_b  = nullptr;

    // MHSA  (relative_key: Q/K/V/O + distance_embedding[73, head_dim])
    ggml_tensor *attn_ln_w   = nullptr, *attn_ln_b   = nullptr;
    ggml_tensor *attn_q_w    = nullptr, *attn_q_b    = nullptr;
    ggml_tensor *attn_k_w    = nullptr, *attn_k_b    = nullptr;
    ggml_tensor *attn_v_w    = nullptr, *attn_v_b    = nullptr;
    ggml_tensor *attn_o_w    = nullptr, *attn_o_b    = nullptr;
    ggml_tensor *attn_dist_emb = nullptr;  // [73, head_dim]  F32

    // ConvModule  (GLU + depthwise k=31 + LN + SiLU)
    ggml_tensor *conv_ln_w   = nullptr, *conv_ln_b   = nullptr;
    ggml_tensor *conv_pw1_w  = nullptr;   // [2D, D, 1] → reshaped to [D, 2D]
    ggml_tensor *conv_dw_w   = nullptr;   // [D, 1, K]
    ggml_tensor *conv_dw_ln_w= nullptr, *conv_dw_ln_b= nullptr;
    ggml_tensor *conv_pw2_w  = nullptr;   // [D, D, 1] → reshaped to [D, D]

    // FFN2  (Macaron half-residual)
    ggml_tensor *ffn2_ln_w   = nullptr, *ffn2_ln_b   = nullptr;
    ggml_tensor *ffn2_fc1_w  = nullptr, *ffn2_fc1_b  = nullptr;
    ggml_tensor *ffn2_fc2_w  = nullptr, *ffn2_fc2_b  = nullptr;

    // Final per-block LayerNorm
    ggml_tensor *final_ln_w  = nullptr, *final_ln_b  = nullptr;
};

// ---------------------------------------------------------------------------
// All weights (feature_projection + N layers)
// ---------------------------------------------------------------------------
struct Weights {
    ggml_tensor *fp_ln_w   = nullptr, *fp_ln_b   = nullptr;
    ggml_tensor *fp_proj_w = nullptr, *fp_proj_b = nullptr;
    std::vector<ConformerLayer> layers;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
class W2VBertModel {
public:
    W2VBertModel() = default;

    // Map tensors from the given ggml context. All tensors whose names start
    // with `tensor_prefix` are loaded; `ctx_gguf` supplies metadata KV pairs.
    // `backend` is the buffer owner for the GGUF tensors.
    bool Load(ggml_context *gguf_data, gguf_context *ctx_gguf,
              ggml_backend_t backend,
              const std::string &tensor_prefix = "indextts2.w2v_bert.");

// Run the encoder on one utterance. input_features[nelem = T * 160]
    // is C-order [T, 160] f32.  hidden0_out [T, 1024] is the feature_projection
    // output.  hidden17_out[T, 1024] is HF hidden_states[17] (encoder layer
    // 16 output).
    // Both are C-order [T, D] f32.
    // If hidden0_out is nullptr, it is not read back (saves a device→host copy).
    bool Forward(const float *input_features, int T,
                 float *hidden0_out, float *hidden17_out,
                 ggml_backend_sched_t sched);

    const HParams &hparams() const { return hp_; }
    ggml_backend_t backend()    const { return backend_; }

private:
    HParams        hp_;
    Weights        w_;
    ggml_backend_t backend_ = nullptr;

    // Cache the per-T (q,k) → distance gather index. Shape: [T*T]. Same index
    // is reused across all 18 conformer layers (and across forward calls when
    // T is stable). Materialized inside the graph via ggml_get_rows on each
    // layer's attn_dist_emb table to keep memory at O(T*T*4) instead of
    // O(n_layers*T*T*head_dim*4).
    int                  cached_T_ = -1;
    std::vector<int32_t> dist_gather_idx_;
};

} // namespace w2v_bert
