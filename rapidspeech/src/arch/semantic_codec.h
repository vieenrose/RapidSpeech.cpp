#pragma once

// MaskGCT semantic_codec: RepCodec encoder + VQ quantizer.
//
// Architecture (from amphion/MaskGCT/semantic_codec):
//   encoder = VocosBackbone(12-layer ConvNeXt, dim=384, intermediate=2048)
//           + Linear(384 → 1024)
//   quantizer = ResidualVQ(1× codebook 8192×8, in_proj 1024→8, out_proj 8→1024)
//
// The quantize() path (used at inference):
//   1. encoder(x)           → [T, 1024]
//   2. in_proj(x)           → [T, 8], L2-normalize
//   3. argmin |proj - codebook[i]| → codes [T]
//   4. out_proj(codebook[codes])  → sc_hidden [T, 1024]
//
// Input:  W2V-BERT features (after (feat-mean)/std normalization)
// Output: sc_hidden [T, 1024] — continuous reconstruction after VQ

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace semantic_codec {

struct HParams {
    int hidden         = 1024;   // input/output dim
    int vocos_dim      = 384;    // ConvNeXt internal dim
    int intermediate   = 2048;   // ConvNeXt FFN dim
    int num_layers     = 12;     // ConvNeXt blocks
    int codebook_size  = 8192;
    int codebook_dim   = 8;
};

// All weights bound from GGUF.
struct Weights {
    // encoder.0 (VocosBackbone)
    ggml_tensor *enc_embed_w       = nullptr;  // Conv1d: [384, 1024, 7]
    ggml_tensor *enc_embed_b       = nullptr;  // [384]
    ggml_tensor *enc_norm_w        = nullptr;  // [384]
    ggml_tensor *enc_norm_b        = nullptr;  // [384]

    struct ConvNeXtBlock {
        ggml_tensor *dwconv_w  = nullptr;  // Conv1d: [384, 1, 7]  (depthwise)
        ggml_tensor *dwconv_b  = nullptr;  // [384]
        ggml_tensor *norm_w    = nullptr;  // [384]
        ggml_tensor *norm_b    = nullptr;  // [384]
        ggml_tensor *pwconv1_w = nullptr;  // [2048, 384]
        ggml_tensor *pwconv1_b = nullptr;  // [2048]
        ggml_tensor *pwconv2_w = nullptr;  // [384, 2048]
        ggml_tensor *pwconv2_b = nullptr;  // [384]
        ggml_tensor *gamma     = nullptr;  // [384]
    };
    std::vector<ConvNeXtBlock> convnext_blocks;

    ggml_tensor *enc_final_ln_w   = nullptr;  // [384]
    ggml_tensor *enc_final_ln_b   = nullptr;  // [384]

    // encoder.1 (output projection)
    ggml_tensor *enc_out_w        = nullptr;  // [1024, 384]
    ggml_tensor *enc_out_b        = nullptr;  // [1024]

    // quantizer (ResidualVQ with weight_norm in_project/out_project)
    ggml_tensor *q_codebook       = nullptr;  // [codebook_size, codebook_dim]
    ggml_tensor *q_in_proj_w_g    = nullptr;  // weight_norm scale [D]
    ggml_tensor *q_in_proj_w_v    = nullptr;  // weight_norm dir [D, C]
    ggml_tensor *q_in_proj_b      = nullptr;  // [D]
    ggml_tensor *q_out_proj_w_g   = nullptr;  // weight_norm scale [C]
    ggml_tensor *q_out_proj_w_v   = nullptr;  // weight_norm dir [C, D]
    ggml_tensor *q_out_proj_b     = nullptr;  // [C]
};

class SemanticCodecModel {
public:
    SemanticCodecModel() = default;

    bool Load(ggml_context *gguf_data, gguf_context *ctx_gguf,
              ggml_backend_t backend,
              const std::string &tensor_prefix = "indextts2.semantic_codec.");

    // Quantize: encoder + VQ → sc_hidden [T, 1024] (C-order).
    // Also returns discrete codes [T] if codes_out != nullptr.
    bool Forward(const float *input_features, int T,
                 float *sc_hidden_out, int *codes_out,
                 ggml_backend_sched_t sched);

    // Decode discrete VQ codes through the quantizer out projection, mirroring
    // upstream `semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))`.
    bool DecodeCodes(const int32_t *codes, int T, float *sc_hidden_out);

    const HParams &hparams() const { return hp_; }

private:
    HParams        hp_;
    Weights        w_;
    ggml_backend_t backend_ = nullptr;
};

} // namespace semantic_codec
