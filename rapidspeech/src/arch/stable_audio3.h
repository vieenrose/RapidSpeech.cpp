#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "llm_graph.h"
#include "llm_model.h"
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Stable Audio 3 (small-music) — text-conditioned audio/music generation.
 *
 * Implemented against the official PyTorch/MLX source (the clean MLX ports in
 * optimized/mlx/models/defs/ are the reference: dit_mlx.py, same_s_decoder.py,
 * sa3_pipeline.py). Built on RapidSpeech primitives (ISpeechModel + hand-built
 * ggml graphs), no external abstraction layer.
 *
 * Pipeline:
 *   prompt ─► T5Gemma ─► cross_attn_cond [B,257,768]
 *   seconds_total ─► expo-fourier ─► global_cond [B,768]
 *   noise [256,T_lat] ─► DiT (20 blocks, ping-pong rectified flow, CFG) ─► latents
 *   latents ─► SAME-S decoder ─► [512,T_lat*16] ─► patched_decode ─► [2,T_lat*4096] @ 44.1k
 *
 * Status: DiT/sampler/SAME-S faithfully translated from the MLX reference but
 * UNTESTED (no compile/checkpoint validation here). Use RS_SA3_DUMP_LATENTS to
 * cross-check the diffusion output against the reference before the decoder.
 * T5Gemma text encoder is still a stub (runs unconditional until wired).
 */

#define SA3_MAX_NODES 16384

// =====================================================================
// Hyperparameters (small-music defaults; read from `stable-audio3.*` KVs)
// =====================================================================

struct StableAudio3HParams
{
    // SAME / audio
    int32_t latent_dim        = 256;
    int32_t audio_sample_rate = 44100;
    int32_t audio_channels    = 2;
    int32_t patch_size        = 256;     // SAME patch -> samples; 16*256 = 4096
    int32_t same_downsample   = 4096;

    // DiT (ContinuousTransformer)
    int32_t dit_embed_dim   = 1024;
    int32_t dit_depth       = 20;
    int32_t dit_n_head      = 16;
    int32_t dit_head_dim    = 64;
    int32_t dit_rope_dims   = 32;        // half of head_dim
    int32_t dit_ff_inner    = 4096;
    int32_t dit_n_memory    = 64;        // learned memory tokens
    int32_t cond_token_dim  = 768;       // T5Gemma hidden
    int32_t global_cond_dim = 768;
    int32_t local_cond_dim  = 257;       // inpaint mask(1)+masked input(256)
    int32_t timestep_feat   = 256;       // ExpoFourier dim
    float   rms_eps         = 1e-5f;
    float   qk_rms_eps      = 1e-6f;

    // T5Gemma text encoder (google/t5gemma-b-b-ul2; Gemma2-style)
    int32_t txt_hidden   = 768;
    int32_t txt_layers   = 12;
    int32_t txt_heads    = 12;
    int32_t txt_head_dim = 64;
    int32_t txt_inter    = 2048;          // GeGLU intermediate
    int32_t txt_vocab    = 256000;
    int32_t txt_max_len  = 256;
    int32_t txt_pad_id   = 0;
    float   txt_rope_theta = 10000.0f;
    float   txt_rms_eps  = 1e-6f;
    float   txt_softcap  = 50.0f;
    int32_t txt_q_scalar = 64;            // query_pre_attn_scalar

    // SAME-S decoder
    int32_t same_dim        = 768;
    int32_t same_depth      = 6;
    int32_t same_n_head     = 12;
    int32_t same_head_dim   = 64;
    int32_t same_rope_dims  = 32;
    int32_t same_ff_inner   = 2304;
    int32_t same_out_ch     = 512;       // = audio_channels * patch_size
    int32_t same_stride     = 16;        // patches per latent
    float   same_qk_eps     = 1e-3f;     // DyT q/k norm uses DyT, eps unused

    // Sampling (ping-pong rectified flow)
    int32_t n_sample_steps = 8;
    float   cfg_scale      = 7.0f;       // TODO(weights): confirm default
    float   logsnr_anchor  = -6.2f;
    float   logsnr_end     = 2.0f;
    float   seconds_min    = 0.0f;
    float   seconds_max    = 384.0f;

    float   max_duration_s = 120.0f;
    float   eps            = 1e-6f;
};

// =====================================================================
// Per-request state
// =====================================================================

struct StableAudio3State : public RSState
{
    std::string prompt;
    float       duration_s = 10.0f;

    // T5Gemma cross-attn conditioning, pre-projected layout [cond_token_dim, T_text].
    std::vector<float> text_cond;
    int                n_text_tokens = 0;

    // diffusion working latents [latent_dim, T_lat]
    std::vector<float> latents;
    int                n_latent_frames = 0;

    // output interleaved stereo @ 44.1k
    std::vector<float> audio_output;
    bool               done = false;

    std::mt19937 rng_{42};
    std::mt19937 &rng() { return rng_; }
};

// =====================================================================
// DiT block weights (ContinuousTransformer layer; see dit_mlx.py)
// =====================================================================

struct SA3DiTBlock
{
    ggml_tensor *pre_norm = nullptr;             // RMSNorm

    // self-attn (fused qkv, no bias)
    ggml_tensor *self_qkv = nullptr;             // [embed, 3*embed]
    ggml_tensor *self_out = nullptr;             // [embed, embed]
    ggml_tensor *self_q_norm = nullptr;          // RMSNorm(head_dim)
    ggml_tensor *self_k_norm = nullptr;

    // cross-attn (separate q / kv)
    ggml_tensor *cross_norm = nullptr;           // RMSNorm (cross_attend_norm)
    ggml_tensor *cross_q  = nullptr;             // [embed, embed]
    ggml_tensor *cross_kv = nullptr;             // [cond, 2*cond]
    ggml_tensor *cross_out = nullptr;
    ggml_tensor *cross_q_norm = nullptr;
    ggml_tensor *cross_k_norm = nullptr;

    // ff (GLU)
    ggml_tensor *ff_norm = nullptr;              // RMSNorm
    ggml_tensor *ff_glu_w = nullptr, *ff_glu_b = nullptr;   // [embed, 2*ff_inner]
    ggml_tensor *ff_out_w = nullptr, *ff_out_b = nullptr;   // [ff_inner, embed]

    // modulation + local cond
    ggml_tensor *scale_shift_gate = nullptr;     // [6*embed] learned bias
    ggml_tensor *local0_w = nullptr, *local0_b = nullptr;   // [257, embed]
    ggml_tensor *local2_w = nullptr, *local2_b = nullptr;   // [embed, embed]
};

// =====================================================================
// SAME-S decoder block (differential attention + DyT; see same_s_decoder.py)
// =====================================================================

struct SA3DyT { ggml_tensor *alpha = nullptr, *gamma = nullptr, *beta = nullptr; };

struct SA3SameBlock
{
    SA3DyT       pre_norm;
    ggml_tensor *to_qkv = nullptr;               // [dim, 5*dim] no bias
    ggml_tensor *to_out = nullptr;               // [dim, dim] no bias
    SA3DyT       q_norm;                          // DyT(head_dim)
    SA3DyT       k_norm;
    SA3DyT       ff_norm;
    ggml_tensor *ff_glu_w = nullptr, *ff_glu_b = nullptr;  // [dim, 2*ff_inner]
    ggml_tensor *ff_out_w = nullptr, *ff_out_b = nullptr;  // [ff_inner, dim]
};

// =====================================================================
// T5Gemma encoder layer (Gemma2-style sandwich norm; see t5gemma_mlx.py)
// =====================================================================

struct SA3T5Layer
{
    ggml_tensor *q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr, *o_proj = nullptr;
    ggml_tensor *gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
    ggml_tensor *n_pre_attn = nullptr, *n_post_attn = nullptr;   // RMSNorm weights
    ggml_tensor *n_pre_ff = nullptr,   *n_post_ff = nullptr;
};

// SentencePiece-unigram tokenizer (Viterbi), vocab+scores from GGUF.
struct SA3T5Tokenizer
{
    std::vector<std::string>                  id_to_token;
    std::unordered_map<std::string, int32_t>  token_to_id;
    std::vector<float>                        scores;
    bool loaded = false;
    bool load(std::vector<std::string> toks, std::vector<float> sc);
    std::vector<int32_t> encode(const std::string &text) const;  // ▁-prefixed unigram
};

// =====================================================================
// Stable Audio 3 Model
// =====================================================================

class StableAudio3Model : public ISpeechModel
{
public:
    StableAudio3Model();
    ~StableAudio3Model() override;

    bool Load(const std::unique_ptr<rs_context_t> &ctx, ggml_backend_t backend) override;
    std::shared_ptr<RSState> CreateState() override;

    bool Encode(const std::vector<float> &input_frames, RSState &state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState &state, ggml_backend_sched_t sched) override;

    std::string GetTranscription(RSState &state) override { (void)state; return ""; }
    const RSModelMeta &GetMeta() const override { return meta_; }

    bool PushText(RSState &state, const char *text, const char *language = nullptr,
                  const char *instruct = nullptr) override;
    int  GetAudioOutput(RSState &state, float **out_data) override;
    void SetDiffusionSteps(int n_steps) override { hparams_.n_sample_steps = n_steps; }
    void SetSeed(uint64_t seed) override { seed_ = seed; }

    void SetDurationSeconds(float s) { pending_duration_s_ = s; }
    void SetCfgScale(float s) { hparams_.cfg_scale = s; }

    void set_imatrix_callback(std::function<void(struct ggml_tensor *)> cb)
    { imatrix_cb_ = std::move(cb); }

private:
    RSModelMeta         meta_;
    StableAudio3HParams hparams_;
    uint64_t            seed_ = 42;
    float               pending_duration_s_ = 10.0f;

    ggml_backend_t backend_ = nullptr;
    ggml_context  *gguf_data_ = nullptr;

    // Fourier/sinusoid tables (own buffer)
    ggml_tensor          *timestep_freqs_ = nullptr;   // [timestep_feat/2]
    ggml_tensor          *seconds_freqs_  = nullptr;   // [timestep_feat/2]
    ggml_backend_buffer_t tables_buf_ = nullptr;
    ggml_context         *tables_ctx_ = nullptr;

    // T5Gemma encoder
    ggml_tensor *t5_embed_ = nullptr;     // [hidden, vocab]
    ggml_tensor *t5_norm_  = nullptr;     // final RMSNorm [hidden]
    std::vector<SA3T5Layer> t5_layers_;
    SA3T5Tokenizer t5_tok_;
    bool t5_loaded_ = false;

    // --- DiT top-level weights ---
    ggml_tensor *pre_conv_ = nullptr, *post_conv_ = nullptr;       // Conv1d k=1
    ggml_tensor *to_cond0_ = nullptr, *to_cond2_ = nullptr;        // 768->E->E (no bias)
    ggml_tensor *to_global0_ = nullptr, *to_global2_ = nullptr;    // 768->E->E (no bias)
    ggml_tensor *to_tstep0_w_ = nullptr, *to_tstep0_b_ = nullptr;  // 256->E (bias)
    ggml_tensor *to_tstep2_w_ = nullptr, *to_tstep2_b_ = nullptr;  // E->E (bias)
    ggml_tensor *project_in_ = nullptr, *project_out_ = nullptr;   // 256<->E (no bias)
    ggml_tensor *memory_tokens_ = nullptr;                         // [E, n_memory]
    ggml_tensor *gce0_w_ = nullptr, *gce0_b_ = nullptr;            // E->E (bias)
    ggml_tensor *gce2_w_ = nullptr, *gce2_b_ = nullptr;            // E->6E (bias)
    std::vector<SA3DiTBlock> dit_blocks_;

    // --- duration (seconds_total) conditioner ---
    ggml_tensor *seconds_w_ = nullptr, *seconds_b_ = nullptr;      // [256,768]/[768]
    ggml_tensor *prompt_pad_ = nullptr;                            // [768]

    // --- SAME-S decoder weights ---
    ggml_tensor *same_running_std_ = nullptr;     // scalar
    ggml_tensor *same_proj_in_w_ = nullptr, *same_proj_in_b_ = nullptr;  // 256->768
    ggml_tensor *same_new_tokens_ = nullptr;      // [768]
    ggml_tensor *same_map_w_ = nullptr, *same_map_b_ = nullptr;          // Conv1d 768->512 k3
    std::vector<SA3SameBlock> same_blocks_;
    bool same_loaded_ = false;

    std::vector<float> t_schedule_;   // ping-pong schedule [n_steps+1]
    std::function<void(struct ggml_tensor *)> imatrix_cb_;

    // load
    bool LoadHParams(gguf_context *ctx_gguf);
    bool LoadTextEncoder(const std::unique_ptr<rs_context_t> &ctx,
                         std::map<std::string, ggml_tensor *> &t);
    bool LoadDiT(std::map<std::string, ggml_tensor *> &t);
    bool LoadConditioner(std::map<std::string, ggml_tensor *> &t);
    bool LoadSameDecoder(std::map<std::string, ggml_tensor *> &t);
    bool AllocTables(ggml_backend_t backend);
    void BuildSchedule();

    // compute
    bool RunTextEncoder(StableAudio3State &s, ggml_backend_sched_t sched);
    bool RunDiffusion(StableAudio3State &s, ggml_backend_sched_t sched);
    bool RunSameDecoder(StableAudio3State &s, ggml_backend_sched_t sched);

    // T5Gemma: tokenize + run encoder -> text_cond [hidden, S] into state.
    bool LoadT5Tokenizer(gguf_context *ctx_gguf);
    ggml_tensor *BuildT5Encoder(ggml_context *ctx, ggml_tensor *ids,
                                ggml_tensor *add_mask, int64_t S) const;

    // DiT velocity v(x,t,cross,global) for the CFG batch; returns [256,T,B].
    ggml_tensor *BuildDiTForward(ggml_context *ctx, ggml_tensor *x,
                                 float t_value, ggml_tensor *cross_cond,
                                 ggml_tensor *global_cond,
                                 ggml_tensor *&pos_out) const;
    ggml_tensor *BuildDiTBlock(ggml_context *ctx, const SA3DiTBlock &b,
                               ggml_tensor *x, ggml_tensor *context,
                               ggml_tensor *g6, ggml_tensor *pos) const;
    ggml_tensor *BuildSameBlock(ggml_context *ctx, const SA3SameBlock &b,
                                ggml_tensor *x) const;
};
