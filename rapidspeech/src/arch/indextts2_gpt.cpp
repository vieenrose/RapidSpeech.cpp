// IndexTTS-2 AR forward path: Conformer + Perceiver + GPT-2 + sampling.
//
// Layout mirrors upstream indextts/gpt/model_v2.py:UnifiedVoice:
//
//   semantic_codec.quantize(w2v_hidden) → S_ref [B, T_sc, sc.hidden=1024]
//        ↓ conditioning_encoder (Conformer 6L/8h/512d)
//   S_cond [B, T_sc, 512]
//        ↓ perceiver_encoder (PerceiverResampler 32 latents → 1280d)
//   cond_latents [B, 32, 1280]
//
//   (emo path: emo_conditioning_encoder 4L → emo_perceiver_encoder 1 latent →
//    emovec_layer 1024→1280 → merge_emovec → emo_emb)
//
//   prefix = [cond_latents, text_emb+pos, start_mel] + emo_emb broadcast
//        ↓ GPT-2 (24L, 1280d, 20h, KV cached)
//        ↓ final_norm + mel_head
//        ↓ top-k/top-p sample → next mel code
//   Repeat until stop_mel(=8193) or max_mel_tokens.
//
//   Second pass: re-run GPT-2 over the full sequence with all sampled codes,
//   read hidden states after final_norm at the mel positions → lm_latent.

#include "indextts2.h"

#include "core/rs_context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace indextts2 {

// -----------------------------------------------------------------------------
// GPT-2 KV-cache + compute scratch (per-state would be ideal; we keep it on the
// Model for now and rebuild on first call).
// -----------------------------------------------------------------------------
struct Model::GPTContext {
    // Layout: [head_dim, kv_max_ctx, n_head, n_layer] — matches the 1.5
    // reference's KV cache. Stored as a single F16 tensor per K and V.
    ggml_context        *ctx_kv = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;
    ggml_tensor         *kv_k   = nullptr;
    ggml_tensor         *kv_v   = nullptr;

    int n_past     = 0;
    int kv_max_ctx = 0;

    std::vector<uint8_t> compute_meta;
};

// -----------------------------------------------------------------------------
// CPU tensor helpers
// -----------------------------------------------------------------------------
static void tensor_get_f32(ggml_tensor *t, float *dst, size_t n_elem) {
    if (!t) { std::memset(dst, 0, n_elem * sizeof(float)); return; }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst, 0, n_elem * sizeof(float));
        return;
    }
    // Generic path: round-trip through ggml's quant→f32 conversion.
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    if (t->type == GGML_TYPE_F16) {
        auto *src = reinterpret_cast<ggml_fp16_t *>(raw.data());
        for (size_t i = 0; i < n_elem; ++i) dst[i] = ggml_fp16_to_fp32(src[i]);
        return;
    }
    // Quantized tables (e.g. embeddings quantized for CUDA): dequantize via
    // ggml's per-type to_float. Without this the host read would be garbage and
    // the AR loop would misbehave (embeds ≈ 0 → never emits stop token).
    {
        const auto *tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) {
            tt->to_float(raw.data(), dst, (int64_t)n_elem);
            return;
        }
    }
    // Truly unsupported type: zero rather than crash.
    std::memset(dst, 0, n_elem * sizeof(float));
}

static void set_position_ids(ggml_cgraph *gf, int n_past, int n_tokens) {
    ggml_tensor *pos = ggml_graph_get_tensor(gf, "position_ids");
    if (!pos) return;
    std::vector<int32_t> ids((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) ids[(size_t)i] = n_past + i;
    ggml_backend_tensor_set(pos, ids.data(), 0, ids.size() * sizeof(int32_t));
}


// -----------------------------------------------------------------------------
// Build the GPT-2 graph (KV-cached). Mirrors the IndexTTS 1.5 reference and is
// shape-compatible with the HuggingFace GPT-2 Conv1D layout (weights stored as
// [in_features, out_features] — `ggml_mul_mat(W, x)` does the right thing
// because ggml's mul_mat is `dst[i,j] = sum_k W[k,i] * x[k,j]`).
// -----------------------------------------------------------------------------
static ggml_cgraph *build_gpt_step_graph(const Model &model_dummy_unused,
                                         const HParams &hp,
                                         const std::vector<GPT2Block> &blocks,
                                         ggml_tensor *gpt_ln_f_w,
                                         ggml_tensor *gpt_ln_f_b,
                                         ggml_tensor *final_norm_w,
                                         ggml_tensor *final_norm_b,
                                         ggml_tensor *mel_head_w,
                                         ggml_tensor *mel_head_b,
                                         ggml_tensor *gpt_wpe_w,
                                         Model::GPTContext &gc,
                                         int n_past, int n_tokens) {
    (void)model_dummy_unused;
    const int D = hp.n_embd;
    const int n_h = hp.n_head;
    const int hd = hp.head_dim;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_tokens;
    const int Lk = n_past + T;

    ggml_init_params ip = { gc.compute_meta.size(), gc.compute_meta.data(),
                            true };
    ggml_context *ctx0 = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor *embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor *causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor *cur = embeds;
    if (gpt_wpe_w) {
        ggml_tensor *pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(pos, "position_ids");
        ggml_set_input(pos);
        ggml_tensor *pos_emb = ggml_get_rows(ctx0, gpt_wpe_w, pos);
        cur = ggml_add(ctx0, cur, pos_emb);
    }

    for (int il = 0; il < (int)blocks.size(); ++il) {
        const GPT2Block &b = blocks[il];
        ggml_tensor *residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor *x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.ln1_w);
        x = ggml_add(ctx0, x, b.ln1_b);

        // Fused QKV (HF Conv1D: weight [D, 3D])
        ggml_tensor *qkv = ggml_mul_mat(ctx0, b.attn_qkv_w, x);
        qkv = ggml_add(ctx0, qkv, b.attn_qkv_b);

        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor *Q = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor *K = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts);
        ggml_tensor *V = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts);
        if (T > 1) { Q = ggml_cont(ctx0, Q); K = ggml_cont(ctx0, K); V = ggml_cont(ctx0, V); }

        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);

        ggml_tensor *K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor *V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        ggml_tensor *k_view = ggml_view_4d(
            ctx0, gc.kv_k, hd, T, n_h, 1, gc.kv_k->nb[1], gc.kv_k->nb[2],
            gc.kv_k->nb[3],
            (size_t)il * gc.kv_k->nb[3] + (size_t)n_past * gc.kv_k->nb[1]);
        ggml_tensor *v_view = ggml_view_4d(
            ctx0, gc.kv_v, hd, T, n_h, 1, gc.kv_v->nb[1], gc.kv_v->nb[2],
            gc.kv_v->nb[3],
            (size_t)il * gc.kv_v->nb[3] + (size_t)n_past * gc.kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));

        ggml_tensor *k_layer_view = ggml_view_3d(
            ctx0, gc.kv_k, hd, Lk, n_h, gc.kv_k->nb[1], gc.kv_k->nb[2],
            (size_t)il * gc.kv_k->nb[3]);
        ggml_tensor *v_layer_view = ggml_view_3d(
            ctx0, gc.kv_v, hd, Lk, n_h, gc.kv_v->nb[1], gc.kv_v->nb[2],
            (size_t)il * gc.kv_v->nb[3]);
        // flash_attn_ext reads K/V through their nb strides, so the padded
        // [hd, Lk, n_h] view into the [hd, kv_max_ctx, n_h] cache works directly.
        // Skipping ggml_cont here avoids copying the whole (growing) KV history
        // every decode step — the dominant per-token cost in the AR loop.
        ggml_tensor *Kfull = (T == 1) ? k_layer_view : ggml_cont(ctx0, k_layer_view);
        ggml_tensor *Vfull = (T == 1) ? v_layer_view : ggml_cont(ctx0, v_layer_view);

        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        ggml_tensor *attn = ggml_flash_attn_ext(
            ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask,
            attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        attn = ggml_mul_mat(ctx0, b.attn_proj_w, attn);
        attn = ggml_add(ctx0, attn, b.attn_proj_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.ln2_w);
        x = ggml_add(ctx0, x, b.ln2_b);

        ggml_tensor *mlp = ggml_mul_mat(ctx0, b.mlp_fc_w, x);
        mlp = ggml_add(ctx0, mlp, b.mlp_fc_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, b.mlp_proj_w, mlp);
        mlp = ggml_add(ctx0, mlp, b.mlp_proj_b);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // HF GPT-2 final ln_f
    if (gpt_ln_f_w && gpt_ln_f_b) {
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, gpt_ln_f_w);
        cur = ggml_add(ctx0, cur, gpt_ln_f_b);
    }

    // Expose full-sequence hidden (post gpt.ln_f, pre last-row slice) for
    // bit-exact verification against PT.
    ggml_tensor *hidden_full = cur;
    ggml_set_name(hidden_full, "gpt_last_hidden_full");
    ggml_set_output(hidden_full);
    ggml_build_forward_expand(gf, hidden_full);

    if (T > 1) {
        cur = ggml_view_2d(ctx0, hidden_full, D, 1, hidden_full->nb[1],
                           (size_t)(T - 1) * hidden_full->nb[1]);
    }
    // IndexTTS-2 final_norm (separate from gpt.ln_f)
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, final_norm_w);
    cur = ggml_add(ctx0, cur, final_norm_b);
    ggml_set_name(cur, "final_norm_out");
    ggml_set_output(cur);

    // mel_head
    cur = ggml_mul_mat(ctx0, mel_head_w, cur);
    if (mel_head_b) cur = ggml_add(ctx0, cur, mel_head_b);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// -----------------------------------------------------------------------------
// Top-k / Top-p sampler with optional temperature and repetition penalty.
// Mirrors upstream's HF generation utility defaults: temperature=0.8, top_p=0.8,
// top_k=30, repetition_penalty=10.0.
// -----------------------------------------------------------------------------
static int sample_top_kp(std::mt19937 &rng, std::vector<float> logits,
                         float temperature, int top_k, float top_p,
                         float rep_penalty,
                         const std::vector<int32_t> &history,
                         int stop_token) {
    const int V = (int)logits.size();
    if (rep_penalty > 1.0f) {
        for (int t : history) {
            if (t < 0 || t >= V) continue;
            float &l = logits[t];
            l = (l > 0) ? (l / rep_penalty) : (l * rep_penalty);
        }
    }
    if (temperature <= 0.0f) {
        // Greedy: argmax (still respects stop semantics).
        return (int)std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
    }
    for (float &l : logits) l /= temperature;

    // Stable softmax.
    float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(V);
    float Z = 0.0f;
    for (int i = 0; i < V; ++i) { probs[i] = std::exp(logits[i] - mx); Z += probs[i]; }
    if (Z <= 0.0f) return stop_token;
    for (float &p : probs) p /= Z;

    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    int k = (top_k > 0 && top_k < V) ? top_k : V;
    float cum = 0.0f;
    int keep = 0;
    for (; keep < k; ++keep) {
        cum += probs[idx[keep]];
        if (cum >= top_p && keep + 1 >= 1) { keep++; break; }
    }
    if (keep <= 0) keep = 1;

    float Zk = 0.0f;
    for (int i = 0; i < keep; ++i) Zk += probs[idx[i]];
    if (Zk <= 0.0f) return idx[0];

    std::uniform_real_distribution<float> U(0.0f, Zk);
    float r = U(rng);
    float acc = 0.0f;
    for (int i = 0; i < keep; ++i) {
        acc += probs[idx[i]];
        if (acc >= r) return idx[i];
    }
    return idx[keep - 1];
}

// -----------------------------------------------------------------------------
// Build the conditioning embedding prefix:
//   [cond_latents (32 × D), text_emb (start, t1..tn, stop) + text_pos,
//    start_mel + mel_pos_0]
// Optionally adds the emo embedding broadcast at every cond position
// (merge_emovec is a per-token additive merge in upstream).
// -----------------------------------------------------------------------------
// Build the GPT-2 prefill embedding sequence as IndexTTS-2 v2 assembles it:
//
//   inputs_embeds = [ cond_full(cond_num+2, D),
//                     text_emb(start_text) + text_pos(0),
//                     text_emb(text_tokens[i]) + text_pos(i+1),
//                     text_emb(stop_text) + text_pos(L+1),
//                     mel_emb(start_mel) + mel_pos(0) ]
//
// where cond_full is the [cond_num+2, D] block returned by upstream
// (`speech_conditioning_latent + emo_vec` for the 32 spk latents, then
// duration_emb_half = speed_emb(1), then duration_emb = speed_emb(0)).
// The single-batch inference path always has padding=0 so we don't model it.
static std::vector<float> build_prefix_embeds(
    const HParams &hp,
    const std::vector<float> &text_emb_table,    // [text_vocab, D]
    const std::vector<float> &text_pos_table,    // [max_text_tok+2, D]
    const std::vector<float> &mel_emb_table,     // [mel_codes, D]
    const std::vector<float> &mel_pos_table,     // [max_mel_tok+3, D]
    const std::vector<float> &cond_full,         // [cond_num+2, D]
    const std::vector<int32_t> &text_tokens) {
    const int D = hp.n_embd;
    const int cond_len = hp.cond_latents + 2; // 32 + dur_half + dur = 34
    const int text_len = (int)text_tokens.size();
    const int prefix_len = cond_len + text_len + 2 + 1; // +start_text+stop_text+start_mel
    std::vector<float> embeds((size_t)prefix_len * D, 0.0f);
    int pos = 0;

    // 1. Conditioning block (assembled upstream from spk+emo, dur_half, dur).
    if ((int)cond_full.size() == cond_len * D) {
        std::memcpy(embeds.data(), cond_full.data(),
                    (size_t)cond_len * D * sizeof(float));
    }
    pos += cond_len;

    // 2. Text: start_text, text_tokens, stop_text + positional adds.
    const int max_tpi = (int)(text_pos_table.size() / D) - 1;
    auto add_text = [&](int tok, int tpi) {
        if (tok < 0 || tok >= hp.text_vocab) tok = 0;
        if (tpi < 0) tpi = 0;
        if (tpi > max_tpi) tpi = max_tpi;
        for (int j = 0; j < D; ++j)
            embeds[(size_t)pos * D + j] =
                text_emb_table[(size_t)tok * D + j] +
                text_pos_table[(size_t)tpi * D + j];
        pos++;
    };
    add_text(0, 0);                                              // start_text_token
    for (int i = 0; i < text_len; ++i) add_text(text_tokens[i], i + 1);
    add_text(1, text_len + 1);                                   // stop_text_token

    // 3. start_mel + mel_pos_embedding(0).
    int start_tok = hp.start_mel;
    if (start_tok >= hp.mel_codes) start_tok = 0;
    for (int j = 0; j < D; ++j)
        embeds[(size_t)pos * D + j] =
            mel_emb_table[(size_t)start_tok * D + j] + mel_pos_table[j];
    return embeds;
}

// Assemble the v2 conditioning block from raw pieces. Mirrors model_v2.py
// inference_speech / forward:
//     conds_latent = cat([spk_latents + emo_vec.unsqueeze(1),
//                         duration_emb_half.unsqueeze(1),
//                         duration_emb.unsqueeze(1)], dim=1)
// where duration_emb_half = speed_emb(1) and duration_emb = speed_emb(0).
static std::vector<float> assemble_cond_full(
    const HParams &hp,
    const std::vector<float> &spk_latents,       // [cond_num, D]
    const std::vector<float> &emo_vec,           // [D]
    const std::vector<float> &speed_emb_table) { // [2, D]
    const int D = hp.n_embd;
    const int n_spk = hp.cond_latents;
    const int cond_total = n_spk + 2;
    std::vector<float> cf((size_t)cond_total * D, 0.0f);
    const bool have_spk = (int)spk_latents.size() == n_spk * D;
    const bool have_emo = (int)emo_vec.size() == D;
    for (int r = 0; r < n_spk; ++r) {
        for (int j = 0; j < D; ++j) {
            float s = have_spk ? spk_latents[(size_t)r * D + j] : 0.0f;
            float e = have_emo ? emo_vec[j] : 0.0f;
            cf[(size_t)r * D + j] = s + e;
        }
    }
    if ((int)speed_emb_table.size() == 2 * D) {
        // PT: cat([..., speed_emb(1), speed_emb(0)], dim=1)
        std::memcpy(&cf[(size_t)n_spk * D],
                    &speed_emb_table[(size_t)1 * D],
                    (size_t)D * sizeof(float));
        std::memcpy(&cf[(size_t)(n_spk + 1) * D],
                    &speed_emb_table[(size_t)0 * D],
                    (size_t)D * sizeof(float));
    }
    return cf;
}

// Forward declaration (defined in the smoke section below).
static bool gpt_smoke_dump_f32(const std::string &path,
                               const std::vector<float> &data);
static inline void dump_f32(const std::string &path,
                            const std::vector<float> &data) {
    gpt_smoke_dump_f32(path, data);
}

// -----------------------------------------------------------------------------
// Conformer / Perceiver
//
// IndexTTS-2 conditioning_encoder receives semantic-codec hidden states (1024-d)
// and downsamples via Conv2dSubsampling(1024→512, stride=2) + Linear(511*512→512).
// Each encoder block follows ESPnet ordering (no macaron pre-FF):
//   MHSA (rel-pos) → Conv module → FF → Final LN
// The emo_conditioning_encoder has the same architecture (4 layers).
// -----------------------------------------------------------------------------
static ggml_tensor *try_t(const std::unordered_map<std::string, ggml_tensor *> &m,
                          const std::string &k) {
    auto it = m.find(k);
    return (it == m.end()) ? nullptr : it->second;
}

static ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x,
                           ggml_tensor *w, ggml_tensor *b) {
    if (!w) return x;
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

static ggml_tensor *layernorm(ggml_context *ctx, ggml_tensor *x,
                              ggml_tensor *gamma, ggml_tensor *beta,
                              float eps = 1e-5f) {
    ggml_tensor *y = ggml_norm(ctx, x, eps);
    if (gamma) y = ggml_mul(ctx, y, gamma);
    if (beta)  y = ggml_add(ctx, y, beta);
    return y;
}

// Build a single Conformer block graph over x [D, T].
// Implements the ESPnet / IndexTTS-2 block ordering (no macaron):
//   MHSA (with rel-pos biases) → conv_module → FF → final LN
// pos_enc [D, T]: absolute position table slice (passed separately for attention).
static ggml_tensor *conformer_block(ggml_context *ctx, ggml_tensor *x,
                                    const ConformerBlock &b, int n_heads,
                                    int head_dim, ggml_tensor *pos_enc) {
    const int T = (int)x->ne[1];
    const int D = (int)x->ne[0];

    // ---- 1. MHSA with relative position biases ----
    ggml_tensor *res = x;
    ggml_tensor *t = layernorm(ctx, x,
                               try_t(b.by_name, "norm_mha.weight"),
                               try_t(b.by_name, "norm_mha.bias"));
    ggml_tensor *Q = linear(ctx, t,
        try_t(b.by_name, "self_attn.linear_q.weight"),
        try_t(b.by_name, "self_attn.linear_q.bias"));
    ggml_tensor *K = linear(ctx, t,
        try_t(b.by_name, "self_attn.linear_k.weight"),
        try_t(b.by_name, "self_attn.linear_k.bias"));
    ggml_tensor *V = linear(ctx, t,
        try_t(b.by_name, "self_attn.linear_v.weight"),
        try_t(b.by_name, "self_attn.linear_v.bias"));

    // R = linear_pos(pos_enc) — project position table to [D, T].
    ggml_tensor *lp_w = try_t(b.by_name, "self_attn.linear_pos.weight");
    ggml_tensor *R    = lp_w ? ggml_mul_mat(ctx, lp_w, pos_enc) : pos_enc;

    // Untied biases: Q_u = Q + pos_bias_u, Q_v = Q + pos_bias_v.
    ggml_tensor *pu = try_t(b.by_name, "self_attn.pos_bias_u");
    ggml_tensor *pv = try_t(b.by_name, "self_attn.pos_bias_v");
    // pos_bias tensors may be stored as F16; cast to F32 for add.
    auto cast32 = [&](ggml_tensor *p) -> ggml_tensor * {
        if (!p) return nullptr;
        p = ggml_reshape_1d(ctx, p, D);
        return (p->type != GGML_TYPE_F32) ? ggml_cast(ctx, p, GGML_TYPE_F32) : p;
    };
    ggml_tensor *Q_u = pu ? ggml_add(ctx, Q, cast32(pu)) : Q;
    ggml_tensor *Q_v = pv ? ggml_add(ctx, Q, cast32(pv)) : Q;

    // Reshape to [head_dim, n_heads, T] then permute to [head_dim, T, n_heads].
    auto reshape_heads = [&](ggml_tensor *p) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, p, head_dim, n_heads, T), 0, 2, 1, 3));
    };
    Q_u = reshape_heads(Q_u);   // [head_dim, T, n_heads]
    Q_v = reshape_heads(Q_v);
    K   = reshape_heads(K);
    V   = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, V, head_dim, n_heads, T), 0, 2, 1, 3));
    R   = reshape_heads(R);     // [head_dim, T, n_heads]

    // BD = Q_v @ R^T — scaled position bias matrix fed to flash_attn_ext.
    // Relative shift is omitted (comment from CrispASR: "useless in speech").
    const float scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor *BD = ggml_mul_mat(ctx, ggml_cont(ctx, R), Q_v); // [T, T, n_heads]
    BD = ggml_cast(ctx, ggml_scale(ctx, ggml_cont(ctx, BD), scale), GGML_TYPE_F16);

    ggml_tensor *attn = ggml_flash_attn_ext(
        ctx, ggml_cont(ctx, Q_u), ggml_cont(ctx, K), V,
        BD, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx, attn, D, T);
    attn = linear(ctx, attn,
        try_t(b.by_name, "self_attn.linear_out.weight"),
        try_t(b.by_name, "self_attn.linear_out.bias"));
    x = ggml_add(ctx, res, attn);

    // ---- 2. Conv module ----
    // norm → pw1 → GLU(sigmoid) → depthwise(K=15,pad=7) → LN → SiLU → pw2 → residual
    res = x;
    t = layernorm(ctx, x,
                  try_t(b.by_name, "norm_conv.weight"),
                  try_t(b.by_name, "norm_conv.bias"));

    ggml_tensor *pw1_w = try_t(b.by_name, "conv_module.pointwise_conv1.weight");
    ggml_tensor *pw1_b = try_t(b.by_name, "conv_module.pointwise_conv1.bias");
    if (pw1_w) {
        // pw1_w is stored as [2D, D] (ggml: ne[0]=D, ne[1]=2D) matching mul_mat.
        // Reshape from any stored conv1d shape to 2D for matmul.
        ggml_tensor *w2d = ggml_reshape_2d(ctx, pw1_w, D, 2 * D);
        ggml_tensor *pw1_out = ggml_mul_mat(ctx, w2d, t); // [2D, T]
        if (pw1_b) pw1_out = ggml_add(ctx, pw1_out, pw1_b);

        // GLU: val = pw1_out[:D], gate = sigmoid(pw1_out[D:])
        ggml_tensor *val  = ggml_view_2d(ctx, pw1_out, D, T, pw1_out->nb[1], 0);
        ggml_tensor *gate = ggml_view_2d(ctx, pw1_out, D, T, pw1_out->nb[1],
                                         D * ggml_type_size(pw1_out->type));
        val  = ggml_cont(ctx, val);
        gate = ggml_cont(ctx, gate);
        t = ggml_mul(ctx, val, ggml_sigmoid(ctx, gate));

        // Depthwise conv1d (K=15, padding=7): weight [K, 1, D] in ggml.
        ggml_tensor *dw_w = try_t(b.by_name, "conv_module.depthwise_conv.weight");
        ggml_tensor *dw_b = try_t(b.by_name, "conv_module.depthwise_conv.bias");
        if (dw_w) {
            // conv_1d_dw expects input [T, C] (ne[0]=T, ne[1]=C).
            // t is [D, T] — transpose to [T, D] then apply dw conv.
            ggml_tensor *t_TC = ggml_cont(ctx, ggml_transpose(ctx, t)); // [T, D]
            // Cast kernel to F16 for ggml_conv_1d_dw (uses im2col F16 path).
            ggml_tensor *dw_f16 = ggml_cast(ctx, dw_w, GGML_TYPE_F16);
            ggml_tensor *t_dw = ggml_conv_1d_dw(ctx, dw_f16, t_TC, 1, 7, 1); // [T, D, 1]
            t_dw = ggml_reshape_2d(ctx, t_dw, T, D);   // ne=(T, D) — T fastest
            // Transpose back to [D, T] so the rest of the conv module
            // (LN, SiLU, pw2) keeps D as the fast axis.
            t = ggml_cont(ctx, ggml_transpose(ctx, t_dw));
            if (dw_b) {
                ggml_tensor *b2 = ggml_reshape_2d(ctx, dw_b, D, 1);
                t = ggml_add(ctx, t, b2);
            }
        }

        // conv_module.norm (LayerNorm in eval mode) + SiLU.
        t = layernorm(ctx, t,
                      try_t(b.by_name, "conv_module.norm.weight"),
                      try_t(b.by_name, "conv_module.norm.bias"));
        t = ggml_silu(ctx, t);

        // pw2: [D, D] pointwise
        ggml_tensor *pw2_w = try_t(b.by_name, "conv_module.pointwise_conv2.weight");
        ggml_tensor *pw2_b = try_t(b.by_name, "conv_module.pointwise_conv2.bias");
        if (pw2_w) {
            ggml_tensor *w2d2 = ggml_reshape_2d(ctx, pw2_w, D, D);
            t = ggml_mul_mat(ctx, w2d2, t);
            if (pw2_b) t = ggml_add(ctx, t, pw2_b);
        }
        x = ggml_add(ctx, res, t);
    }

    // ---- 3. Feed-Forward ----
    res = x;
    t = layernorm(ctx, x,
                  try_t(b.by_name, "norm_ff.weight"),
                  try_t(b.by_name, "norm_ff.bias"));
    t = linear(ctx, t, try_t(b.by_name, "feed_forward.w_1.weight"),
                       try_t(b.by_name, "feed_forward.w_1.bias"));
    t = ggml_silu(ctx, t);
    t = linear(ctx, t, try_t(b.by_name, "feed_forward.w_2.weight"),
                       try_t(b.by_name, "feed_forward.w_2.bias"));
    x = ggml_add(ctx, res, t);

    // ---- 4. Output norm ----
    x = layernorm(ctx, x,
                  try_t(b.by_name, "norm_final.weight"),
                  try_t(b.by_name, "norm_final.bias"));
    return x;
}

bool Model::RunConditioning(State &s, ggml_backend_sched_t sched,
                            const std::vector<float> &sc_hidden, int T_sc,
                            std::vector<float> &cond_latents) {
    (void)s;
    const int D_in   = hp_.sc_hidden;     // 1024
    const int D_cond = hp_.cond_dim;      // 512
    const int D_out  = hp_.n_embd;        // 1280
    const int N_lat  = hp_.cond_latents;  // 32
    if ((int)sc_hidden.size() != D_in * T_sc) {
        RS_LOG_ERR("[indextts2] RunConditioning: hidden size mismatch (%zu vs %d)",
                   sc_hidden.size(), D_in * T_sc);
        return false;
    }

    // After Conv2dSubsampling(stride=2, kernel=3) the time and feature dims change:
    //   T_enc = (T_sc - 3) / 2 + 1
    //   W_enc = (D_in - 3)  / 2 + 1 = (1024 - 3) / 2 + 1 = 511
    // The linear then maps 511 * 512 = 261632 → 512.
    const int T_enc = (T_sc - 3) / 2 + 1;
    const int W_enc = (D_in - 3) / 2 + 1; // 511

    // -------------------------------------------------------------------------
    // 1. Conv2dSubsampling + Linear embed.
    // -------------------------------------------------------------------------
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 32768 +
                              ggml_graph_overhead_custom(32768, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 32768, false);

    // Input [D_in=1024, T_sc] — ggml ne[0]=D_in (fastest).
    // ggml_conv_2d expects b in [W, H, C_in, N] layout (ne[0]=W fastest).
    // We treat D_in as W=1024 and T_sc as H, so the input tensor is
    // [ne[0]=D_in, ne[1]=T_sc, ne[2]=C_in=1, ne[3]=1].
    ggml_tensor *x_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_in, T_sc, 1, 1);
    ggml_set_name(x_in, "sc_in");
    ggml_set_input(x_in);

    // embed.conv.0: Conv2d [C_out=512, C_in=1, KH=3, KW=3], stride=(2,2), no pad.
    // ggml_conv_2d kernel shape: [KW, KH, C_in, C_out].
    ggml_tensor *cur = nullptr;
    ggml_tensor *conv_w = try_t(cond_enc_.by_name, "embed.conv.0.weight");
    ggml_tensor *conv_b = try_t(cond_enc_.by_name, "embed.conv.0.bias");
    if (conv_w) {
        cur = ggml_conv_2d(ctx, conv_w, x_in, 2, 2, 0, 0, 1, 1);
        // Output: [W_enc=511, T_enc, C_out=512, 1]. Add bias [C_out].
        if (conv_b) {
            ggml_tensor *bias4d = ggml_cast(ctx,
                ggml_reshape_4d(ctx, conv_b, 1, 1, D_cond, 1), GGML_TYPE_F32);
            cur = ggml_add(ctx, cur, bias4d);
        }
        cur = ggml_relu(ctx, cur);
        // Permute [W_enc, T_enc, C_out, 1] → [W_enc, C_out, T_enc, 1]
        //   permute(0,2,1,3): ne[0]=W_enc, ne[1]=C_out, ne[2]=T_enc stays as
        //   contiguous [W_enc*C_out, T_enc] after reshape.
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, cur, W_enc * D_cond, T_enc); // [261632, T_enc]
    } else {
        // Fallback: treat as passthrough (should not happen in a complete GGUF).
        cur = ggml_reshape_2d(ctx, x_in, D_in, T_sc);
    }

    // embed.out.0: Linear [512, 261632] → [512, T_enc].
    ggml_tensor *lin_w = try_t(cond_enc_.by_name, "embed.out.0.weight");
    ggml_tensor *lin_b = try_t(cond_enc_.by_name, "embed.out.0.bias");
    if (lin_w) {
        cur = ggml_mul_mat(ctx, lin_w, cur);
        if (lin_b) cur = ggml_add(ctx, cur, lin_b);
    }
    // Scale by sqrt(d_cond) — ESPnet RelPositionalEncoding convention.
    cur = ggml_scale(ctx, cur, std::sqrt((float)D_cond));

    // Build position encoding slice: pe_full is [D_cond, 5000, 1] in ggml
    // (after converter stores [1, 5000, D_cond] PyTorch tensor as [D, T, 1]).
    ggml_tensor *pe_full = try_t(cond_enc_.by_name, "embed.pos_enc.pe");
    ggml_tensor *pos_enc = nullptr;
    if (pe_full && T_enc > 0) {
        // View the first T_enc columns; pe_full ne[0]=D_cond, ne[1]=5000.
        pos_enc = ggml_cast(ctx,
            ggml_view_2d(ctx, pe_full, D_cond, T_enc, pe_full->nb[1], 0),
            GGML_TYPE_F32);
    } else {
        // Zero position encoding fallback.
        pos_enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_cond, T_enc);
        ggml_set_zero(pos_enc);
    }

    // Conformer blocks.
    const int head_dim = D_cond / hp_.cond_heads;
    for (int il = 0; il < (int)cond_enc_.blocks.size(); ++il)
        cur = conformer_block(ctx, cur, cond_enc_.blocks[il],
                              hp_.cond_heads, head_dim, pos_enc);

    // Post-encoder norm.
    cur = layernorm(ctx, cur,
                    try_t(cond_enc_.by_name, "after_norm.weight"),
                    try_t(cond_enc_.by_name, "after_norm.bias"));
    ggml_set_name(cur, "conformer_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] conformer graph alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, sc_hidden.data(), 0,
                            sc_hidden.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] conformer compute failed");
        ggml_free(ctx);
        return false;
    }
    std::vector<float> conf_out((size_t)D_cond * T_enc);
    ggml_backend_tensor_get(cur, conf_out.data(), 0,
                            conf_out.size() * sizeof(float));
    ggml_free(ctx);

    // -------------------------------------------------------------------------
    // 2. Perceiver resampler graph.
    // -------------------------------------------------------------------------
    ggml_init_params ip2 = { meta.size(), meta.data(), true };
    ctx = ggml_init(ip2);
    gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_tensor *ctx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_cond, T_enc);
    ggml_set_name(ctx_in, "perc_ctx");
    ggml_set_input(ctx_in);

    // proj_context (D_cond → D_out=1280).
    ggml_tensor *kv_ctx = linear(ctx, ctx_in, cond_perceiver_.proj_ctx_w,
                                 cond_perceiver_.proj_ctx_b);
    if (!kv_ctx) kv_ctx = ctx_in;

    const bool perc_debug = std::getenv("RS_INDEXTTS2_PERC_DEBUG") != nullptr;
    std::vector<ggml_tensor *> dbg_taps;
    auto tap = [&](ggml_tensor *t, const char *nm) {
        if (perc_debug) {
            ggml_set_name(t, nm);
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
            dbg_taps.push_back(t);
        }
    };
    tap(kv_ctx, "perc_kv_ctx");

    // Promote latents to F32 immediately — stored F16 in GGUF but every
    // downstream op (mul_mat output, GEGLU, final RMSNorm) is F32, and
    // ggml_add with mismatched dtypes silently leaves the F16 path on the LHS
    // unchanged, costing ~12 dB SNR by the end of depth=2.
    ggml_tensor *latents = ggml_dup(ctx, cond_perceiver_.latents);
    if (latents->type != GGML_TYPE_F32)
        latents = ggml_cast(ctx, latents, GGML_TYPE_F32);
    const int n_h = 8;
    const int hd  = D_out / n_h;

    int dbg_li = 0;
    for (auto &L : cond_perceiver_.layers) {
        ggml_tensor *q_w  = try_t(L.by_name, "0.to_q.weight");
        ggml_tensor *kv_w = try_t(L.by_name, "0.to_kv.weight");
        ggml_tensor *o_w  = try_t(L.by_name, "0.to_out.weight");
        if (q_w && kv_w && o_w) {
            // cross_attn_include_queries=True → KV input is concat(latents, kv_ctx)
            // along the sequence axis, so KV length becomes N_lat + T_enc.
            // latents: [D_out, N_lat], kv_ctx: [D_out, T_enc] → concat on ne[1].
            ggml_tensor *lat_for_cat = (latents->type == kv_ctx->type) ? latents
                : ggml_cast(ctx, latents, kv_ctx->type);
            ggml_tensor *kv_in = ggml_concat(ctx, lat_for_cat, kv_ctx, /*dim*/ 1);
            const int kv_len = N_lat + T_enc;

            ggml_tensor *q  = ggml_mul_mat(ctx, q_w, latents);
            ggml_mul_mat_set_prec(q, GGML_PREC_F32);
            ggml_tensor *kv = ggml_mul_mat(ctx, kv_w, kv_in);
            ggml_mul_mat_set_prec(kv, GGML_PREC_F32);
            const int dim_inner = (int)q->ne[0];
            const size_t ts = ggml_type_size(kv->type);
            ggml_tensor *k = ggml_view_2d(ctx, kv, dim_inner, kv_len,
                                          kv->nb[1], 0);
            ggml_tensor *v = ggml_view_2d(ctx, kv, dim_inner, kv_len,
                                          kv->nb[1], dim_inner * ts);
            k = ggml_cont(ctx, k);
            v = ggml_cont(ctx, v);
            const int hd_inner = dim_inner / n_h;
            q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, q, hd_inner, n_h, N_lat), 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, k, hd_inner, n_h, kv_len), 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, v, hd_inner, n_h, kv_len), 0, 2, 1, 3));
            ggml_tensor *a = ggml_flash_attn_ext(
                ctx, q, k, v, nullptr,
                1.0f / std::sqrt((float)hd_inner), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
            a = ggml_reshape_2d(ctx, a, dim_inner, N_lat);
            a = ggml_mul_mat(ctx, o_w, a);
            ggml_mul_mat_set_prec(a, GGML_PREC_F32);
            if (perc_debug) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "perc_l%d_attn_raw", dbg_li);
                tap(a, nm);
            }
            latents = ggml_add(ctx, latents, a);
            if (perc_debug) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "perc_l%d_attn_res", dbg_li);
                tap(latents, nm);
            }
        }
        // GEGLU FFN. PT uses Sequential(*filter(exists, ...)) so when conv=None
        // the second Linear is at index .1.2 (not .1.3); each Linear has a bias.
        ggml_tensor *ff_in   = try_t(L.by_name, "1.0.weight");
        ggml_tensor *ff_in_b = try_t(L.by_name, "1.0.bias");
        ggml_tensor *ff_out  = try_t(L.by_name, "1.2.weight");
        ggml_tensor *ff_out_b = try_t(L.by_name, "1.2.bias");
        if (ff_in && ff_out) {
            ggml_tensor *h = ggml_mul_mat(ctx, ff_in, latents);
            ggml_mul_mat_set_prec(h, GGML_PREC_F32);
            if (ff_in_b) h = ggml_add(ctx, h, ff_in_b);
            const int two_di = (int)h->ne[0];
            const int di = two_di / 2;
            const size_t ts = ggml_type_size(h->type);
            // PT: x, gate = chunk(2, dim=-1) → x=first_half, gate=second_half.
            // GEGLU returns gelu(gate) * x.
            ggml_tensor *val  = ggml_cont(ctx,
                ggml_view_2d(ctx, h, di, h->ne[1], h->nb[1], 0));
            ggml_tensor *gate = ggml_cont(ctx,
                ggml_view_2d(ctx, h, di, h->ne[1], h->nb[1], di * ts));
            h = ggml_mul(ctx, val, ggml_gelu_erf(ctx, gate));
            h = ggml_mul_mat(ctx, ff_out, h);
            ggml_mul_mat_set_prec(h, GGML_PREC_F32);
            if (ff_out_b) h = ggml_add(ctx, h, ff_out_b);
            if (perc_debug) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "perc_l%d_ff_raw", dbg_li);
                tap(h, nm);
            }
            latents = ggml_add(ctx, latents, h);
            if (perc_debug) {
                char nm[64];
                std::snprintf(nm, sizeof(nm), "perc_l%d_ff_res", dbg_li);
                tap(latents, nm);
            }
        }
        dbg_li++;
    }
    if (cond_perceiver_.norm_gamma) {
        // rms_norm requires F32; cast if flash_attn_ext produced F16 accumulation.
        if (latents->type != GGML_TYPE_F32)
            latents = ggml_cast(ctx, latents, GGML_TYPE_F32);
        latents = ggml_rms_norm(ctx, latents, 1e-8f);
        latents = ggml_mul(ctx, latents, cond_perceiver_.norm_gamma);
    }
    ggml_set_name(latents, "perc_out");
    ggml_set_output(latents);
    ggml_build_forward_expand(gf, latents);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] perceiver alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(ctx_in, conf_out.data(), 0,
                            conf_out.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] perceiver compute failed");
        ggml_free(ctx);
        return false;
    }
    cond_latents.assign((size_t)N_lat * D_out, 0.0f);
    ggml_backend_tensor_get(latents, cond_latents.data(), 0,
                            cond_latents.size() * sizeof(float));

    // Dump per-layer taps (if RS_INDEXTTS2_PERC_DEBUG set) before freeing ctx.
    if (perc_debug) {
        const char *dir_c = std::getenv("RS_INDEXTTS2_CONFORMER_TEST_DIR");
        if (dir_c) {
            std::string dir = dir_c;
            if (!dir.empty() && dir.back() != '/') dir.push_back('/');
            for (ggml_tensor *t : dbg_taps) {
                size_t n = (size_t)ggml_nelements(t);
                std::vector<float> buf(n);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
                dump_f32(dir + std::string(ggml_get_name(t)) + ".f32", buf);
                RS_LOG_INFO("[perc-debug] dumped %s nelem=%zu ne=(%lld,%lld,%lld,%lld)",
                            ggml_get_name(t), n,
                            (long long)t->ne[0], (long long)t->ne[1],
                            (long long)t->ne[2], (long long)t->ne[3]);
            }
        }
    }
    ggml_free(ctx);

    // Optional: dump intermediate tensors for diff vs PT reference.
    // Triggered when RS_INDEXTTS2_CONFORMER_TEST_DIR is set.
    if (const char *dir_c = std::getenv("RS_INDEXTTS2_CONFORMER_TEST_DIR")) {
        std::string dir = dir_c;
        if (!dir.empty() && dir.back() != '/') dir.push_back('/');
        // conf_out has ggml shape ne=(D_cond, T_enc) → D_cond is fastest, so
        // the memory linear order is identical to NumPy [T_enc, D_cond] C-fast.
        // No transpose needed for the diff dump.
        dump_f32(dir + "conformer_out.f32", conf_out);
        // cond_latents is [N_lat * D_out] = [32, 1280] row-major. PT saves same.
        dump_f32(dir + "cond_latents.f32", cond_latents);
        RS_LOG_INFO("[conformer-smoke] dumped conformer_out [%d,%d] and "
                    "cond_latents [%d,%d] to %s",
                    T_enc, D_cond, N_lat, D_out, dir.c_str());
    }
    return true;
}

bool Model::RunGetEmovec(State &s, ggml_backend_sched_t sched,
                         const std::vector<float> &emo_hidden, int T_emo,
                         std::vector<float> &emo_latent) {
    (void)s;
    const int D_in   = hp_.sc_hidden;  // 1024 (same W2V-BERT hidden)
    const int D_cond = hp_.cond_dim;   // 512
    const int D_out  = hp_.n_embd;     // 1280 (after emovec_layer + emo_layer)
    const int N_lat  = 1;              // emo perceiver has 1 latent → [1, 1024]

    if ((int)emo_hidden.size() != D_in * T_emo) {
        RS_LOG_ERR("[indextts2] RunGetEmovec: hidden size mismatch (%zu vs %d)",
                   emo_hidden.size(), D_in * T_emo);
        emo_latent.assign((size_t)D_out, 0.0f);
        return true; // non-fatal: zero emo still allows AR to run
    }

    const int T_enc = (T_emo - 3) / 2 + 1;
    const int W_enc = (D_in - 3)  / 2 + 1; // 511

    std::vector<uint8_t> meta(ggml_tensor_overhead() * 32768 +
                              ggml_graph_overhead_custom(32768, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 32768, false);

    ggml_tensor *x_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_in, T_emo, 1, 1);
    ggml_set_name(x_in, "emo_in");
    ggml_set_input(x_in);

    ggml_tensor *cur = nullptr;
    ggml_tensor *conv_w = try_t(emo_cond_enc_.by_name, "embed.conv.0.weight");
    ggml_tensor *conv_b = try_t(emo_cond_enc_.by_name, "embed.conv.0.bias");
    if (conv_w) {
        cur = ggml_conv_2d(ctx, conv_w, x_in, 2, 2, 0, 0, 1, 1);
        if (conv_b) {
            ggml_tensor *bias4d = ggml_cast(ctx,
                ggml_reshape_4d(ctx, conv_b, 1, 1, D_cond, 1), GGML_TYPE_F32);
            cur = ggml_add(ctx, cur, bias4d);
        }
        cur = ggml_relu(ctx, cur);
        cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx, cur, W_enc * D_cond, T_enc);
    } else {
        cur = ggml_reshape_2d(ctx, x_in, D_in, T_emo);
    }
    ggml_tensor *lin_w = try_t(emo_cond_enc_.by_name, "embed.out.0.weight");
    ggml_tensor *lin_b = try_t(emo_cond_enc_.by_name, "embed.out.0.bias");
    if (lin_w) {
        cur = ggml_mul_mat(ctx, lin_w, cur);
        if (lin_b) cur = ggml_add(ctx, cur, lin_b);
    }
    cur = ggml_scale(ctx, cur, std::sqrt((float)D_cond));

    ggml_tensor *pe_full = try_t(emo_cond_enc_.by_name, "embed.pos_enc.pe");
    ggml_tensor *pos_enc = nullptr;
    if (pe_full && T_enc > 0) {
        pos_enc = ggml_cast(ctx,
            ggml_view_2d(ctx, pe_full, D_cond, T_enc, pe_full->nb[1], 0),
            GGML_TYPE_F32);
    } else {
        pos_enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_cond, T_enc);
        ggml_set_zero(pos_enc);
    }

    const int head_dim = D_cond / hp_.emo_heads;
    for (int il = 0; il < (int)emo_cond_enc_.blocks.size(); ++il)
        cur = conformer_block(ctx, cur, emo_cond_enc_.blocks[il],
                              hp_.emo_heads, head_dim, pos_enc);

    cur = layernorm(ctx, cur,
                    try_t(emo_cond_enc_.by_name, "after_norm.weight"),
                    try_t(emo_cond_enc_.by_name, "after_norm.bias"));
    ggml_set_name(cur, "emo_conformer_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] emo conformer alloc failed");
        ggml_free(ctx); emo_latent.assign(D_out, 0.0f); return true;
    }
    ggml_backend_tensor_set(x_in, emo_hidden.data(), 0,
                            emo_hidden.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] emo conformer compute failed");
        ggml_free(ctx); emo_latent.assign(D_out, 0.0f); return true;
    }
    std::vector<float> emo_conf((size_t)D_cond * T_enc);
    ggml_backend_tensor_get(cur, emo_conf.data(), 0,
                            emo_conf.size() * sizeof(float));
    ggml_free(ctx);

    // Emo Perceiver: 1 latent, D_perceiver=1024, then emovec_layer 1024→1280.
    const int D_perc = (int)emo_cond_perceiver_.latents->ne[0]; // 1024
    ggml_init_params ip2 = { meta.size(), meta.data(), true };
    ctx = ggml_init(ip2);
    gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_tensor *ctx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_cond, T_enc);
    ggml_set_name(ctx_in, "emo_perc_ctx");
    ggml_set_input(ctx_in);

    ggml_tensor *kv_ctx = linear(ctx, ctx_in,
        emo_cond_perceiver_.proj_ctx_w, emo_cond_perceiver_.proj_ctx_b);
    if (!kv_ctx) kv_ctx = ctx_in;

    ggml_tensor *latents = ggml_dup(ctx, emo_cond_perceiver_.latents); // [D_perc, 1]
    // Promote to F32 (see comment in cond perceiver: ggml_add dtype mismatch).
    if (latents->type != GGML_TYPE_F32)
        latents = ggml_cast(ctx, latents, GGML_TYPE_F32);
    // PT emo Perceiver uses heads=emo_condition_module.attention_heads=4.
    const int n_h2 = hp_.emo_heads; // 4

    for (auto &L : emo_cond_perceiver_.layers) {
        ggml_tensor *q_w  = try_t(L.by_name, "0.to_q.weight");
        ggml_tensor *kv_w = try_t(L.by_name, "0.to_kv.weight");
        ggml_tensor *o_w  = try_t(L.by_name, "0.to_out.weight");
        if (q_w && kv_w && o_w) {
            // cross_attn_include_queries=True → KV = concat(latents, kv_ctx).
            ggml_tensor *lat_for_cat = (latents->type == kv_ctx->type) ? latents
                : ggml_cast(ctx, latents, kv_ctx->type);
            ggml_tensor *kv_in = ggml_concat(ctx, lat_for_cat, kv_ctx, /*dim*/ 1);
            const int kv_len = N_lat + T_enc;
            ggml_tensor *q  = ggml_mul_mat(ctx, q_w, latents);
            ggml_mul_mat_set_prec(q, GGML_PREC_F32);
            ggml_tensor *kv = ggml_mul_mat(ctx, kv_w, kv_in);
            ggml_mul_mat_set_prec(kv, GGML_PREC_F32);
            const int dim_inner = (int)q->ne[0];
            const size_t ts = ggml_type_size(kv->type);
            ggml_tensor *k = ggml_cont(ctx,
                ggml_view_2d(ctx, kv, dim_inner, kv_len, kv->nb[1], 0));
            ggml_tensor *v = ggml_cont(ctx,
                ggml_view_2d(ctx, kv, dim_inner, kv_len, kv->nb[1], dim_inner * ts));
            const int hd2 = dim_inner / n_h2;
            q = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, q, hd2, n_h2, N_lat), 0, 2, 1, 3));
            k = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, k, hd2, n_h2, kv_len), 0, 2, 1, 3));
            v = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, v, hd2, n_h2, kv_len), 0, 2, 1, 3));
            ggml_tensor *a = ggml_flash_attn_ext(
                ctx, q, k, v, nullptr,
                1.0f / std::sqrt((float)hd2), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
            a = ggml_reshape_2d(ctx, a, dim_inner, N_lat);
            a = ggml_mul_mat(ctx, o_w, a);
            latents = ggml_add(ctx, latents, a);
        }
        ggml_tensor *ff_in   = try_t(L.by_name, "1.0.weight");
        ggml_tensor *ff_in_b = try_t(L.by_name, "1.0.bias");
        ggml_tensor *ff_out  = try_t(L.by_name, "1.2.weight");
        ggml_tensor *ff_out_b = try_t(L.by_name, "1.2.bias");
        if (ff_in && ff_out) {
            ggml_tensor *h = ggml_mul_mat(ctx, ff_in, latents);
            if (ff_in_b) h = ggml_add(ctx, h, ff_in_b);
            const int two_di = (int)h->ne[0];
            const int di = two_di / 2;
            const size_t ts = ggml_type_size(h->type);
            ggml_tensor *val  = ggml_cont(ctx,
                ggml_view_2d(ctx, h, di, h->ne[1], h->nb[1], 0));
            ggml_tensor *gate = ggml_cont(ctx,
                ggml_view_2d(ctx, h, di, h->ne[1], h->nb[1], di * ts));
            h = ggml_mul(ctx, val, ggml_gelu_erf(ctx, gate));
            h = ggml_mul_mat(ctx, ff_out, h);
            if (ff_out_b) h = ggml_add(ctx, h, ff_out_b);
            latents = ggml_add(ctx, latents, h);
        }
    }
    if (emo_cond_perceiver_.norm_gamma) {
        if (latents->type != GGML_TYPE_F32)
            latents = ggml_cast(ctx, latents, GGML_TYPE_F32);
        latents = ggml_rms_norm(ctx, latents, 1e-8f);
        latents = ggml_mul(ctx, latents, emo_cond_perceiver_.norm_gamma);
    }
    // Tap pre-emovec latents → "emo_latents_raw" (shape [D_perc=1024]).
    ggml_tensor *emo_pre = latents;
    ggml_set_name(emo_pre, "emo_latents_raw");
    ggml_set_output(emo_pre);
    ggml_build_forward_expand(gf, emo_pre);
    // emovec_layer: linear [D_perc=1024] → [D_out=1280]
    if (emovec_layer_w_) {
        latents = ggml_mul_mat(ctx, emovec_layer_w_, latents);
        if (emovec_layer_b_) latents = ggml_add(ctx, latents, emovec_layer_b_);
    }
    // Tap post-emovec_layer (pre-emo_layer) → "emo_emovec_layer" [D_out=1280].
    ggml_tensor *emo_emovec = latents;
    ggml_set_name(emo_emovec, "emo_emovec_layer");
    ggml_set_output(emo_emovec);
    ggml_build_forward_expand(gf, emo_emovec);
    // emo_layer: linear [D_out=1280] → [D_out=1280] (completes gpt.get_emovec).
    if (emo_layer_w_) {
        latents = ggml_mul_mat(ctx, emo_layer_w_, latents);
        if (emo_layer_b_) latents = ggml_add(ctx, latents, emo_layer_b_);
    }
    ggml_set_name(latents, "emo_out");
    ggml_set_output(latents);
    ggml_build_forward_expand(gf, latents);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] emo perceiver alloc failed");
        ggml_free(ctx); emo_latent.assign(D_out, 0.0f); return true;
    }
    ggml_backend_tensor_set(ctx_in, emo_conf.data(), 0,
                            emo_conf.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] emo perceiver compute failed");
        ggml_free(ctx); emo_latent.assign(D_out, 0.0f); return true;
    }
    // latents shape after emo_layer: [D_out, N_lat=1] — full get_emovec output.
    emo_latent.assign((size_t)D_out, 0.0f);
    ggml_backend_tensor_get(latents, emo_latent.data(), 0,
                            emo_latent.size() * sizeof(float));
    std::vector<float> emo_latents_raw_buf((size_t)D_perc);
    ggml_backend_tensor_get(emo_pre, emo_latents_raw_buf.data(), 0,
                            emo_latents_raw_buf.size() * sizeof(float));
    std::vector<float> emo_emovec_buf((size_t)D_out);
    ggml_backend_tensor_get(emo_emovec, emo_emovec_buf.data(), 0,
                            emo_emovec_buf.size() * sizeof(float));
    ggml_free(ctx);

    // Optional smoke dumps (env var RS_INDEXTTS2_CONFORMER_TEST_DIR).
    if (const char *dir_c = std::getenv("RS_INDEXTTS2_CONFORMER_TEST_DIR")) {
        std::string dir = dir_c;
        if (!dir.empty() && dir.back() != '/') dir.push_back('/');
        dump_f32(dir + "emo_conformer_out.f32",  emo_conf);
        dump_f32(dir + "emo_latents_raw.f32",    emo_latents_raw_buf);
        dump_f32(dir + "emo_emovec_layer.f32",   emo_emovec_buf);
        dump_f32(dir + "emo_vec.f32",            emo_latent);
        RS_LOG_INFO("[emo-smoke] dumped emo_conformer_out [%d,%d], "
                    "emo_latents_raw [%d], emo_emovec_layer [%d], emo_vec [%d] to %s",
                    T_enc, D_cond, D_perc, D_out, D_out, dir.c_str());
    }
    return true;
}

// -----------------------------------------------------------------------------
// AR generation loop + second pass for lm_latent
// -----------------------------------------------------------------------------
bool Model::RunAR(State &s, ggml_backend_sched_t sched) {
    if (s.text_token_ids.empty()) {
        RS_LOG_ERR("[indextts2] RunAR: no text tokens (BPE not wired?)");
        return false;
    }
    if (!text_emb_w_ || !mel_emb_w_ || !text_pos_emb_w_ || !mel_pos_emb_w_) {
        RS_LOG_WARN("[indextts2] RunAR: GPT embedding tables not bound — "
                    "skipping AR loop (gpt.pth not yet converted). "
                    "mel_codes will be empty.");
        s.mel_codes.clear();
        s.lm_latent.clear();
        return true;
    }

    const int D = hp_.n_embd;

    // Allocate / reuse KV cache. Size to (max_text + max_mel + cond_latents)
    // tokens, which bounds any single AR loop.
    GPTContext gc;
    gc.kv_max_ctx = hp_.cond_latents + hp_.max_text_tok + hp_.max_mel_tok + 16;
    gc.compute_meta.resize(ggml_tensor_overhead() * 16384 +
                           ggml_graph_overhead_custom(16384, false));
    ggml_init_params kvp = { ggml_tensor_overhead() * 8, nullptr, true };
    gc.ctx_kv = ggml_init(kvp);
    gc.kv_k = ggml_new_tensor_4d(gc.ctx_kv, GGML_TYPE_F16,
                                 hp_.head_dim, gc.kv_max_ctx,
                                 hp_.n_head, hp_.n_layer);
    gc.kv_v = ggml_new_tensor_4d(gc.ctx_kv, GGML_TYPE_F16,
                                 hp_.head_dim, gc.kv_max_ctx,
                                 hp_.n_head, hp_.n_layer);
    gc.buf_kv = ggml_backend_alloc_ctx_tensors(gc.ctx_kv, backend_);
    if (!gc.buf_kv) {
        RS_LOG_ERR("[indextts2] KV cache alloc failed");
        ggml_free(gc.ctx_kv);
        return false;
    }

    // Read embedding tables on the host (small, ~150MB total — acceptable).
    std::vector<float> text_emb_table((size_t)hp_.text_vocab * D);
    tensor_get_f32(text_emb_w_, text_emb_table.data(), text_emb_table.size());
    std::vector<float> text_pos_table(
        (size_t)(hp_.max_text_tok + 2) * D);
    tensor_get_f32(text_pos_emb_w_, text_pos_table.data(),
                   text_pos_table.size());
    std::vector<float> mel_emb_table((size_t)hp_.mel_codes * D);
    tensor_get_f32(mel_emb_w_, mel_emb_table.data(), mel_emb_table.size());
    std::vector<float> mel_pos_table(
        (size_t)(hp_.max_mel_tok + 3) * D);
    tensor_get_f32(mel_pos_emb_w_, mel_pos_table.data(),
                   mel_pos_table.size());

    // Cond latents: when sc_hidden has been populated upstream (PushReferenceAudio
    // or env override RS_INDEXTTS2_SC_HIDDEN_NPY → Decode), run the validated
    // Conformer+Perceiver path to fill spk_latents and emo_vec. Otherwise fall
    // back to zeros — the AR loop still produces tokens, voice-agnostic.
    // Layout (v2): [cond_num spk+emo rows, dur_half row, dur row].
    const int cond_total = hp_.cond_latents + 2;
    std::vector<float> spk_latents_z((size_t)hp_.cond_latents * D, 0.0f);
    std::vector<float> emo_vec_z((size_t)D, 0.0f);
    const int D_in = hp_.sc_hidden;
    // The Conformer/Perceiver conditioning consumes spk_cond_emb (normalized
    // W2V-BERT hidden[17]) — NOT the semantic-codec S_ref (that feeds S2Mel).
    // Fall back to sc_hidden only for the RS_INDEXTTS2_SC_HIDDEN_NPY bring-up
    // override, which populates sc_hidden directly.
    const std::vector<float> &cond_feat =
        !s.spk_cond_emb.empty() ? s.spk_cond_emb : s.sc_hidden;
    const auto t_cond0 = std::chrono::steady_clock::now();
    if (!cond_feat.empty() && s.T_sc > 0 &&
        (int)cond_feat.size() == s.T_sc * D_in) {
        if (!RunConditioning(s, sched, cond_feat, s.T_sc, spk_latents_z)) {
            RS_LOG_WARN("[indextts2] RunConditioning failed — using zeros");
            spk_latents_z.assign((size_t)hp_.cond_latents * D, 0.0f);
        }
        const auto t_cond1 = std::chrono::steady_clock::now();
        RS_LOG_INFO("[timing] RunConditioning %.2fs",
                    std::chrono::duration<double>(t_cond1 - t_cond0).count());
        // Emotion: full 4-mode resolution (base/alpha blend + emo_matrix vector
        // path + QwenEmotion). Mirrors infer_v2.py infer_generator.
        if (!ResolveEmotion(s, sched, emo_vec_z)) {
            RS_LOG_WARN("[indextts2] ResolveEmotion failed — using zeros");
            emo_vec_z.assign((size_t)D, 0.0f);
        }
    } else {
        RS_LOG_INFO("[indextts2] RunAR: sc_hidden not set — voice-agnostic AR "
                    "(set RS_INDEXTTS2_SC_HIDDEN_NPY or wait for W2V-BERT)");
    }
    std::vector<float> speed_emb_table;
    if (speed_emb_w_) {
        speed_emb_table.assign((size_t)2 * D, 0.0f);
        tensor_get_f32(speed_emb_w_, speed_emb_table.data(),
                       speed_emb_table.size());
    }
    std::vector<float> cond_full = assemble_cond_full(
        hp_, spk_latents_z, emo_vec_z, speed_emb_table);
    (void)cond_total;

    std::vector<float> prefix = build_prefix_embeds(
        hp_, text_emb_table, text_pos_table, mel_emb_table, mel_pos_table,
        cond_full, s.text_token_ids);
    const int prefix_len = (int)prefix.size() / D;
    if (prefix_len > gc.kv_max_ctx) {
        RS_LOG_ERR("[indextts2] prefix (%d) exceeds KV (%d)", prefix_len,
                   gc.kv_max_ctx);
        ggml_backend_buffer_free(gc.buf_kv);
        ggml_free(gc.ctx_kv);
        return false;
    }

    // ------ Prefill -----------------------------------------------------
    {
        ggml_cgraph *gf = build_gpt_step_graph(
            *this, hp_, blocks_, gpt_ln_f_w_, gpt_ln_f_b_,
            final_norm_w_, final_norm_b_, mel_head_w_, mel_head_b_,
            gpt_wpe_w_, gc, 0, prefix_len);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            RS_LOG_ERR("[indextts2] AR prefill alloc failed");
            ggml_backend_buffer_free(gc.buf_kv);
            ggml_free(gc.ctx_kv);
            return false;
        }
        std::vector<ggml_fp16_t> mask(
            (size_t)prefix_len * prefix_len,
            ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < prefix_len; ++q)
            for (int k = q + 1; k < prefix_len; ++k)
                mask[(size_t)q * prefix_len + k] = neg_inf;
        ggml_backend_tensor_set(
            ggml_graph_get_tensor(gf, "inputs_embeds"), prefix.data(), 0,
            prefix.size() * sizeof(float));
        set_position_ids(gf, 0, prefix_len);
        ggml_backend_tensor_set(
            ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
            mask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[indextts2] AR prefill compute failed");
            ggml_backend_buffer_free(gc.buf_kv);
            ggml_free(gc.ctx_kv);
            return false;
        }
        // First logits → first sampled token.
        std::vector<float> logits(hp_.mel_codes);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"),
                                logits.data(), 0,
                                logits.size() * sizeof(float));
        gc.n_past = prefix_len;
        int t = sample_top_kp(s.rng, logits, s.temperature, s.top_k, s.top_p,
                              /*rep=*/10.0f, s.mel_codes, hp_.stop_mel);
        if (t != hp_.stop_mel) s.mel_codes.push_back(t);
    }

    // ------ Loop --------------------------------------------------------
    const auto t_loop0 = std::chrono::steady_clock::now();
    while ((int)s.mel_codes.size() < hp_.max_mel_tok &&
           (s.mel_codes.empty() || s.mel_codes.back() != hp_.stop_mel)) {
        const int mel_pos_idx = (int)s.mel_codes.size(); // already incl start
        const int tok = s.mel_codes.back();
        if (tok < 0 || tok >= hp_.mel_codes) break;
        std::vector<float> embed(D);
        const int pid = std::min(mel_pos_idx, hp_.max_mel_tok + 2);
        for (int j = 0; j < D; ++j)
            embed[j] = mel_emb_table[(size_t)tok * D + j] +
                       mel_pos_table[(size_t)pid * D + j];

        ggml_cgraph *gf = build_gpt_step_graph(
            *this, hp_, blocks_, gpt_ln_f_w_, gpt_ln_f_b_,
            final_norm_w_, final_norm_b_, mel_head_w_, mel_head_b_,
            gpt_wpe_w_, gc, gc.n_past, 1);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            RS_LOG_ERR("[indextts2] AR step alloc failed");
            break;
        }
        ggml_backend_tensor_set(
            ggml_graph_get_tensor(gf, "inputs_embeds"), embed.data(), 0,
            embed.size() * sizeof(float));
        set_position_ids(gf, gc.n_past, 1);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[indextts2] AR step compute failed at pos %d", mel_pos_idx);
            break;
        }
        gc.n_past += 1;
        std::vector<float> logits(hp_.mel_codes);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"),
                                logits.data(), 0,
                                logits.size() * sizeof(float));
        int nt = sample_top_kp(s.rng, logits, s.temperature, s.top_k, s.top_p,
                               10.0f, s.mel_codes, hp_.stop_mel);
        s.mel_codes.push_back(nt);
        if (nt == hp_.stop_mel) break;
    }
    if (!s.mel_codes.empty() && s.mel_codes.back() == hp_.stop_mel)
        s.mel_codes.pop_back();
    RS_LOG_INFO("[indextts2] AR produced %d mel codes", (int)s.mel_codes.size());
    RS_LOG_INFO("[timing] AR sampling loop %.2fs (%d tokens)",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop0).count(),
                (int)s.mel_codes.size());

    // ------ Second pass: extract lm_latent ------------------------------
    // Re-run GPT-2 over [prefix + all sampled mel codes] with a fresh KV
    // cache, read post-final_norm hidden states at the mel positions.
    // This mirrors infer_v2.py: gpt(speech_cond_latent, text_tokens, ...,
    //   codes, ...) → hidden[:, mel_range, :] → lm_latent [T_mel, D].
    const int T_mel = (int)s.mel_codes.size();
    if (T_mel > 0) {
        // Build full sequence: prefix + mel embeddings for each sampled code.
        std::vector<float> full_seq = prefix; // [prefix_len, D]
        full_seq.reserve(prefix.size() + (size_t)T_mel * D);
        for (int i = 0; i < T_mel; ++i) {
            int tok = s.mel_codes[i];
            if (tok < 0 || tok >= hp_.mel_codes) tok = 0;
            const int pid = std::min(i + 1, hp_.max_mel_tok + 2);
            for (int j = 0; j < D; ++j)
                full_seq.push_back(mel_emb_table[(size_t)tok * D + j] +
                                   mel_pos_table[(size_t)pid * D + j]);
        }
        const int full_len = prefix_len + T_mel;

        // Fresh KV cache for second pass.
        GPTContext gc2;
        gc2.kv_max_ctx = full_len + 4;
        gc2.compute_meta = gc.compute_meta;
        ggml_init_params kvp2 = { ggml_tensor_overhead() * 8, nullptr, true };
        gc2.ctx_kv = ggml_init(kvp2);
        gc2.kv_k = ggml_new_tensor_4d(gc2.ctx_kv, GGML_TYPE_F16,
                                      hp_.head_dim, gc2.kv_max_ctx,
                                      hp_.n_head, hp_.n_layer);
        gc2.kv_v = ggml_new_tensor_4d(gc2.ctx_kv, GGML_TYPE_F16,
                                      hp_.head_dim, gc2.kv_max_ctx,
                                      hp_.n_head, hp_.n_layer);
        gc2.buf_kv = ggml_backend_alloc_ctx_tensors(gc2.ctx_kv, backend_);
        if (gc2.buf_kv) {
            ggml_cgraph *gf2 = build_gpt_step_graph(
                *this, hp_, blocks_, gpt_ln_f_w_, gpt_ln_f_b_,
                final_norm_w_, final_norm_b_, mel_head_w_, mel_head_b_,
                gpt_wpe_w_, gc2, 0, full_len);
            ggml_backend_sched_reset(sched);
            bool ok = ggml_backend_sched_alloc_graph(sched, gf2);
            if (ok) {
                std::vector<ggml_fp16_t> mask2(
                    (size_t)full_len * full_len, ggml_fp32_to_fp16(0.0f));
                const ggml_fp16_t neg_inf2 = ggml_fp32_to_fp16(-INFINITY);
                for (int q2 = 0; q2 < full_len; ++q2)
                    for (int k2 = q2 + 1; k2 < full_len; ++k2)
                        mask2[(size_t)q2 * full_len + k2] = neg_inf2;
                ggml_backend_tensor_set(
                    ggml_graph_get_tensor(gf2, "inputs_embeds"),
                    full_seq.data(), 0, full_seq.size() * sizeof(float));
                set_position_ids(gf2, 0, full_len);
                ggml_backend_tensor_set(
                    ggml_graph_get_tensor(gf2, "causal_mask"),
                    mask2.data(), 0, mask2.size() * sizeof(ggml_fp16_t));
                ok = (ggml_backend_sched_graph_compute(sched, gf2) ==
                      GGML_STATUS_SUCCESS);
            }
            if (ok) {
                ggml_tensor *hidden_full =
                    ggml_graph_get_tensor(gf2, "gpt_last_hidden_full");
                if (hidden_full) {
                    std::vector<float> full_hidden((size_t)D * full_len);
                    ggml_backend_tensor_get(hidden_full, full_hidden.data(), 0,
                                           full_hidden.size() * sizeof(float));
                    s.lm_latent.resize((size_t)T_mel * D);
                    for (int i = 0; i < T_mel; ++i)
                        std::memcpy(&s.lm_latent[(size_t)i * D],
                                    &full_hidden[(size_t)(prefix_len + i) * D],
                                    D * sizeof(float));
                    RS_LOG_INFO("[indextts2] lm_latent extracted [%d, %d]",
                                T_mel, D);
                } else {
                    RS_LOG_WARN("[indextts2] gpt_last_hidden_full not in graph "
                                "— lm_latent zero");
                    s.lm_latent.assign((size_t)T_mel * D, 0.0f);
                }
            } else {
                RS_LOG_WARN("[indextts2] second-pass compute failed — "
                            "lm_latent zero");
                s.lm_latent.assign((size_t)T_mel * D, 0.0f);
            }
            ggml_backend_buffer_free(gc2.buf_kv);
        } else {
            RS_LOG_WARN("[indextts2] second-pass KV alloc failed — "
                        "lm_latent zero");
            s.lm_latent.assign((size_t)T_mel * D, 0.0f);
        }
        ggml_free(gc2.ctx_kv);
    } else {
        s.lm_latent.clear();
    }

    ggml_backend_buffer_free(gc.buf_kv);
    ggml_free(gc.ctx_kv);
    return true;
}

// -----------------------------------------------------------------------------
// Smoke harness: load inputs_embeds.npy from RS_INDEXTTS2_GPT_TEST_DIR, run a
// single prefill pass through build_gpt_step_graph, and dump the three named
// intermediates for scripts/diff_gpt_prefill.py.
// -----------------------------------------------------------------------------

// Local .npy reader mirroring the s2mel-smoke helper (kept here to avoid a
// shared header just for the bring-up scaffolding).
static bool gpt_smoke_load_npy_f32(const std::string &path,
                                   std::vector<int> &shape,
                                   std::vector<float> &data) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { RS_LOG_ERR("[gpt-smoke] cannot open %s", path.c_str()); return false; }
    char magic[10];
    if (std::fread(magic, 1, 10, f) != 10 ||
        std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        RS_LOG_ERR("[gpt-smoke] %s: not a .npy", path.c_str());
        std::fclose(f);
        return false;
    }
    uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
    std::vector<char> hdr(hlen);
    if (std::fread(hdr.data(), 1, hlen, f) != hlen) {
        std::fclose(f);
        return false;
    }
    std::string hs(hdr.begin(), hdr.end());
    if (hs.find("'<f4'") == std::string::npos ||
        hs.find("'fortran_order': False") == std::string::npos) {
        RS_LOG_ERR("[gpt-smoke] %s: must be C-order float32", path.c_str());
        std::fclose(f);
        return false;
    }
    size_t shp = hs.find("'shape':");
    size_t lp  = hs.find('(', shp);
    size_t rp  = hs.find(')', lp);
    std::string shape_str = hs.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        std::string tok = shape_str.substr(pos, comma == std::string::npos
                                                  ? std::string::npos
                                                  : comma - pos);
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
        if (!tok.empty()) shape.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    size_t n = 1; for (int sd : shape) n *= (size_t)sd;
    data.assign(n, 0.0f);
    if (std::fread(data.data(), sizeof(float), n, f) != n) {
        RS_LOG_ERR("[gpt-smoke] %s: short read", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

static bool gpt_smoke_dump_f32(const std::string &path,
                               const std::vector<float> &v) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { RS_LOG_WARN("[gpt-smoke] cannot open %s for write", path.c_str()); return false; }
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return true;
}

// Lightweight int64 .npy loader (only used by the prefix smoke for text_inputs).
static bool gpt_smoke_load_npy_i64(const std::string &path,
                                   std::vector<int>   &shape,
                                   std::vector<int64_t> &data) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[10];
    if (std::fread(magic, 1, 10, f) != 10 ||
        std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fclose(f); return false;
    }
    uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
    std::vector<char> hdr(hlen);
    if (std::fread(hdr.data(), 1, hlen, f) != hlen) {
        std::fclose(f); return false;
    }
    std::string hs(hdr.begin(), hdr.end());
    // Accept little-endian int64 ('<i8') only — what np.save emits on macOS/x86/ARM.
    if (hs.find("'<i8'") == std::string::npos ||
        hs.find("'fortran_order': False") == std::string::npos) {
        RS_LOG_ERR("[gpt-smoke] %s: must be C-order int64", path.c_str());
        std::fclose(f); return false;
    }
    size_t shp = hs.find("'shape':");
    size_t lp  = hs.find('(', shp);
    size_t rp  = hs.find(')', lp);
    std::string shape_str = hs.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        std::string tok = shape_str.substr(pos, comma == std::string::npos
                                                  ? std::string::npos
                                                  : comma - pos);
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
        if (!tok.empty()) shape.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    size_t n = 1; for (int sd : shape) n *= (size_t)sd;
    data.assign(n, 0);
    if (std::fread(data.data(), sizeof(int64_t), n, f) != n) {
        std::fclose(f); return false;
    }
    std::fclose(f);
    return true;
}

bool Model::RunGPTPrefillSmoke(State &s, ggml_backend_sched_t sched,
                               bool &taken) {
    (void)s;
    taken = false;
    const char *dir_c = std::getenv("RS_INDEXTTS2_GPT_TEST_DIR");
    if (!dir_c) return true;
    std::string dir = dir_c;
    if (!dir.empty() && dir.back() != '/') dir.push_back('/');
    taken = true;

    if (!gpt_ln_f_w_ || !final_norm_w_ || !mel_head_w_ || blocks_.empty()) {
        RS_LOG_ERR("[gpt-smoke] GPT weights not loaded — gpt.gguf missing?");
        return false;
    }

    const int D = hp_.n_embd;

    // ---- Load inputs_embeds.npy ----------------------------------------
    std::vector<int>   sh_emb;
    std::vector<float> embeds;
    if (!gpt_smoke_load_npy_f32(dir + "inputs_embeds.npy", sh_emb, embeds))
        return false;
    if (sh_emb.size() != 2 || sh_emb[1] != D) {
        RS_LOG_ERR("[gpt-smoke] inputs_embeds.npy shape mismatch: got [%d,%d], "
                   "want [*, %d]",
                   sh_emb.size() >= 1 ? sh_emb[0] : -1,
                   sh_emb.size() >= 2 ? sh_emb[1] : -1, D);
        return false;
    }
    const int T = sh_emb[0];
    RS_LOG_INFO("[gpt-smoke] T_prefix=%d D=%d V_mel=%d", T, D, hp_.mel_codes);

    // ---- Optional: rebuild prefix from cond_full + text_inputs and diff vs
    //      inputs_embeds.npy. Validates build_prefix_embeds without needing
    //      Conformer/Perceiver/emo to be implemented yet.
    {
        std::vector<int>   sh_cf;
        std::vector<float> cond_full;
        std::vector<int>   sh_ti;
        std::vector<int64_t> text_i64;
        bool have_cf = gpt_smoke_load_npy_f32(dir + "cond_full.npy",
                                              sh_cf, cond_full);
        bool have_ti = gpt_smoke_load_npy_i64(dir + "text_inputs.npy",
                                              sh_ti, text_i64);
        // Optional: rebuild cond_full from raw pieces (spk_latents + emo_vec +
        // speed_emb lookup) to validate the assemble + speed_emb path.
        std::vector<int>   sh_spk, sh_ev;
        std::vector<float> spk_latents, emo_vec_buf;
        bool have_spk = gpt_smoke_load_npy_f32(dir + "spk_latents.npy",
                                               sh_spk, spk_latents);
        bool have_ev  = gpt_smoke_load_npy_f32(dir + "emo_vec.npy",
                                               sh_ev,  emo_vec_buf);
        if (have_spk && have_ev && speed_emb_w_ &&
            sh_spk.size() == 2 && sh_spk[0] == hp_.cond_latents &&
            sh_spk[1] == D && sh_ev.size() == 1 && sh_ev[0] == D) {
            std::vector<float> speed_emb_table((size_t)2 * D);
            tensor_get_f32(speed_emb_w_, speed_emb_table.data(),
                           speed_emb_table.size());
            std::vector<float> cf_built = assemble_cond_full(
                hp_, spk_latents, emo_vec_buf, speed_emb_table);
            gpt_smoke_dump_f32(dir + "cond_full.f32", cf_built);
            RS_LOG_INFO("[gpt-smoke] dumped cond_full.f32 (assembled from "
                        "spk+emo+speed_emb)");
            // Prefer the C++-assembled cond_full for the prefix build so we
            // exercise the full path including speed_emb lookup.
            cond_full = std::move(cf_built);
            have_cf = true;
            sh_cf = { hp_.cond_latents + 2, D };
        }
        if (have_cf && have_ti && text_emb_w_ && text_pos_emb_w_ &&
            mel_emb_w_ && mel_pos_emb_w_) {
            const int cond_total = hp_.cond_latents + 2;
            if (sh_cf.size() == 2 && sh_cf[0] == cond_total && sh_cf[1] == D) {
                std::vector<float> text_emb_table((size_t)hp_.text_vocab * D);
                tensor_get_f32(text_emb_w_, text_emb_table.data(),
                               text_emb_table.size());
                std::vector<float> text_pos_table(
                    (size_t)(hp_.max_text_tok + 2) * D);
                tensor_get_f32(text_pos_emb_w_, text_pos_table.data(),
                               text_pos_table.size());
                std::vector<float> mel_emb_table((size_t)hp_.mel_codes * D);
                tensor_get_f32(mel_emb_w_, mel_emb_table.data(),
                               mel_emb_table.size());
                std::vector<float> mel_pos_table(
                    (size_t)(hp_.max_mel_tok + 3) * D);
                tensor_get_f32(mel_pos_emb_w_, mel_pos_table.data(),
                               mel_pos_table.size());
                std::vector<int32_t> text_tok;
                text_tok.reserve(text_i64.size());
                for (int64_t v : text_i64) text_tok.push_back((int32_t)v);

                std::vector<float> prefix_built = build_prefix_embeds(
                    hp_, text_emb_table, text_pos_table,
                    mel_emb_table, mel_pos_table, cond_full, text_tok);
                gpt_smoke_dump_f32(dir + "prefix_embeds.f32", prefix_built);
                RS_LOG_INFO("[gpt-smoke] dumped prefix_embeds.f32 (T=%d)",
                            (int)(prefix_built.size() / D));
            } else {
                RS_LOG_WARN("[gpt-smoke] cond_full.npy shape mismatch "
                            "(got %d,%d expect %d,%d)",
                            sh_cf.size() >= 1 ? sh_cf[0] : -1,
                            sh_cf.size() >= 2 ? sh_cf[1] : -1,
                            cond_total, D);
            }
        }
    }

    // ---- KV cache (same shape as RunAR) --------------------------------
    GPTContext gc;
    gc.kv_max_ctx = std::max(T,
        hp_.cond_latents + hp_.max_text_tok + hp_.max_mel_tok + 16);
    gc.compute_meta.resize(ggml_tensor_overhead() * 16384 +
                           ggml_graph_overhead_custom(16384, false));
    ggml_init_params kvp = { ggml_tensor_overhead() * 8, nullptr, true };
    gc.ctx_kv = ggml_init(kvp);
    gc.kv_k = ggml_new_tensor_4d(gc.ctx_kv, GGML_TYPE_F16,
                                 hp_.head_dim, gc.kv_max_ctx,
                                 hp_.n_head, hp_.n_layer);
    gc.kv_v = ggml_new_tensor_4d(gc.ctx_kv, GGML_TYPE_F16,
                                 hp_.head_dim, gc.kv_max_ctx,
                                 hp_.n_head, hp_.n_layer);
    gc.buf_kv = ggml_backend_alloc_ctx_tensors(gc.ctx_kv, backend_);
    if (!gc.buf_kv) {
        RS_LOG_ERR("[gpt-smoke] KV cache alloc failed");
        ggml_free(gc.ctx_kv);
        return false;
    }

    // ---- Build prefill graph (n_past=0, n_tokens=T) --------------------
    ggml_cgraph *gf = build_gpt_step_graph(
        *this, hp_, blocks_, gpt_ln_f_w_, gpt_ln_f_b_,
        final_norm_w_, final_norm_b_, mel_head_w_, mel_head_b_,
        gpt_wpe_w_, gc, /*n_past=*/0, /*n_tokens=*/T);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[gpt-smoke] graph alloc failed");
        ggml_backend_buffer_free(gc.buf_kv);
        ggml_free(gc.ctx_kv);
        return false;
    }

    // Causal mask (only created when T > 1; here we expect T >> 1).
    std::vector<ggml_fp16_t> mask((size_t)T * T, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; ++q)
        for (int k = q + 1; k < T; ++k)
            mask[(size_t)q * T + k] = neg_inf;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"),
                            embeds.data(), 0, embeds.size() * sizeof(float));
    set_position_ids(gf, 0, T);
    if (ggml_tensor *m = ggml_graph_get_tensor(gf, "causal_mask")) {
        ggml_backend_tensor_set(m, mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[gpt-smoke] graph compute failed");
        ggml_backend_buffer_free(gc.buf_kv);
        ggml_free(gc.ctx_kv);
        return false;
    }

    // ---- Read back outputs --------------------------------------------
    auto read_out = [&](const char *name, std::vector<float> &dst,
                        size_t n_elem) -> bool {
        ggml_tensor *t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            RS_LOG_ERR("[gpt-smoke] missing tensor '%s'", name);
            return false;
        }
        dst.assign(n_elem, 0.0f);
        tensor_get_f32(t, dst.data(), n_elem);
        return true;
    };

    std::vector<float> hidden_full, fn_out, logits;
    if (!read_out("gpt_last_hidden_full", hidden_full, (size_t)T * D))   return false;
    if (!read_out("final_norm_out",       fn_out,      (size_t)D))       return false;
    if (!read_out("logits",               logits,      (size_t)hp_.mel_codes)) return false;

    gpt_smoke_dump_f32(dir + "gpt_last_hidden.f32", hidden_full);
    gpt_smoke_dump_f32(dir + "final_norm_out.f32", fn_out);
    gpt_smoke_dump_f32(dir + "mel_head_logits.f32", logits);

    int first_tok = (int)std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end()));
    {
        FILE *f = std::fopen((dir + "first_sampled_token.txt").c_str(), "w");
        if (f) { std::fprintf(f, "%d\n", first_tok); std::fclose(f); }
    }
    RS_LOG_INFO("[gpt-smoke] first_sampled_token=%d  hidden[0..3]=%.4f %.4f %.4f",
                first_tok,
                hidden_full.size() > 0 ? hidden_full[0] : 0.0f,
                hidden_full.size() > 1 ? hidden_full[1] : 0.0f,
                hidden_full.size() > 2 ? hidden_full[2] : 0.0f);

    ggml_backend_buffer_free(gc.buf_kv);
    ggml_free(gc.ctx_kv);
    return true;
}

} // namespace indextts2
