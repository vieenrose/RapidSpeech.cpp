// W2V-BERT 2.0 forward — one-shot ggml compute graph.
//
// Macaron Conformer block (HF Wav2Vec2BertEncoderLayer):
//   FFN1(half) → MHSA(relative_key) → ConvModule(GLU+dw+SiLU) → FFN2(half) → finalLN
//
// Relative-key attention: scores = QK^T/sqrt(d) + einsum(Q, pos_emb)/sqrt(d)
// where pos_emb[i,j,:] = distance_embedding[clamp(j-i, -L, R)+L, :].
// Implemented via Q@dist_emb^T → gather → flash_attn bias.
//
// The graph follows the CrispASR indextts2_gpt.cpp conformer_block pattern
// (ggml ne[0]=D fastest, mul_mat conventions, flash_attn_ext).

#include "w2v_bert.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace w2v_bert {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x,
                           ggml_tensor *w, ggml_tensor *b) {
    if (!w) return x;
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    if (b) {
        ggml_tensor *b2 = ggml_reshape_2d(ctx, b, (int64_t)b->ne[0], 1);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

static ggml_tensor *layernorm(ggml_context *ctx, ggml_tensor *x,
                              ggml_tensor *gamma, ggml_tensor *beta,
                              float eps = 1e-5f) {
    ggml_tensor *y = ggml_norm(ctx, x, eps);
    if (gamma) y = ggml_mul(ctx, y, gamma);
    if (beta) {
        ggml_tensor *b2 = ggml_reshape_2d(ctx, beta, (int64_t)beta->ne[0], 1);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

// Build the [T*T] gather index: gather_idx[q*T + k] = clamp(k-q, -L, R) + L,
// a value in [0, dist_emb_size). Used by ggml_get_rows to materialize the
// per-(q,k) positional embedding inside the graph (avoids host-side gather
// and keeps memory at O(T*T*4) instead of O(T*T*head_dim*4) per layer).
static void build_dist_gather_idx(int T, int left_max, int right_max,
                                  std::vector<int32_t> &idx) {
    idx.resize((size_t)T * T);
    for (int q = 0; q < T; ++q) {
        for (int k = 0; k < T; ++k) {
            int d = k - q;
            if (d < -left_max) d = -left_max;
            if (d > right_max) d = right_max;
            d += left_max;
            idx[(size_t)q * T + k] = d;
        }
    }
}

static int32_t gguf_u32_or(gguf_context *ctx, const char *key, int32_t dflt) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return dflt;
    return (int32_t)gguf_get_val_u32(ctx, i);
}

static std::string w2v_layer_dump_dir() {
    const char *enabled = std::getenv("RS_W2V_BERT_DUMP_LAYERS");
    if (!enabled || !*enabled || std::strcmp(enabled, "0") == 0) return {};
    const char *dir = std::getenv("RS_INDEXTTS2_W2V_BERT_TEST_DIR");
    if (!dir || !*dir) return {};
    std::string out = dir;
    if (!out.empty() && out.back() != '/') out.push_back('/');
    return out;
}

static bool dump_tensor_f32(const std::string &path, const std::vector<float> &v) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Single Conformer layer graph. gather_idx is a [T*T] I32 tensor mapping each
// (q, k) pair to a distance index in [0, dist_emb_size); it is shared across
// all layers (sequence-length-dependent but layer-independent).
// ---------------------------------------------------------------------------
static ggml_tensor *conformer_layer(ggml_context *ctx, ggml_tensor *x,
                                    const ConformerLayer &L,
                                    const HParams &hp, int T,
                                    ggml_tensor *gather_idx) {
    const int D        = hp.hidden;
    const int n_heads  = hp.n_heads;
    const int head_dim = hp.head_dim;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    // ---- 1. FFN1 (half residual) ----
    ggml_tensor *res = x;
    if (L.ffn1_fc1_w) {
        ggml_tensor *t = layernorm(ctx, x, L.ffn1_ln_w, L.ffn1_ln_b);
        t = linear(ctx, t, L.ffn1_fc1_w, L.ffn1_fc1_b);
        t = ggml_silu(ctx, t);
        t = linear(ctx, t, L.ffn1_fc2_w, L.ffn1_fc2_b);
        x = ggml_add(ctx, x, ggml_scale(ctx, t, 0.5f));
    }

    // ---- 2. MHSA with relative-key bias ----
    res = x;
    ggml_tensor *t = layernorm(ctx, x, L.attn_ln_w, L.attn_ln_b);
    ggml_tensor *Q = linear(ctx, t, L.attn_q_w, L.attn_q_b);
    ggml_tensor *K = linear(ctx, t, L.attn_k_w, L.attn_k_b);
    ggml_tensor *V = linear(ctx, t, L.attn_v_w, L.attn_v_b);

    auto reshape_heads = [&](ggml_tensor *p) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, p, head_dim, n_heads, T), 0, 2, 1, 3));
    };
    Q = reshape_heads(Q);
    K = reshape_heads(K);
    V = reshape_heads(V);

    // Relative-key bias: BD[h, q, k] = Q[h, q, :] · pgde[q, k, :] / sqrt(d).
    // Build pgde in-graph via ggml_get_rows on the per-layer distance embedding
    // table, using the shared (q,k) → distance gather index.
    //   dist_emb:    [head_dim, dist_emb_size]   (ne[0]=head_dim, ne[1]=De)
    //   gather_idx:  [T*T] I32, values in [0, De)
    //   get_rows(dist_emb, gather_idx) →  [head_dim, T*T]
    //   reshape to [head_dim, T_key, T_query] (k=fastest within each query row)
    // Then batched mul_mat over T_query:
    //   pgde: [head_dim, T_key, T_query], Qp: [head_dim, n_heads, T_query]
    //   → raw[T_key, n_heads, T_query] = BD[k, h, q]
    // Permute to [T_key, T_query, n_heads] for flash_attn_ext mask layout.
    ggml_tensor *BD = nullptr;
    if (gather_idx && L.attn_dist_emb) {
        ggml_tensor *de = L.attn_dist_emb;
        if (de->type != GGML_TYPE_F32) {
            de = ggml_cast(ctx, de, GGML_TYPE_F32);
        }
        ggml_tensor *pgde_flat = ggml_get_rows(ctx, de, gather_idx);
        ggml_tensor *pgde = ggml_reshape_3d(ctx, pgde_flat, head_dim, T, T);

        ggml_tensor *Qp = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        ggml_tensor *raw = ggml_mul_mat(ctx, pgde, Qp);
        raw = ggml_scale(ctx, raw, attn_scale);
        // flash_attn_ext requires the mask in F16, shape [n_kv, n_q, n_heads, 1].
        BD = ggml_cont(ctx, ggml_permute(ctx, raw, 0, 2, 1, 3));
        BD = ggml_cast(ctx, BD, GGML_TYPE_F16);
    }

    ggml_tensor *attn = ggml_flash_attn_ext(
        ctx, ggml_cont(ctx, Q), ggml_cont(ctx, K),
        ggml_cont(ctx, V), BD, attn_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    attn = ggml_reshape_2d(ctx, attn, D, T);
    attn = linear(ctx, attn, L.attn_o_w, L.attn_o_b);
    x = ggml_add(ctx, res, attn);

    // ---- 3. Conv module ----
    res = x;
    t = layernorm(ctx, x, L.conv_ln_w, L.conv_ln_b);

    if (L.conv_pw1_w) {
        ggml_tensor *w2d = ggml_reshape_2d(ctx, L.conv_pw1_w, D, 2 * D);
        ggml_tensor *pw1_out = ggml_mul_mat(ctx, w2d, t);

        ggml_tensor *val  = ggml_view_2d(ctx, pw1_out, D, T, pw1_out->nb[1], 0);
        ggml_tensor *gate = ggml_view_2d(ctx, pw1_out, D, T, pw1_out->nb[1],
                                         D * ggml_type_size(pw1_out->type));
        val  = ggml_cont(ctx, val);
        gate = ggml_cont(ctx, gate);
        t = ggml_mul(ctx, val, ggml_sigmoid(ctx, gate));

        // Causal depthwise conv: left-pad K-1, then conv_1d_dw with pad=0.
        ggml_tensor *t_TC = ggml_cont(ctx, ggml_transpose(ctx, t));
        int K = hp.conv_kernel;
        ggml_tensor *t_pad = ggml_pad_ext(ctx, t_TC, K - 1, 0, 0, 0, 0, 0, 0, 0);

        ggml_tensor *dw_f16 = ggml_cast(ctx, L.conv_dw_w, GGML_TYPE_F16);
        ggml_tensor *t_dw = ggml_conv_1d_dw(ctx, dw_f16, t_pad, 1, 0, 1);
        t_dw = ggml_reshape_2d(ctx, t_dw, T, D);
        t = ggml_cont(ctx, ggml_transpose(ctx, t_dw));

        t = layernorm(ctx, t, L.conv_dw_ln_w, L.conv_dw_ln_b);
        t = ggml_silu(ctx, t);

        if (L.conv_pw2_w) {
            ggml_tensor *pw2w = ggml_reshape_2d(ctx, L.conv_pw2_w, D, D);
            t = ggml_mul_mat(ctx, pw2w, t);
        }
        x = ggml_add(ctx, res, t);
    }

    // ---- 4. FFN2 (half residual) ----
    res = x;
    if (L.ffn2_fc1_w) {
        t = layernorm(ctx, x, L.ffn2_ln_w, L.ffn2_ln_b);
        t = linear(ctx, t, L.ffn2_fc1_w, L.ffn2_fc1_b);
        t = ggml_silu(ctx, t);
        t = linear(ctx, t, L.ffn2_fc2_w, L.ffn2_fc2_b);
        x = ggml_add(ctx, x, ggml_scale(ctx, t, 0.5f));
    }

    // ---- 5. Final per-block LayerNorm ----
    x = layernorm(ctx, x, L.final_ln_w, L.final_ln_b);
    return x;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
bool W2VBertModel::Load(ggml_context *gguf_data, gguf_context *ctx_gguf,
                         ggml_backend_t backend,
                         const std::string &tensor_prefix) {
    backend_ = backend;

    hp_.n_layers     = gguf_u32_or(ctx_gguf, (tensor_prefix + "n_layers").c_str(),     hp_.n_layers);
    hp_.hidden       = gguf_u32_or(ctx_gguf, (tensor_prefix + "hidden").c_str(),        hp_.hidden);
    hp_.n_heads      = gguf_u32_or(ctx_gguf, (tensor_prefix + "n_heads").c_str(),       hp_.n_heads);
    hp_.head_dim     = hp_.hidden / hp_.n_heads;
    hp_.intermediate = gguf_u32_or(ctx_gguf, (tensor_prefix + "intermediate").c_str(),  hp_.intermediate);
    hp_.conv_kernel  = gguf_u32_or(ctx_gguf, (tensor_prefix + "conv_kernel").c_str(),   hp_.conv_kernel);
    hp_.left_max_pos = gguf_u32_or(ctx_gguf, (tensor_prefix + "left_max_pos").c_str(),  hp_.left_max_pos);
    hp_.right_max_pos= gguf_u32_or(ctx_gguf, (tensor_prefix + "right_max_pos").c_str(), hp_.right_max_pos);
    hp_.fp_input_dim = gguf_u32_or(ctx_gguf, (tensor_prefix + "fp_input_dim").c_str(),  hp_.fp_input_dim);
    hp_.dist_emb_size = hp_.left_max_pos + hp_.right_max_pos + 1;

    w_.layers.resize(hp_.n_layers);
    int n_bound = 0;
    const size_t pfx_len = tensor_prefix.size();

    for (ggml_tensor *t = ggml_get_first_tensor(gguf_data); t != nullptr;
         t = ggml_get_next_tensor(gguf_data, t)) {
        const std::string name = ggml_get_name(t);
        if (name.size() <= pfx_len || name.compare(0, pfx_len, tensor_prefix) != 0)
            continue;
        const std::string key = name.substr(pfx_len);

        if (key == "fp.ln.weight")        { w_.fp_ln_w   = t; n_bound++; continue; }
        if (key == "fp.ln.bias")          { w_.fp_ln_b   = t; n_bound++; continue; }
        if (key == "fp.proj.weight")      { w_.fp_proj_w = t; n_bound++; continue; }
        if (key == "fp.proj.bias")        { w_.fp_proj_b = t; n_bound++; continue; }

        if (key.compare(0, 7, "layers.") != 0) continue;
        size_t dot2 = key.find('.', 7);
        if (dot2 == std::string::npos) continue;
        int L = std::atoi(key.c_str() + 7);
        if (L < 0 || L >= hp_.n_layers) continue;
        const std::string sub = key.substr(dot2 + 1);
        auto &ly = w_.layers[L];

        if      (sub == "ffn1_ln.weight")   { ly.ffn1_ln_w    = t; n_bound++; continue; }
        else if (sub == "ffn1_ln.bias")     { ly.ffn1_ln_b    = t; n_bound++; continue; }
        else if (sub == "ffn1.fc1.weight")  { ly.ffn1_fc1_w   = t; n_bound++; continue; }
        else if (sub == "ffn1.fc1.bias")    { ly.ffn1_fc1_b   = t; n_bound++; continue; }
        else if (sub == "ffn1.fc2.weight")  { ly.ffn1_fc2_w   = t; n_bound++; continue; }
        else if (sub == "ffn1.fc2.bias")    { ly.ffn1_fc2_b   = t; n_bound++; continue; }
        else if (sub == "attn_ln.weight")   { ly.attn_ln_w    = t; n_bound++; continue; }
        else if (sub == "attn_ln.bias")     { ly.attn_ln_b    = t; n_bound++; continue; }
        else if (sub == "attn.q.weight")    { ly.attn_q_w     = t; n_bound++; continue; }
        else if (sub == "attn.q.bias")      { ly.attn_q_b     = t; n_bound++; continue; }
        else if (sub == "attn.k.weight")    { ly.attn_k_w     = t; n_bound++; continue; }
        else if (sub == "attn.k.bias")      { ly.attn_k_b     = t; n_bound++; continue; }
        else if (sub == "attn.v.weight")    { ly.attn_v_w     = t; n_bound++; continue; }
        else if (sub == "attn.v.bias")      { ly.attn_v_b     = t; n_bound++; continue; }
        else if (sub == "attn.o.weight")    { ly.attn_o_w     = t; n_bound++; continue; }
        else if (sub == "attn.o.bias")      { ly.attn_o_b     = t; n_bound++; continue; }
        else if (sub == "attn.dist_emb.weight") { ly.attn_dist_emb = t; n_bound++; continue; }
        else if (sub == "conv.ln.weight")   { ly.conv_ln_w    = t; n_bound++; continue; }
        else if (sub == "conv.ln.bias")     { ly.conv_ln_b    = t; n_bound++; continue; }
        else if (sub == "conv.pw1.weight")  { ly.conv_pw1_w   = t; n_bound++; continue; }
        else if (sub == "conv.dw.weight")   { ly.conv_dw_w    = t; n_bound++; continue; }
        else if (sub == "conv.dw_ln.weight"){ ly.conv_dw_ln_w = t; n_bound++; continue; }
        else if (sub == "conv.dw_ln.bias")  { ly.conv_dw_ln_b = t; n_bound++; continue; }
        else if (sub == "conv.pw2.weight")  { ly.conv_pw2_w   = t; n_bound++; continue; }
        else if (sub == "ffn2_ln.weight")   { ly.ffn2_ln_w    = t; n_bound++; continue; }
        else if (sub == "ffn2_ln.bias")     { ly.ffn2_ln_b    = t; n_bound++; continue; }
        else if (sub == "ffn2.fc1.weight")  { ly.ffn2_fc1_w   = t; n_bound++; continue; }
        else if (sub == "ffn2.fc1.bias")    { ly.ffn2_fc1_b   = t; n_bound++; continue; }
        else if (sub == "ffn2.fc2.weight")  { ly.ffn2_fc2_w   = t; n_bound++; continue; }
        else if (sub == "ffn2.fc2.bias")    { ly.ffn2_fc2_b   = t; n_bound++; continue; }
        else if (sub == "final_ln.weight")  { ly.final_ln_w   = t; n_bound++; continue; }
        else if (sub == "final_ln.bias")    { ly.final_ln_b   = t; n_bound++; continue; }
    }

    if (!w_.fp_ln_w || !w_.fp_proj_w) {
        std::fprintf(stderr, "[w2v_bert] FATAL: feature_projection not found "
                     "under prefix '%s'\n", tensor_prefix.c_str());
        return false;
    }
    std::printf("[w2v_bert] loaded %d tensors, %d layers, hidden=%d heads=%d\n",
                n_bound, hp_.n_layers, hp_.hidden, hp_.n_heads);
    return true;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
bool W2VBertModel::Forward(const float *input_features, int T,
                            float *hidden0_out, float *hidden17_out,
                            ggml_backend_sched_t sched) {
    const int D    = hp_.hidden;
    const int D_in = hp_.fp_input_dim;

    if (T != cached_T_) {
        build_dist_gather_idx(T, hp_.left_max_pos, hp_.right_max_pos,
                              dist_gather_idx_);
        cached_T_ = T;
    }

    const size_t n_nodes = (size_t)hp_.n_layers * 128 + 64;
    const size_t meta_sz  = ggml_tensor_overhead() * n_nodes * 4 +
                            ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> compute_meta(meta_sz);
    ggml_init_params params = { compute_meta.size(), compute_meta.data(), true };
    ggml_context *ctx = ggml_init(params);
    if (!ctx) { std::fprintf(stderr, "[w2v_bert] ggml_init failed\n"); return false; }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // ---- Input ----
    ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_in, T);
    ggml_set_name(inp, "input_features");
    ggml_set_input(inp);

    // ---- Shared (q,k) → distance index gather tensor ----
    ggml_tensor *gather_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)T * T);
    ggml_set_name(gather_idx, "rel_key_gather_idx");
    ggml_set_input(gather_idx);

    // ---- Feature projection ----
    ggml_tensor *x = layernorm(ctx, inp, w_.fp_ln_w, w_.fp_ln_b);
    x = linear(ctx, x, w_.fp_proj_w, w_.fp_proj_b);

    // hidden0: feature_projection output (input to layer 0).
    ggml_tensor *hidden0 = x;
    ggml_set_name(hidden0, "hidden0");
    ggml_set_output(hidden0);

    // Upstream IndexTTS-2 uses HF `hidden_states[17]`. In transformers, index
    // 0 is the feature_projection output, so index 17 is after encoder layer
    // 16. The GGUF may contain layer 17 too, but it is not part of this path.
    const int n_run_layers = std::min(hp_.n_layers, 17);

    // ---- Conformer layers 0..16 ----
    std::vector<ggml_tensor *> layer_outputs;
    const std::string layer_dump_dir = w2v_layer_dump_dir();
    if (!layer_dump_dir.empty()) layer_outputs.reserve(n_run_layers);
    for (int Li = 0; Li < n_run_layers; ++Li) {
        x = conformer_layer(ctx, x, w_.layers[Li], hp_, T, gather_idx);
        if (!layer_dump_dir.empty()) {
            char name[32];
            std::snprintf(name, sizeof(name), "layer%02d", Li);
            ggml_set_name(x, name);
            ggml_set_output(x);
            layer_outputs.push_back(x);
        }
    }
    ggml_set_name(x, "hidden17");
    ggml_set_output(x);

    // Register the full graph.
    ggml_build_forward_expand(gf, x);

    // ---- Allocate ----
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "[w2v_bert] sched_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // ---- Set input data ----
    {
        std::vector<float> inp_ggml((size_t)D_in * T);
        // ggml tensor shape is [D_in, T], with ne[0] contiguous. That memory
        // layout is identical to C-order [T, D_in].
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_in; ++d)
                inp_ggml[(size_t)t * D_in + d] =
                    input_features[(size_t)t * D_in + d];
        ggml_backend_tensor_set(inp, inp_ggml.data(), 0,
                                inp_ggml.size() * sizeof(float));
    }

    // Upload the shared (q,k) → distance gather index.
    ggml_backend_tensor_set(gather_idx, dist_gather_idx_.data(), 0,
                            dist_gather_idx_.size() * sizeof(int32_t));

    // ---- Compute ----
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[w2v_bert] graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // ---- Read outputs → C-order [T, D] ----
    auto read_out = [&](const char *name, float *dst) {
        ggml_tensor *t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            std::fprintf(stderr, "[w2v_bert] output '%s' not found\n", name);
            return false;
        }
        std::vector<float> buf((size_t)D * T);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        // Output tensor is [D, T] in ggml, which is already C-order [T, D]
        // in the contiguous buffer.
        std::memcpy(dst, buf.data(), buf.size() * sizeof(float));
        return true;
    };

    if (hidden17_out && !read_out("hidden17", hidden17_out)) { ggml_free(ctx); return false; }
    if (hidden0_out  && !read_out("hidden0",  hidden0_out))  { ggml_free(ctx); return false; }

    if (!layer_dump_dir.empty()) {
        for (int Li = 0; Li < (int)layer_outputs.size(); ++Li) {
            ggml_tensor *lt = layer_outputs[Li];
            std::vector<float> buf((size_t)D * T);
            ggml_backend_tensor_get(lt, buf.data(), 0, buf.size() * sizeof(float));
            char name[64];
            std::snprintf(name, sizeof(name), "w2v_layer%02d.f32", Li);
            dump_tensor_f32(layer_dump_dir + name, buf);
        }
    }

    ggml_free(ctx);
    return true;
}

} // namespace w2v_bert
