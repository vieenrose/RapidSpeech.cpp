#include "arch/semantic_codec.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace semantic_codec {

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
                              float eps = 1e-6f) {
    ggml_tensor *y = ggml_norm(ctx, x, eps);
    if (gamma) y = ggml_mul(ctx, y, gamma);
    if (beta) {
        ggml_tensor *b2 = ggml_reshape_2d(ctx, beta, (int64_t)beta->ne[0], 1);
        y = ggml_add(ctx, y, b2);
    }
    return y;
}

// Conv1d with k=7, symmetric padding 3. Input [C_in, T].
// GGUF stores Conv1d weights as [K, IC, OC] (ggml-friendly layout).
// Uses manual im2col + mul_mat (same pattern as omnivoice conv_1d_f32).
// Output: [C_out, T] in ggml [ne0=C_out, ne1=T] layout.
static ggml_tensor *conv1d_k7p3(ggml_context *ctx, ggml_tensor *x,
                                 ggml_tensor *w, ggml_tensor *b) {
    int C_in  = (int)x->ne[0];
    int T     = (int)x->ne[1];
    int K     = (int)w->ne[0];  // GGUF layout: [K, IC, OC]
    int IC    = (int)w->ne[1];
    int C_out = (int)w->ne[2];

    // Kernel is already [K, IC, OC] — no permute needed.

    // Reshape input from [C_in, T] to [T, C_in, 1] (3D) for ggml_im2col.
    ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, x));    // [T, C_in]
    ggml_tensor *x3d = ggml_reshape_3d(ctx, xt, T, C_in, 1);   // [T, C_in, 1]

    ggml_tensor *im2col = ggml_im2col(ctx, w, x3d, 1, 0, 3, 0, 1, 0, false, GGML_TYPE_F16);
    // im2col: [IC*K, T_out, 1, 1]

    ggml_tensor *result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));

    // result: [T_out, OC, 1] — transpose to [C_out, T_out] for subsequent layers.
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], w->ne[2], im2col->ne[2]);
    result = ggml_reshape_2d(ctx, result, result->ne[0], result->ne[1]);  // [T_out, OC]
    result = ggml_cont(ctx, ggml_transpose(ctx, result));                 // [OC, T_out]

    if (b) {
        ggml_tensor *b2 = ggml_reshape_2d(ctx, b, (int64_t)C_out, 1);
        result = ggml_add(ctx, result, b2);
    }
    return result;
}

// Depthwise Conv1d k=7, symmetric padding 3.
// Input [C, T]. GGUF stores dwconv kernel as [K, 1, C].
// Internally transposes to [T, C], pads, conv_1d_dw, transposes back.
static ggml_tensor *dwconv_k7p3(ggml_context *ctx, ggml_tensor *x,
                                 ggml_tensor *w, ggml_tensor *b) {
    int C = (int)x->ne[0];
    int T = (int)x->ne[1];
    int K = (int)w->ne[0];  // GGUF layout: [K, 1, C]

    // Transpose to [T, C] for ggml_conv_1d_dw.
    ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Symmetric pad: 3 left, 3 right on T dimension (dim 0 after transpose).
    ggml_tensor *xp = ggml_pad_ext(ctx, xt, K/2, K/2, 0, 0, 0, 0, 0, 0);

    // Kernel is already [K, 1, C] — no permute needed.
    ggml_tensor *wf16 = ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor *yd = ggml_conv_1d_dw(ctx, wf16, xp, 1, 0, 1);

    // Output shape after conv_1d_dw: [T, C]. Transpose back to [C, T].
    ggml_tensor *yt = ggml_reshape_2d(ctx, yd, T, C);
    ggml_tensor *out = ggml_cont(ctx, ggml_transpose(ctx, yt));
    if (b) {
        ggml_tensor *b2 = ggml_reshape_2d(ctx, b, (int64_t)C, 1);
        out = ggml_add(ctx, out, b2);
    }
    return out;
}

static int32_t gguf_u32_or(gguf_context *ctx, const char *key, int32_t dflt) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return dflt;
    return (int32_t)gguf_get_val_u32(ctx, i);
}

static std::string semantic_dump_dir() {
    const char *d = std::getenv("RS_SEMANTIC_CODEC_DUMP_DIR");
    if (!d || !*d) return {};
    std::string s = d;
    if (!s.empty() && s.back() != '/') s.push_back('/');
    return s;
}

static void dump_f32_file(const std::string &path, const std::vector<float> &v) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Build encoder graph: VocosBackbone + output projection.
// x: [hidden=1024, T] → output: [hidden=1024, T]
// ---------------------------------------------------------------------------
static ggml_tensor *build_encoder(ggml_context *ctx, ggml_tensor *x,
                                   const Weights &w, const HParams &hp,
                                   std::vector<ggml_tensor *> *dump_nodes) {
    const int C     = hp.hidden;     // 1024
    const int Vdim  = hp.vocos_dim;  // 384

    // embed: Conv1d(C→Vdim, k=7, p=3) → [Vdim, T]
    x = conv1d_k7p3(ctx, x, w.enc_embed_w, w.enc_embed_b);
    if (dump_nodes) { ggml_set_name(x, "sc_embed"); ggml_set_output(x); dump_nodes->push_back(x); }

    // LayerNorm
    x = layernorm(ctx, x, w.enc_norm_w, w.enc_norm_b);
    if (dump_nodes) { ggml_set_name(x, "sc_norm"); ggml_set_output(x); dump_nodes->push_back(x); }

    // 12 ConvNeXt blocks.
    for (int i = 0; i < hp.num_layers; ++i) {
        const auto &blk = w.convnext_blocks[i];
        ggml_tensor *res = x;

        // Depthwise conv k=7, p=3.
        ggml_tensor *t = dwconv_k7p3(ctx, x, blk.dwconv_w, blk.dwconv_b);

        // LayerNorm (on Vdim).
        t = layernorm(ctx, t, blk.norm_w, blk.norm_b);

        // pwconv1: Linear(Vdim → intermediate).
        t = linear(ctx, t, blk.pwconv1_w, blk.pwconv1_b);

        // GELU.
        t = ggml_gelu(ctx, t);

        // pwconv2: Linear(intermediate → Vdim).
        t = linear(ctx, t, blk.pwconv2_w, blk.pwconv2_b);

        // Gamma scaling.
        if (blk.gamma) {
            ggml_tensor *g = ggml_reshape_2d(ctx, blk.gamma, Vdim, 1);
            t = ggml_mul(ctx, t, g);
        }

        // Residual.
        x = ggml_add(ctx, res, t);
        if (dump_nodes) {
            char name[32];
            std::snprintf(name, sizeof(name), "sc_block%02d", i);
            ggml_set_name(x, name);
            ggml_set_output(x);
            dump_nodes->push_back(x);
        }
    }

    // Final LayerNorm.
    x = layernorm(ctx, x, w.enc_final_ln_w, w.enc_final_ln_b);
    if (dump_nodes) { ggml_set_name(x, "sc_final_ln"); ggml_set_output(x); dump_nodes->push_back(x); }

    // Output projection: Linear(Vdim → C).
    x = linear(ctx, x, w.enc_out_w, w.enc_out_b);

    return x;
}

// ---------------------------------------------------------------------------
// Host-side VQ quantize: L2-normalize → find nearest codebook entry → out_proj.
// enc_out: [T, C] C-order. Returns sc_hidden [T, C] C-order, codes [T].
// ---------------------------------------------------------------------------
static std::vector<float> read_tensor_f32(ggml_tensor *t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i)
            out[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        std::fprintf(stderr, "[semantic_codec] unsupported tensor type %d\n", t->type);
        std::abort();
    }
    return out;
}

static std::vector<float> weight_norm_reconstruct(ggml_tensor *v_ggml,
                                                    ggml_tensor *g_ggml,
                                                    int out_dim, int in_dim) {
    std::vector<float> v = read_tensor_f32(v_ggml);
    std::vector<float> g = read_tensor_f32(g_ggml);
    int64_t n_g = g.size();

    std::vector<float> w(out_dim * in_dim, 0.0f);

    for (int o = 0; o < out_dim; ++o) {
        double norm2 = 0.0;
        for (int i = 0; i < in_dim; ++i) {
            // PyTorch Conv1d weight_v is stored C-order as [out_dim, in_dim, 1].
            float val = v[(size_t)o * in_dim + i];
            norm2 += (double)val * val;
        }
        float scale = (float)(1.0 / std::sqrt(norm2 + 1e-12));
        float g_val = g[o < (int)n_g ? o : 0];
        for (int i = 0; i < in_dim; ++i) {
            w[(size_t)o * in_dim + i] =
                g_val * scale * v[(size_t)o * in_dim + i];
        }
    }
    return w;
}

static void vq_quantize(const float *enc_out, int T,
                         const HParams &hp, const Weights &w,
                         float *sc_hidden, int *codes) {
    const int C  = hp.hidden;
    const int D  = hp.codebook_dim;
    const int CS = hp.codebook_size;

    // Read codebook [CS, D] C-order.
    std::vector<float> cb(CS * D);
    {
        std::vector<float> cb_ggml = read_tensor_f32(w.q_codebook);
        std::copy(cb_ggml.begin(), cb_ggml.end(), cb.begin());
    }

    // L2-normalize codebook rows.
    std::vector<float> cb_norm(CS * D);
    for (int c = 0; c < CS; ++c) {
        double sum2 = 0.0;
        for (int d = 0; d < D; ++d) sum2 += (double)cb[c * D + d] * cb[c * D + d];
        float inv = (float)(1.0 / std::sqrt(sum2 + 1e-12));
        for (int d = 0; d < D; ++d) cb_norm[c * D + d] = cb[c * D + d] * inv;
    }

    // Reconstruct in_proj weight [D, C] from weight_norm.
    std::vector<float> in_proj_w = weight_norm_reconstruct(
        w.q_in_proj_w_v, w.q_in_proj_w_g, D, C);
    std::vector<float> in_proj_b(D, 0.0f);
    if (w.q_in_proj_b) {
        std::vector<float> b = read_tensor_f32(w.q_in_proj_b);
        std::copy(b.begin(), b.end(), in_proj_b.begin());
    }

    // Reconstruct out_proj weight [C, D] from weight_norm.
    std::vector<float> out_proj_w = weight_norm_reconstruct(
        w.q_out_proj_w_v, w.q_out_proj_w_g, C, D);
    std::vector<float> out_proj_b(C, 0.0f);
    if (w.q_out_proj_b) {
        std::vector<float> b = read_tensor_f32(w.q_out_proj_b);
        std::copy(b.begin(), b.end(), out_proj_b.begin());
    }

    // Process each time step.
    std::vector<float> proj(D);
    std::vector<float> qry(D);
    for (int t = 0; t < T; ++t) {
        const float *row = enc_out + (size_t)t * C;

        // in_proj: proj = row @ in_proj_w^T + in_proj_b  → [D]
        for (int d = 0; d < D; ++d) {
            double acc = (double)in_proj_b[d];
            for (int c = 0; c < C; ++c)
                acc += (double)row[c] * (double)in_proj_w[d * C + c];
            proj[d] = (float)acc;
        }

        // L2-normalize.
        double norm2 = 0.0;
        for (int d = 0; d < D; ++d) norm2 += (double)proj[d] * proj[d];
        float inv_norm = (float)(1.0 / std::sqrt(norm2 + 1e-12));
        for (int d = 0; d < D; ++d) qry[d] = proj[d] * inv_norm;

        // Find nearest codebook entry (max dot product).
        int best_c = 0;
        float best_dot = -1e30f;
        for (int c = 0; c < CS; ++c) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += (double)qry[d] * (double)cb_norm[c * D + d];
            if (dot > best_dot) {
                best_dot = (float)dot;
                best_c = c;
            }
        }
        if (codes) codes[t] = best_c;

        // out_proj: sc_hidden[t] = codebook[best_c] @ out_proj_w^T + out_proj_b
        for (int c = 0; c < C; ++c) {
            double acc = (double)out_proj_b[c];
            for (int d = 0; d < D; ++d)
                acc += (double)cb[best_c * D + d] * (double)out_proj_w[c * D + d];
            sc_hidden[(size_t)t * C + c] = (float)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
bool SemanticCodecModel::Load(ggml_context *gguf_data, gguf_context *ctx_gguf,
                               ggml_backend_t backend,
                               const std::string &tensor_prefix) {
    backend_ = backend;

    hp_.hidden        = gguf_u32_or(ctx_gguf, (tensor_prefix + "hidden").c_str(),        hp_.hidden);
    hp_.vocos_dim     = gguf_u32_or(ctx_gguf, (tensor_prefix + "vocos_dim").c_str(),     hp_.vocos_dim);
    hp_.intermediate  = gguf_u32_or(ctx_gguf, (tensor_prefix + "intermediate").c_str(),  hp_.intermediate);
    hp_.num_layers    = gguf_u32_or(ctx_gguf, (tensor_prefix + "num_layers").c_str(),    hp_.num_layers);
    hp_.codebook_size = gguf_u32_or(ctx_gguf, (tensor_prefix + "codebook_size").c_str(), hp_.codebook_size);
    hp_.codebook_dim  = gguf_u32_or(ctx_gguf, (tensor_prefix + "codebook_dim").c_str(),  hp_.codebook_dim);

    w_.convnext_blocks.resize(hp_.num_layers);
    int n_bound = 0;
    const size_t pfx_len = tensor_prefix.size();

    for (ggml_tensor *t = ggml_get_first_tensor(gguf_data); t != nullptr;
         t = ggml_get_next_tensor(gguf_data, t)) {
        const std::string name = ggml_get_name(t);
        if (name.size() <= pfx_len || name.compare(0, pfx_len, tensor_prefix) != 0)
            continue;
        const std::string key = name.substr(pfx_len);

        // encoder.0 (VocosBackbone)
        if (key == "encoder.0.embed.weight")         { w_.enc_embed_w     = t; n_bound++; continue; }
        if (key == "encoder.0.embed.bias")           { w_.enc_embed_b     = t; n_bound++; continue; }
        if (key == "encoder.0.norm.weight")          { w_.enc_norm_w      = t; n_bound++; continue; }
        if (key == "encoder.0.norm.bias")            { w_.enc_norm_b      = t; n_bound++; continue; }
        if (key == "encoder.0.final_layer_norm.weight") { w_.enc_final_ln_w = t; n_bound++; continue; }
        if (key == "encoder.0.final_layer_norm.bias")   { w_.enc_final_ln_b = t; n_bound++; continue; }

        // encoder.1 (output Linear)
        if (key == "encoder.1.weight") { w_.enc_out_w = t; n_bound++; continue; }
        if (key == "encoder.1.bias")   { w_.enc_out_b = t; n_bound++; continue; }

        // ConvNeXt blocks: encoder.0.convnext.<i>.<param>
        if (key.compare(0, 19, "encoder.0.convnext.") == 0) {
            size_t dot2 = key.find('.', 19);
            if (dot2 == std::string::npos) continue;
            int idx = std::atoi(key.c_str() + 19);
            if (idx < 0 || idx >= hp_.num_layers) continue;
            std::string sub = key.substr(dot2 + 1);
            auto &blk = w_.convnext_blocks[idx];

            if      (sub == "dwconv.weight")     { blk.dwconv_w  = t; n_bound++; continue; }
            else if (sub == "dwconv.bias")       { blk.dwconv_b  = t; n_bound++; continue; }
            else if (sub == "norm.weight")       { blk.norm_w    = t; n_bound++; continue; }
            else if (sub == "norm.bias")         { blk.norm_b    = t; n_bound++; continue; }
            else if (sub == "pwconv1.weight")    { blk.pwconv1_w = t; n_bound++; continue; }
            else if (sub == "pwconv1.bias")      { blk.pwconv1_b = t; n_bound++; continue; }
            else if (sub == "pwconv2.weight")    { blk.pwconv2_w = t; n_bound++; continue; }
            else if (sub == "pwconv2.bias")      { blk.pwconv2_b = t; n_bound++; continue; }
            else if (sub == "gamma")             { blk.gamma     = t; n_bound++; continue; }
        }

        // quantizer (ResidualVQ with weight_norm projections)
        if (key == "quantizer.quantizers.0.codebook.weight")   { w_.q_codebook     = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.in_project.weight_g") { w_.q_in_proj_w_g = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.in_project.weight_v") { w_.q_in_proj_w_v = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.in_project.bias")     { w_.q_in_proj_b   = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.out_project.weight_g") { w_.q_out_proj_w_g = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.out_project.weight_v") { w_.q_out_proj_w_v = t; n_bound++; continue; }
        if (key == "quantizer.quantizers.0.out_project.bias")     { w_.q_out_proj_b  = t; n_bound++; continue; }
    }

    if (!w_.enc_embed_w || !w_.q_codebook) {
        std::fprintf(stderr, "[semantic_codec] FATAL: encoder or quantizer not found "
                     "under prefix '%s'\n", tensor_prefix.c_str());
        return false;
    }
    std::printf("[semantic_codec] loaded %d tensors, %d layers, "
                "hidden=%d codebook=%dx%d\n",
                n_bound, hp_.num_layers, hp_.hidden,
                hp_.codebook_size, hp_.codebook_dim);
    return true;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
bool SemanticCodecModel::Forward(const float *input_features, int T,
                                  float *sc_hidden_out, int *codes_out,
                                  ggml_backend_sched_t sched) {
    const int C = hp_.hidden;

    // Estimate graph size.
    const size_t n_nodes = (size_t)hp_.num_layers * 40 + 32;
    const size_t meta_sz  = ggml_tensor_overhead() * n_nodes * 4 +
                            ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> compute_meta(meta_sz);
    ggml_init_params params = { compute_meta.size(), compute_meta.data(), true };
    ggml_context *ctx = ggml_init(params);
    if (!ctx) { std::fprintf(stderr, "[semantic_codec] ggml_init failed\n"); return false; }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // Input [C, T] ggml layout.
    ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_set_name(inp, "input_features");
    ggml_set_input(inp);

    // Encoder.
    std::vector<ggml_tensor *> dump_nodes;
    const std::string dump_dir = semantic_dump_dir();
    ggml_tensor *enc_out = build_encoder(ctx, inp, w_, hp_,
                                         dump_dir.empty() ? nullptr : &dump_nodes);
    ggml_set_name(enc_out, "enc_out");
    ggml_set_output(enc_out);

    ggml_build_forward_expand(gf, enc_out);

    // Compute encoder.
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "[semantic_codec] sched_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set input data. ggml tensor shape is [C, T], with ne[0] contiguous.
    // That memory layout is identical to C-order [T, C].
    {
        std::vector<float> inp_ggml((size_t)C * T);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < C; ++d)
                inp_ggml[(size_t)t * C + d] = input_features[(size_t)t * C + d];
        ggml_backend_tensor_set(inp, inp_ggml.data(), 0,
                                inp_ggml.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[semantic_codec] graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read encoder output. [C, T] ggml contiguous memory is already C-order
    // [T, C].
    std::vector<float> enc_host((size_t)C * T);
    {
        ggml_tensor *t = ggml_graph_get_tensor(gf, "enc_out");
        if (!t) {
            std::fprintf(stderr, "[semantic_codec] enc_out not found\n");
            ggml_free(ctx); return false;
        }
        std::vector<float> buf((size_t)C * T);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        std::memcpy(enc_host.data(), buf.data(), buf.size() * sizeof(float));
    }

    if (!dump_dir.empty()) {
        for (ggml_tensor *node : dump_nodes) {
            const char *name = ggml_get_name(node);
            const int D_node = (int)node->ne[0];
            const int T_node = (int)node->ne[1];
            std::vector<float> buf((size_t)D_node * T_node);
            ggml_backend_tensor_get(node, buf.data(), 0, buf.size() * sizeof(float));
            dump_f32_file(dump_dir + name + ".f32", buf);
        }
        dump_f32_file(dump_dir + "sc_enc_out.f32", enc_host);
    }

    ggml_free(ctx);

    // Host-side VQ quantize.
    vq_quantize(enc_host.data(), T, hp_, w_, sc_hidden_out, codes_out);

    if (!dump_dir.empty()) {
        std::vector<float> out(sc_hidden_out, sc_hidden_out + (size_t)T * C);
        dump_f32_file(dump_dir + "sc_hidden.f32", out);
        if (codes_out) {
            std::vector<float> codes(T);
            for (int i = 0; i < T; ++i) codes[i] = (float)codes_out[i];
            dump_f32_file(dump_dir + "sc_codes.f32", codes);
        }
    }

    return true;
}

bool SemanticCodecModel::DecodeCodes(const int32_t *codes, int T,
                                      float *sc_hidden_out) {
    if (!codes || T <= 0 || !sc_hidden_out || !w_.q_codebook ||
        !w_.q_out_proj_w_v || !w_.q_out_proj_w_g) {
        return false;
    }

    const int C  = hp_.hidden;
    const int D  = hp_.codebook_dim;
    const int CS = hp_.codebook_size;

    // q_codebook is stored C-order as [CS, D].
    std::vector<float> cb((size_t)CS * D);
    {
        std::vector<float> cb_ggml = read_tensor_f32(w_.q_codebook);
        std::copy(cb_ggml.begin(), cb_ggml.end(), cb.begin());
    }

    std::vector<float> out_proj_w = weight_norm_reconstruct(
        w_.q_out_proj_w_v, w_.q_out_proj_w_g, C, D);
    std::vector<float> out_proj_b(C, 0.0f);
    if (w_.q_out_proj_b) {
        std::vector<float> b = read_tensor_f32(w_.q_out_proj_b);
        std::copy(b.begin(), b.end(), out_proj_b.begin());
    }

    for (int t = 0; t < T; ++t) {
        int code = codes[t];
        if (code < 0 || code >= CS) code = 0;
        const float *code_row = &cb[(size_t)code * D];
        float *dst = &sc_hidden_out[(size_t)t * C];
        for (int c = 0; c < C; ++c) {
            double acc = (double)out_proj_b[c];
            for (int d = 0; d < D; ++d) {
                acc += (double)code_row[d] * (double)out_proj_w[(size_t)c * D + d];
            }
            dst[c] = (float)acc;
        }
    }

    return true;
}

} // namespace semantic_codec
