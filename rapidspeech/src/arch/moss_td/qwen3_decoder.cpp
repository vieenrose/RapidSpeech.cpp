#include "qwen3_decoder.hpp"

#include "backend.hpp"
#include "ggml_extend.hpp"
#include "common.hpp"   // MT_LOGE

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mt {

Qwen3Decoder::~Qwen3Decoder() {
    if (kv_buffer_) ggml_backend_buffer_free(kv_buffer_);
}

bool Qwen3Decoder::load(const ModelLoader& m, int max_seq, int n_streams) {
    m_         = &m;
    max_seq_   = max_seq;
    n_streams_ = n_streams > 0 ? n_streams : 1;
    past_len_.assign(n_streams_, 0);
    logical_next_.assign(n_streams_, 0);

    // Hparams come from mt::Config (mtd.text.*), NOT from qwen3.* KV: the
    // converter did not write the qwen3.* metadata block.
    const Config& c = m.config();
    hp_.hidden       = c.text_hidden;
    hp_.n_heads      = c.text_heads;
    hp_.n_kv_heads   = c.text_kv_heads;
    hp_.head_dim     = c.text_head_dim;   // 128 — independent of hidden/n_heads
    hp_.intermediate = c.text_ffn;
    hp_.n_layers     = c.text_layers;
    hp_.text_vocab   = c.text_vocab;
    hp_.rope_base    = c.text_rope_theta;  // 1e6
    hp_.rms_eps      = c.text_rms_eps;     // 1e-6
    hp_.use_rope     = true;

    if (hp_.hidden <= 0 || hp_.n_layers <= 0 || hp_.n_heads <= 0 ||
        hp_.n_kv_heads <= 0 || hp_.head_dim <= 0 || max_seq_ <= 0) {
        MT_LOGE("Qwen3Decoder: bad hparams");
        return false;
    }

    layers_.assign(hp_.n_layers, Qwen3Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!qwen3_load_layer(m, "qwen3", i, &layers_[i])) {
            MT_LOGE("Qwen3Decoder: failed to load layer %d", i);
            return false;
        }
    }

    output_norm_ = m.tensor("qwen3.output_norm.weight");
    if (!output_norm_) { MT_LOGE("Qwen3Decoder: missing qwen3.output_norm.weight"); return false; }

    token_embd_ = m.tensor("token_embd.weight");   // [hidden, vocab] — tied lm_head
    if (!token_embd_) { MT_LOGE("Qwen3Decoder: missing token_embd.weight"); return false; }

    // Persistent device-resident K/V cache: 2 tensors per layer PER STREAM,
    // [head_dim, n_kv_heads, max_seq, 1], indexed [stream * L + layer].
    const int L = hp_.n_layers;
    const size_t n_kv_tensors = (size_t)2 * L * n_streams_;
    kv_ctx_ = make_ctx(ggml_tensor_overhead() * n_kv_tensors + 1024, /*no_alloc=*/true);
    if (!kv_ctx_) return false;
    k_cache_.assign((size_t)L * n_streams_, nullptr);
    v_cache_.assign((size_t)L * n_streams_, nullptr);
    // MT_KV_F16=1 (opt-in): store the cache in F16. Halves KV bytes and the
    // per-token KV bandwidth that dominates long-context decode on CPU.
    // ggml_cpy converts on store; attention scores stay GGML_PREC_F32.
    // Default remains F32 (the byte-validated configuration).
    const char* kvenv = std::getenv("MT_KV_F16");
    const ggml_type kvt = (kvenv && atoi(kvenv) != 0) ? GGML_TYPE_F16
                                                      : GGML_TYPE_F32;
    for (size_t i = 0; i < (size_t)L * n_streams_; ++i) {
        k_cache_[i] = ggml_new_tensor_4d(kv_ctx_.get(), kvt,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
        v_cache_[i] = ggml_new_tensor_4d(kv_ctx_.get(), kvt,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
    }
    kv_buffer_ = allocate_ctx_tensors(kv_ctx_.get());
    if (!kv_buffer_) { MT_LOGE("Qwen3Decoder: KV cache alloc failed"); return false; }

    // Reused metadata-ctx buffer for every run()/decode_batch() graph (avoids
    // a big malloc per step). Node budget: a batch-B step needs about
    // L*(shared ~32 + ~24 per stream) nodes; 4096 covers up to batch 4,
    // larger batches double the budget until it fits.
    graph_nodes_ = 4096;
    const int need = L * (32 + 24 * n_streams_) + 64;
    while (graph_nodes_ < need) graph_nodes_ *= 2;
    scratch_.resize(ggml_tensor_overhead() * 2 * (size_t)graph_nodes_
                  + ggml_graph_overhead_custom(graph_nodes_, false)
                  + (1u << 20));
    return true;
}

void Qwen3Decoder::reset() {
    // views are bounded by past_len_; stale cache bytes are never read
    std::fill(past_len_.begin(), past_len_.end(), 0);
    std::fill(logical_next_.begin(), logical_next_.end(), 0);
}

int Qwen3Decoder::evict_stream(int s, int lo, int keep_from) {
    if (s < 0 || s >= n_streams_) return 0;
    if (lo < 0 || keep_from <= lo || keep_from > past_len_[s]) return 0;
    const int delta = keep_from - lo;
    const int tail  = past_len_[s] - keep_from;
    if (tail > 0) {
        std::vector<uint8_t> stage;
        for (int l = 0; l < hp_.n_layers; ++l) {
            for (struct ggml_tensor* kv : {kc(s, l), vc(s, l)}) {
                const size_t col = kv->nb[2];   // bytes per cached position
                stage.resize((size_t)tail * col);
                ggml_backend_tensor_get(kv, stage.data(),
                                        (size_t)keep_from * col,
                                        (size_t)tail * col);
                ggml_backend_tensor_set(kv, stage.data(),
                                        (size_t)lo * col,
                                        (size_t)tail * col);
            }
        }
    }
    past_len_[s] -= delta;
    return delta;
}

bool Qwen3Decoder::run(int stream, const std::vector<float>& embeds, int T,
                       std::vector<float>* out_hidden) {
    const int L     = hp_.n_layers;
    const int H     = hp_.hidden;
    const int past  = past_len_[stream];
    const int kv    = past + T;
    const int lbase = logical_next_[stream];  // RoPE positions; == past unless evicting

    if (embeds.size() < (size_t)H * T) { MT_LOGE("Qwen3Decoder: embeds too small"); return false; }
    if (past + T > max_seq_) { MT_LOGE("Qwen3Decoder: sequence exceeds max_seq"); return false; }

    auto cctx = make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    // Causal mask only needed when T>1 (prefill). For a T=1 decode step every
    // cached key is causally valid, so mask=null (no bias).
    struct ggml_tensor* mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, T);  // ne0=key, ne1=query
        ggml_set_input(mask);
    }

    auto* gf = ggml_new_graph_custom(ctx, graph_nodes_, false);

    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        auto lo = qwen3_layer_forward(ctx, h, pos, mask, /*k_past=*/nullptr, /*v_past=*/nullptr,
                                      layers_[l], hp_, gf, kc(stream, l), vc(stream, l), past);
        h = lo.y;
    }

    // Final RMSNorm on ALL positions -> [H, T].
    struct ggml_tensor* y = ggml_mul(ctx, ggml_rms_norm(ctx, h, hp_.rms_eps), output_norm_);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    const float ninf = -std::numeric_limits<float>::infinity();
    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, embeds.data(), 0, (size_t)H * T * sizeof(float));

        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = lbase + i;
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)T * sizeof(int32_t));

        if (mask) {
            // mask[query i][key j]: attend iff key j <= query abs pos past+i.
            std::vector<float> mvec((size_t)kv * T);
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < kv; ++j)
                    mvec[(size_t)i * kv + j] = (j <= past + i) ? 0.0f : ninf;
            ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));
        }
    };

    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    past_len_[stream] = kv;
    logical_next_[stream] = lbase + T;
    out_hidden->resize((size_t)H * T);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * T * sizeof(float));
    return true;
}

bool Qwen3Decoder::prefill(const std::vector<float>& embeds, int S,
                           std::vector<float>* all_hidden) {
    return prefill_stream(0, embeds, S, all_hidden);
}

bool Qwen3Decoder::prefill_stream(int s, const std::vector<float>& embeds, int S,
                                  std::vector<float>* all_hidden) {
    if (s < 0 || s >= n_streams_) { MT_LOGE("Qwen3Decoder: bad stream %d", s); return false; }
    past_len_[s] = 0;
    logical_next_[s] = 0;
    return run(s, embeds, S, all_hidden);
}

std::vector<float> Qwen3Decoder::decode_one(const std::vector<float>& embed) {
    std::vector<float> hidden;
    if (!run(0, embed, 1, &hidden)) hidden.clear();
    return hidden;
}

bool Qwen3Decoder::decode_batch(const std::vector<float>& embeds,
                                const std::vector<int>& streams,
                                std::vector<float>* out_hidden) {
    const int B = (int)streams.size();
    const int L = hp_.n_layers;
    const int H = hp_.hidden;
    if (B <= 0 || !out_hidden) return false;
    if (embeds.size() < (size_t)H * B) { MT_LOGE("Qwen3Decoder: batch embeds too small"); return false; }
    for (int s : streams) {
        if (s < 0 || s >= n_streams_) { MT_LOGE("Qwen3Decoder: bad stream %d", s); return false; }
        if (past_len_[s] + 1 > max_seq_) { MT_LOGE("Qwen3Decoder: stream %d exceeds max_seq", s); return false; }
    }

    auto cctx = make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, B);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, B);
    ggml_set_input(pos);

    auto* gf = ggml_new_graph_custom(ctx, graph_nodes_, false);

    std::vector<Qwen3StreamKV> kvs(B);
    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        for (int b = 0; b < B; ++b) {
            kvs[b].k_cache = kc(streams[b], l);
            kvs[b].v_cache = vc(streams[b], l);
            kvs[b].past    = past_len_[streams[b]];
        }
        h = qwen3_layer_forward_batch(ctx, h, pos, layers_[l], hp_, gf,
                                      kvs.data(), B);
    }

    struct ggml_tensor* y = ggml_mul(ctx, ggml_rms_norm(ctx, h, hp_.rms_eps), output_norm_);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, embeds.data(), 0, (size_t)H * B * sizeof(float));
        std::vector<int32_t> p(B);
        for (int b = 0; b < B; ++b) p[b] = logical_next_[streams[b]];
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)B * sizeof(int32_t));
    };

    if (!compute_graph_with_inputs(gf, set_inputs)) return false;

    for (int b = 0; b < B; ++b) {
        past_len_[streams[b]]     += 1;
        logical_next_[streams[b]] += 1;
    }
    out_hidden->resize((size_t)H * B);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * B * sizeof(float));
    return true;
}

std::vector<float> Qwen3Decoder::logits_from_hidden(const std::vector<float>& hidden_row) {
    std::vector<float> out;
    const int H = hp_.hidden;
    if ((int)hidden_row.size() < H || !token_embd_) return out;

    // token_embd_ ne=[hidden, vocab]; mul_mat with [hidden,1] -> [vocab,1].
    size_t buf_sz = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + (1u << 16);
    std::vector<uint8_t> buf(buf_sz);
    auto cctx = make_ctx_buf(buf.data(), buf.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_set_input(hin);
    struct ggml_tensor* logits = ggml_mul_mat(ctx, token_embd_, hin);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_set_output(logits);

    auto* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(hin, hidden_row.data(), 0, (size_t)H * sizeof(float));
    };
    if (!compute_graph_with_inputs(gf, set_inputs)) return out;

    read_tensor_f32(logits, &out);
    return out;
}

}  // namespace mt
