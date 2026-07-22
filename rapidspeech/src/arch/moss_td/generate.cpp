#include "generate.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"
#include "tokenizer.hpp"

#include <cstdlib>
#include <string>

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace mt {

// Host-side fallback for embed_rows_f32: fetch each selected row's raw bytes
// from the (possibly device) weight buffer via ggml_backend_tensor_get and
// dequantize on the CPU with the type's to_float trait. This is the reference
// dequantization, so results match the get_rows graph path. Needed because
// ggml's GPU backends implement GET_ROWS only for a subset of quant types
// (CUDA has no K-quant get_rows kernels as of ggml 0.15) and submitting the
// op anyway GGML_ABORTs the whole process (mudler/LocalAI#10862).
static bool embed_rows_f32_host(struct ggml_tensor* tok, const int32_t* ids,
                                int n_ids, int hidden,
                                std::vector<float>* out) {
    const ggml_type type = tok->type;
    const ggml_type_traits* traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (!traits || !traits->to_float)) {
        MT_LOGE("embed_rows_f32_host: no to_float for type %s",
                ggml_type_name(type));
        return false;
    }
    if (!tok->buffer) {
        MT_LOGE("embed_rows_f32_host: token_embd has no buffer");
        return false;
    }
    // The flat per-row offset math below needs dense rows.
    const size_t row_bytes = ggml_row_size(type, tok->ne[0]);
    if (tok->nb[1] != row_bytes) {
        MT_LOGE("embed_rows_f32_host: token_embd rows are not dense");
        return false;
    }
    out->resize((size_t)n_ids * (size_t)hidden);
    std::vector<uint8_t> raw(row_bytes);
    for (int p = 0; p < n_ids; ++p) {
        ggml_backend_tensor_get(tok, raw.data(), (size_t)ids[p] * row_bytes,
                                row_bytes);
        float* dst = out->data() + (size_t)p * (size_t)hidden;
        if (type == GGML_TYPE_F32) {
            std::memcpy(dst, raw.data(), (size_t)hidden * sizeof(float));
        } else {
            traits->to_float(raw.data(), dst, hidden);
        }
    }
    return true;
}

bool embed_rows_f32(struct ggml_tensor* tok, const int32_t* ids,
                    int n_ids, int hidden, std::vector<float>* out) {
    if (!tok || !ids || n_ids <= 0 || hidden <= 0 || !out) return false;

    const size_t max_nodes = 8;
    size_t buf_sz = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead() +
                    (1u << 16);
    std::vector<uint8_t> buf(buf_sz);
    GgmlCtxPtr cctx = make_ctx_buf(buf.data(), buf.size(), /*no_alloc=*/true);
    ggml_context* ctx = cctx.get();
    if (!ctx) return false;

    struct ggml_tensor* idt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
    ggml_set_input(idt);
    struct ggml_tensor* rows = ggml_get_rows(ctx, tok, idt);  // [hidden, n_ids] F32
    ggml_set_output(rows);

    // GPU backends support GET_ROWS only for a subset of src types (ggml's
    // CUDA backend aborts on K-quants), so ask first and dequantize the rows
    // on the host when the op can't run on the active backend.
    if (!ggml_backend_supports_op(backend(), rows)) {
        static std::once_flag warn_once;
        std::call_once(warn_once, [&] {
            MT_LOGW("embed lookup: %s GET_ROWS unsupported on %s backend, "
                    "using host-side dequant",
                    ggml_type_name(tok->type), backend_name());
        });
        return embed_rows_f32_host(tok, ids, n_ids, hidden, out);
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);

    const bool ok = compute_graph_with_inputs(gf, [&]() {
        ggml_backend_tensor_set(idt, ids, 0, (size_t)n_ids * sizeof(int32_t));
    });
    if (!ok) return false;
    return read_tensor_f32(rows, out);
}

std::vector<float> fuse_embeds(ModelLoader& m,
                               const std::vector<int32_t>& input_ids,
                               const std::vector<float>& audio_embeds,
                               int n_audio, int hidden, int audio_token_id) {
    std::vector<float> out;
    if (hidden <= 0) {
        MT_LOGE("fuse_embeds: invalid hidden=%d", hidden);
        return out;
    }
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) {
        MT_LOGE("fuse_embeds: missing token_embd.weight");
        return out;
    }
    // token_embd.weight ne=[hidden, vocab]: column t is the embedding of token
    // id t, laid out feature-fastest (token_embd_data[t*hidden + h]).
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("fuse_embeds: token_embd hidden %lld != %d",
                (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    const size_t  seq   = input_ids.size();
    if (seq == 0) return out;

    // Validate ids before the get_rows lookup (out-of-range ids would read
    // garbage rows).
    for (size_t p = 0; p < seq; ++p) {
        int32_t t = input_ids[p];
        if (t < 0 || t >= vocab) {
            MT_LOGE("fuse_embeds: input_ids[%zu]=%d out of range [0,%lld)",
                    p, t, (long long)vocab);
            return out;
        }
    }

    // 1) Embed lookup via ggml_get_rows: dequantizes rows for ANY token_embd
    //    type (F32/F16/quant) and returns F32 rows [hidden, seq]. On an F32
    //    token_embd this is bit-identical to the old raw column copy.
    if (!embed_rows_f32(tok, input_ids.data(), (int)seq, hidden, &out)) {
        MT_LOGE("fuse_embeds: get_rows lookup failed");
        out.clear();
        return out;
    }

    // 2) Audio injection (masked_scatter): walk positions in increasing index;
    //    the k-th position with id==audio_token_id gets audio_embeds row k.
    int k = 0;
    for (size_t p = 0; p < seq; ++p) {
        if (input_ids[p] != audio_token_id) continue;
        if (k >= n_audio) {
            MT_LOGE("fuse_embeds: more audio positions than audio rows "
                    "(n_audio=%d)", n_audio);
            out.clear();
            return out;
        }
        const float* src = audio_embeds.data() + (size_t)k * (size_t)hidden;
        float* dst = out.data() + p * (size_t)hidden;
        for (int h = 0; h < hidden; ++h) dst[h] = src[h];
        ++k;
    }
    if (k != n_audio) {
        MT_LOGW("fuse_embeds: consumed %d of %d audio rows", k, n_audio);
    }
    return out;
}

std::vector<float> embed_token(ModelLoader& m, int32_t t, int hidden) {
    std::vector<float> out;
    if (hidden <= 0) return out;
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) { MT_LOGE("embed_token: missing token_embd.weight"); return out; }
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("embed_token: token_embd hidden %lld != %d", (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    if (t < 0 || t >= vocab) {
        MT_LOGE("embed_token: id %d out of range [0,%lld)", (int)t, (long long)vocab);
        return out;
    }
    // Single-row lookup via ggml_get_rows: dequantizes for ANY token_embd type
    // (F32/F16/quant), F32 result [hidden]. Bit-identical to the old raw read
    // on an F32 token_embd.
    if (!embed_rows_f32(tok, &t, 1, hidden, &out)) {
        MT_LOGE("embed_token: get_rows lookup failed for id %d", (int)t);
        out.clear();
    }
    return out;
}

// argmax with FIRST-index-on-tie semantics (torch argmax): strict >.
static int argmax_first(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}

std::vector<int32_t> greedy_generate(Qwen3Decoder& dec, ModelLoader& m,
                                     const std::vector<float>& fused, int seq,
                                     int max_new, int eos) {
    std::vector<int32_t> ids;
    const int H = dec.hidden();
    if (H <= 0 || seq <= 0 || max_new <= 0) {
        MT_LOGE("greedy_generate: bad args (H=%d seq=%d max_new=%d)", H, seq, max_new);
        return ids;
    }

    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid)) { MT_LOGE("greedy_generate: prefill failed"); return ids; }
    if ((int)hid.size() < H * seq) { MT_LOGE("greedy_generate: short prefill hidden"); return ids; }

    // Logits from the last prefilled position.
    std::vector<float> last(hid.end() - H, hid.end());
    std::vector<float> logits = dec.logits_from_hidden(last);
    if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); return ids; }

    ids.reserve((size_t)max_new);
    for (;;) {
        int t = argmax_first(logits);
        ids.push_back(t);
        if (t == eos) break;
        if ((int)ids.size() >= max_new) break;

        std::vector<float> emb = embed_token(m, t, H);
        if (emb.empty()) { MT_LOGE("greedy_generate: embed failed @%d", t); break; }
        std::vector<float> h1 = dec.decode_one(emb);
        if ((int)h1.size() < H) { MT_LOGE("greedy_generate: decode_one failed"); break; }
        logits = dec.logits_from_hidden(h1);
        if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); break; }
    }
    return ids;
}

std::vector<int32_t> greedy_generate_evict(Qwen3Decoder& dec, ModelLoader& m,
                                           const std::vector<float>& fused,
                                           int seq, int max_new, int eos,
                                           const KvEvictOpts& ev) {
    std::vector<int32_t> ids;
    const int H = dec.hidden();
    if (H <= 0 || seq <= 0 || max_new <= 0) {
        MT_LOGE("greedy_generate_evict: bad args");
        return ids;
    }
    if (ev.window_s <= 0.0 || !ev.tok || ev.tok_per_s <= 0.0 ||
        ev.span_end <= ev.span_lo) {
        return greedy_generate(dec, m, fused, seq, max_new, eos);
    }

    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid)) { MT_LOGE("greedy_generate_evict: prefill failed"); return ids; }
    if ((int)hid.size() < H * seq) { MT_LOGE("greedy_generate_evict: short prefill hidden"); return ids; }

    std::vector<float> last(hid.end() - H, hid.end());
    std::vector<float> logits = dec.logits_from_hidden(last);
    if (logits.empty()) { MT_LOGE("greedy_generate_evict: logits failed"); return ids; }

    // Timestamp tracking over the decoded text stream. Timestamps arrive as
    // plain text "[ss.ss]" split across tokens; keep an unconsumed tail and
    // harvest every complete bracket group. Speaker tags "[S01]" fail the
    // digits/dot filter and are ignored.
    std::string tail;
    double max_ts = 0.0;
    int span_end_phys = ev.span_end;   // audio-span end in PHYSICAL columns
    int evicted_total = 0;
    auto harvest_ts = [&]() {
        size_t close;
        while ((close = tail.find(']')) != std::string::npos) {
            const size_t open = tail.rfind('[', close);
            if (open != std::string::npos && close > open + 1) {
                bool num = true; int digits = 0;
                for (size_t i = open + 1; i < close; ++i) {
                    const char ch = tail[i];
                    if (ch >= '0' && ch <= '9') { ++digits; }
                    else if (ch != '.') { num = false; break; }
                }
                if (num && digits > 0) {
                    const double ts = atof(tail.substr(open + 1, close - open - 1).c_str());
                    if (ts > max_ts) max_ts = ts;
                }
            }
            tail.erase(0, close + 1);
        }
        if (tail.size() > 32) tail.erase(0, tail.size() - 32);  // bound memory
    };

    ids.reserve((size_t)max_new);
    for (;;) {
        int t = argmax_first(logits);
        ids.push_back(t);
        if (t == eos) break;
        if ((int)ids.size() >= max_new) break;

        tail += ev.tok->decode({t});
        harvest_ts();
        if (max_ts > ev.window_s) {
            // Absolute audio column for (max_ts - W), shifted into PHYSICAL
            // space by what previous compactions already removed. Without the
            // shift every compaction after the first double-evicts.
            int keep_from = ev.span_lo +
                (int)((max_ts - ev.window_s) * ev.tok_per_s) - evicted_total;
            if (keep_from > span_end_phys) keep_from = span_end_phys;
            if (keep_from - ev.span_lo >= ev.min_batch) {
                const int delta = dec.evict(ev.span_lo, keep_from);
                evicted_total += delta;
                span_end_phys -= delta;
            }
        }

        std::vector<float> emb = embed_token(m, t, H);
        if (emb.empty()) { MT_LOGE("greedy_generate_evict: embed failed @%d", t); break; }
        std::vector<float> h1 = dec.decode_one(emb);
        if ((int)h1.size() < H) { MT_LOGE("greedy_generate_evict: decode_one failed"); break; }
        logits = dec.logits_from_hidden(h1);
        if (logits.empty()) { MT_LOGE("greedy_generate_evict: logits failed"); break; }
    }
    return ids;
}

// Timestamp harvesting for the batched loop — EXACTLY the logic inlined in
// greedy_generate_evict (kept duplicated there to avoid touching the deployed
// single-stream path): scan `tail` for complete "[...]" groups, keep the max
// numeric "[ss.ss]" value in *max_ts, drop consumed text, bound the tail.
static void harvest_ts_tail(std::string& tail, double* max_ts) {
    size_t close;
    while ((close = tail.find(']')) != std::string::npos) {
        const size_t open = tail.rfind('[', close);
        if (open != std::string::npos && close > open + 1) {
            bool num = true; int digits = 0;
            for (size_t i = open + 1; i < close; ++i) {
                const char ch = tail[i];
                if (ch >= '0' && ch <= '9') { ++digits; }
                else if (ch != '.') { num = false; break; }
            }
            if (num && digits > 0) {
                const double ts = atof(tail.substr(open + 1, close - open - 1).c_str());
                if (ts > *max_ts) *max_ts = ts;
            }
        }
        tail.erase(0, close + 1);
    }
    if (tail.size() > 32) tail.erase(0, tail.size() - 32);  // bound memory
}

std::vector<std::vector<int32_t>> greedy_generate_batch(
        Qwen3Decoder& dec, ModelLoader& m,
        const std::vector<BatchStream>& streams, int max_new, int eos,
        const BatchPrefillHook& on_prefill, const BatchTokenHook& on_token) {
    const int N = (int)streams.size();
    const int H = dec.hidden();
    std::vector<std::vector<int32_t>> all_ids(N);
    if (N <= 0 || H <= 0 || max_new <= 0) {
        MT_LOGE("greedy_generate_batch: bad args");
        return all_ids;
    }
    if (N > dec.n_streams()) {
        MT_LOGE("greedy_generate_batch: %d streams but decoder has %d", N, dec.n_streams());
        return all_ids;
    }

    struct St {
        std::vector<float> logits;
        bool failed = false;
        // eviction state (mirrors greedy_generate_evict)
        bool   evict_on = false;
        std::string tail;
        double max_ts = 0.0;
        int    span_end_phys = 0;
        int    evicted_total = 0;
    };
    std::vector<St> st(N);

    // ---- per-stream prefill (single-stream graph against cache set s) ----
    std::vector<int> active;
    for (int s = 0; s < N; ++s) {
        const BatchStream& in = streams[s];
        if (!in.fused || in.seq <= 0) { MT_LOGE("greedy_generate_batch: bad stream %d", s); st[s].failed = true; continue; }
        if (on_prefill) on_prefill(s);
        std::vector<float> hid;
        if (!dec.prefill_stream(s, *in.fused, in.seq, &hid) ||
            (int)hid.size() < H * in.seq) {
            MT_LOGE("greedy_generate_batch: prefill failed (stream %d)", s);
            st[s].failed = true;
            continue;
        }
        std::vector<float> last(hid.end() - H, hid.end());
        st[s].logits = dec.logits_from_hidden(last);
        if (st[s].logits.empty()) { MT_LOGE("greedy_generate_batch: logits failed (stream %d)", s); st[s].failed = true; continue; }
        // Same eviction gating as greedy_generate_evict's fallback check.
        const KvEvictOpts& ev = in.ev;
        st[s].evict_on = ev.window_s > 0.0 && ev.tok && ev.tok_per_s > 0.0 &&
                         ev.span_end > ev.span_lo;
        st[s].span_end_phys = ev.span_end;
        all_ids[s].reserve((size_t)max_new);
        active.push_back(s);
    }

    // ---- batched decode: one token per active stream per step ----
    std::vector<float> embeds, hid;
    while (!active.empty()) {
        std::vector<int> next;
        embeds.clear();
        for (int s : active) {
            const int t = argmax_first(st[s].logits);
            all_ids[s].push_back(t);
            const bool done = (t == eos) ||
                              ((int)all_ids[s].size() >= max_new);
            if (on_token) on_token(s, all_ids[s], done);
            if (done) continue;

            if (st[s].evict_on) {
                const KvEvictOpts& ev = streams[s].ev;
                st[s].tail += ev.tok->decode({t});
                harvest_ts_tail(st[s].tail, &st[s].max_ts);
                if (st[s].max_ts > ev.window_s) {
                    int keep_from = ev.span_lo +
                        (int)((st[s].max_ts - ev.window_s) * ev.tok_per_s) -
                        st[s].evicted_total;
                    if (keep_from > st[s].span_end_phys) keep_from = st[s].span_end_phys;
                    if (keep_from - ev.span_lo >= ev.min_batch) {
                        const int delta = dec.evict_stream(s, ev.span_lo, keep_from);
                        st[s].evicted_total += delta;
                        st[s].span_end_phys -= delta;
                    }
                }
            }

            std::vector<float> emb = embed_token(m, t, H);
            if (emb.empty()) { MT_LOGE("greedy_generate_batch: embed failed @%d (stream %d)", t, s); continue; }
            embeds.insert(embeds.end(), emb.begin(), emb.end());
            next.push_back(s);
        }
        if (next.empty()) break;

        if (!dec.decode_batch(embeds, next, &hid) ||
            (int)hid.size() < H * (int)next.size()) {
            MT_LOGE("greedy_generate_batch: decode_batch failed");
            break;
        }
        // lm_head per stream ([hidden,1]) — see header note on bit-identity.
        bool ok = true;
        for (int b = 0; b < (int)next.size(); ++b) {
            std::vector<float> row(hid.begin() + (size_t)b * H,
                                   hid.begin() + (size_t)(b + 1) * H);
            st[next[b]].logits = dec.logits_from_hidden(row);
            if (st[next[b]].logits.empty()) { MT_LOGE("greedy_generate_batch: logits failed"); ok = false; break; }
        }
        if (!ok) break;
        active = std::move(next);
    }
    return all_ids;
}

}  // namespace mt
