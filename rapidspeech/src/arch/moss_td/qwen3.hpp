#ifndef MT_QWEN3_HPP
#define MT_QWEN3_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <string>
namespace mt {
struct Qwen3Hparams { int hidden=0,n_heads=0,n_kv_heads=0,head_dim=0,intermediate=0,n_layers=0,text_vocab=0; float rope_base=1e6f, rms_eps=1e-6f; bool use_rope = true; };
struct Qwen3Layer {
    struct ggml_tensor *attn_norm=nullptr,*attn_q=nullptr,*attn_k=nullptr,*attn_v=nullptr,*attn_o=nullptr,
                       *q_norm=nullptr,*k_norm=nullptr,*ffn_norm=nullptr,*ffn_gate=nullptr,*ffn_up=nullptr,*ffn_down=nullptr;
};
// Load one transformer layer's weights. The prefix overload reads
// `{prefix}.blk.{i}.*` (e.g. "qwen3" or "local"); the 2-arg overload defaults
// to prefix="qwen3" so all V1 callers keep working unchanged.
bool qwen3_load_layer(const ModelLoader& m, const std::string& prefix, int i, Qwen3Layer* out);  // {prefix}.blk.{i}.*
bool qwen3_load_layer(const ModelLoader& m, int i, Qwen3Layer* out);   // qwen3.blk.{i}.*
// On the additive path k_full/v_full are contiguous (gallocr-safe) graph
// outputs. On the device-cache path (k_cache != null) k_store/v_store are the
// in-graph ggml_cpy nodes that write the new columns into the persistent cache
// (already build_forward_expand'd by the helper); k_full/v_full stay null.
struct Qwen3LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr,
                                           *k_store=nullptr,*v_store=nullptr; };
// x:[hidden,T]; pos:int32[T]; mask:[kv,T] additive (0/-inf) or null (= no bias).
// Additive path: k_past/v_past null on prefill (kv=T); returns k_full/v_full.
// Device-cache path (k_cache != null): store the T new columns into k_cache at
// sequence offset past_seq, read the [0:past_seq+T] prefix by view; the helper
// build_forward_expand's the store nodes into gf (ordering: store before read).
Qwen3LayerOut qwen3_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Qwen3Layer& w, const Qwen3Hparams& hp,
    struct ggml_cgraph* gf = nullptr,
    struct ggml_tensor* k_cache = nullptr, struct ggml_tensor* v_cache = nullptr,
    int past_seq = 0);

// ---- batched multi-stream decode (one token per independent stream) ----
//
// One K/V cache set + past length per stream. The layer runs the SHARED
// weight-bound ops (attn_norm, q/k/v projections, q/k head norms, RoPE,
// o-projection, FFN) once on x:[hidden, B] — each of the B columns is an
// independent dot-product per output row, so per-column numerics match the
// single-stream [hidden, 1] path — and runs attention as B per-stream
// branches, each structurally identical to the single-stream T=1 cache path
// (cpy the stream's new K/V column into ITS cache at past, attend over that
// cache's [0, past+1) prefix, GGML_PREC_F32 scores, no mask). RoPE positions
// come per COLUMN from pos:int32[B] (each stream's own logical position).
// Per-stream attention outputs are ggml_concat'd back to [n_h*hd, B].
struct Qwen3StreamKV {
    struct ggml_tensor* k_cache = nullptr;   // [hd, n_kv_h, max_seq, 1]
    struct ggml_tensor* v_cache = nullptr;
    int past = 0;                            // valid cached columns
};
// x:[hidden, B]; pos:int32[B] (per-stream logical positions); kvs: B entries.
// Returns the layer output [hidden, B]. K/V stores are expanded into gf
// before the attention reads (same ordering contract as the single path).
struct ggml_tensor* qwen3_layer_forward_batch(struct ggml_context* ctx,
    struct ggml_tensor* x, struct ggml_tensor* pos,
    const Qwen3Layer& w, const Qwen3Hparams& hp, struct ggml_cgraph* gf,
    const Qwen3StreamKV* kvs, int n_streams);
}  // namespace mt
#endif
