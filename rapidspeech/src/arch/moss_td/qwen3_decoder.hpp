#ifndef MT_QWEN3_DECODER_HPP
#define MT_QWEN3_DECODER_HPP

// Qwen3-0.6B text decoder for moss-transcribe.cpp.
//
// The transformer stack itself is the vendored `qwen3_layer_forward` (same
// Qwen3 arch as moss-tts: NEOX RoPE base 1e6, per-head q/k RMSNorm before RoPE,
// GQA via mul_mat broadcast, GGML_PREC_F32 scores, SwiGLU). This driver:
//   - builds Qwen3Hparams from mt::Config (NOT from any qwen3.* KV, which the
//     converter did not write),
//   - loads token_embd.weight (tied lm_head) + qwen3.output_norm.weight,
//   - owns a persistent per-layer device-resident K/V cache,
//   - prefills S embedding rows and returns the final-RMSNorm'd hidden for
//     EVERY position (so parity can be checked per-position),
//   - applies the tied lm_head (token_embd.weight @ hidden) on demand.

#include "qwen3.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t

#include <cstdint>
#include <vector>

namespace mt {

class Qwen3Decoder {
public:
    Qwen3Decoder() = default;
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder&) = delete;             // owns a backend buffer
    Qwen3Decoder& operator=(const Qwen3Decoder&) = delete;

    // Build hparams from m.config(), load layers + token_embd + output_norm,
    // and allocate the persistent K/V cache for up to max_seq positions.
    // n_streams > 1 allocates one INDEPENDENT cache set (2*L tensors) plus
    // per-stream past_len/logical_next per stream, for batched multi-window
    // decoding (decode_batch). Stream 0 is the implicit stream of the
    // single-stream API below, so existing callers are unchanged.
    bool load(const ModelLoader& m, int max_seq, int n_streams = 1);

    int hidden() const { return hp_.hidden; }
    const Qwen3Hparams& hparams() const { return hp_; }

    // Prefill S embedding rows (token-major, S*hidden floats: row s = positions
    // s's hidden vector). Resets the cache to position 0. Writes the per-position
    // final-RMSNorm'd hidden states (token-major S*hidden) into *all_hidden.
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* all_hidden);

    // Tied lm_head: logits = token_embd.weight @ hidden_row -> [vocab].
    std::vector<float> logits_from_hidden(const std::vector<float>& hidden_row);

    // One decode step at the next position. embed: [hidden]. Returns the
    // final-RMSNorm'd hidden [hidden]; advances past_len().
    std::vector<float> decode_one(const std::vector<float>& embed);

    // Append T positions in ONE graph (no cache reset) -- benchmarking aid
    // for the batched-decode amortization question; same matmul shapes as a
    // batch-T decode step.
    bool append_batch(const std::vector<float>& embeds, int T,
                      std::vector<float>* out) { return run(0, embeds, T, out); }

    void reset();                    // resets ALL streams
    int  past_len(int s = 0) const { return past_len_[s]; }
    int  n_streams() const { return n_streams_; }

    // Audio-KV eviction: physically compact the cache by sliding columns
    // [keep_from, past_len_) down onto lo, dropping [lo, keep_from). Cached K
    // keeps its baked RoPE (eviction removes keys, never renumbers them); new
    // tokens keep monotone LOGICAL positions (logical_next_), which eviction
    // does not rewind. Returns the number of columns removed.
    int evict(int lo, int keep_from) { return evict_stream(0, lo, keep_from); }

    // ---- batched multi-stream API (load(..., n_streams > 1)) ----

    // Prefill stream s: reset THAT stream's cache to position 0 and run the
    // single-stream graph against it (same shapes/ops as prefill()).
    bool prefill_stream(int s, const std::vector<float>& embeds, int S,
                        std::vector<float>* all_hidden);

    // One decode step for B = streams.size() streams in ONE graph: shared
    // projections/FFN on [hidden, B], per-stream attention over each stream's
    // own cache, per-column RoPE positions. embeds: B token-major rows
    // (b*hidden..); out_hidden gets B final-RMSNorm'd rows in the same order.
    // Advances past_len/logical_next of exactly the listed streams.
    bool decode_batch(const std::vector<float>& embeds,
                      const std::vector<int>& streams,
                      std::vector<float>* out_hidden);

    // Per-stream audio-KV eviction (same semantics as evict()).
    int evict_stream(int s, int lo, int keep_from);

private:
    bool run(int stream, const std::vector<float>& embeds, int T,
             std::vector<float>* out_hidden);
    struct ggml_tensor* kc(int s, int l) const { return k_cache_[(size_t)s * hp_.n_layers + l]; }
    struct ggml_tensor* vc(int s, int l) const { return v_cache_[(size_t)s * hp_.n_layers + l]; }

    Qwen3Hparams hp_{};
    std::vector<Qwen3Layer> layers_;
    struct ggml_tensor* output_norm_ = nullptr;
    struct ggml_tensor* token_embd_  = nullptr;   // [hidden, vocab] — tied lm_head
    const ModelLoader* m_ = nullptr;
    int max_seq_ = 0, n_streams_ = 1, graph_nodes_ = 4096;
    std::vector<int> past_len_;      // per stream
    std::vector<int> logical_next_;  // per stream; >= past_len once evicting

    // Device-resident per-layer K/V cache, one SET per stream:
    // [head_dim, n_kv_heads, max_seq, 1], indexed [stream * n_layers + layer].
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-run graph-metadata ctx buffer
};

}  // namespace mt

#endif  // MT_QWEN3_DECODER_HPP
