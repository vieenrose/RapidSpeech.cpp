#ifndef MT_GENERATE_HPP
#define MT_GENERATE_HPP

// Generation helpers for moss-transcribe.cpp.
//
// fuse_embeds: builds the fused input embeddings the Qwen3 decoder consumes.
// It performs the embed lookup (token_embd.weight rows for input_ids) followed
// by the audio injection (masked_scatter): each position whose id equals
// audio_token_id is overwritten, in increasing index order, with the next
// audio_embeds row. This is exactly torch masked_scatter semantics.

#include "model_loader.hpp"
#include "qwen3_decoder.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace mt {

// Dequantize the rows of token_embd (`tok`, ne=[hidden, vocab]) selected by
// `ids` into F32, out[p*hidden + h] = feature h of ids[p]. Works for ANY
// token_embd type (F32, F16, Q8_0, K-quants). Runs a ggml_get_rows graph on
// the active backend when it supports the op; otherwise (e.g. CUDA has no
// K-quant get_rows kernels) falls back to a host-side per-row copy + dequant
// with identical results. Returns false on failure.
bool embed_rows_f32(struct ggml_tensor* tok, const int32_t* ids, int n_ids,
                    int hidden, std::vector<float>* out);

// Fetch the embedding of token id `t` = column t of token_embd.weight
// (ne=[hidden,vocab]) -> [hidden]. Returns empty on error.
std::vector<float> embed_token(ModelLoader& m, int32_t t, int hidden);

// Greedy autoregressive generation matching torch/HF greedy decoding exactly.
//
// - Prefill `fused` (seq token-major embedding rows, seq*hidden floats).
// - Argmax the last-position logits (FIRST max index on ties) -> next token,
//   append it. If it is `eos`, stop (the returned ids INCLUDE the trailing EOS).
//   Else if we have produced max_new tokens, stop. Otherwise embed the new
//   token, decode_one (advancing the KV cache), argmax again.
//
// Returns the generated token ids (INCLUDING the trailing EOS when hit).
std::vector<int32_t> greedy_generate(Qwen3Decoder& dec, ModelLoader& m,
                                     const std::vector<float>& fused, int seq,
                                     int max_new, int eos);

// Optional audio-KV eviction for greedy_generate (MT_KV_EVICT_S): ASR
// attention is effectively local in time — when emitting text for time T the
// decoder needs audio KV near T, not minutes back. As the max emitted
// timestamp advances, columns of the prompt's audio span older than
// (max_ts - window_s) are physically compacted out of the KV cache, so
// per-token attention traffic stays O(window) instead of O(audio). Ported
// from the pre-purification RapidSpeech engine (RS_AUDIO_KV_WINDOW).
class Tokenizer;
struct KvEvictOpts {
    double window_s  = 0.0;      // eviction window in seconds (<=0: off)
    int    span_lo   = 0;        // first audio-span column in the prompt
    int    span_end  = 0;        // one past the last audio-span column
    double tok_per_s = 0.0;      // audio-span columns per second of audio
    const Tokenizer* tok = nullptr;  // to detect emitted [ss.ss] timestamps
    int    min_batch = 256;      // amortize the compaction memmove
};
std::vector<int32_t> greedy_generate_evict(Qwen3Decoder& dec, ModelLoader& m,
                                           const std::vector<float>& fused,
                                           int seq, int max_new, int eos,
                                           const KvEvictOpts& ev);

// Batched greedy decoding over N INDEPENDENT streams (windows): the decoder
// must have been load()ed with n_streams >= N. Each stream is prefilled
// separately (single-stream graph against its own cache set), then all still-
// active streams advance one token per decode_batch step so per-step weight
// reads are shared. Per-stream semantics (argmax FIRST-index tie-break, EOS /
// max_new stop, INCLUDING the trailing EOS, optional per-stream audio-KV
// eviction driven by emitted "[ss.ss]" timestamps) are exactly those of
// greedy_generate / greedy_generate_evict — per-stream results are expected
// byte-identical to sequential decoding. A stream that stops is dropped from
// the batch (the next step runs with fewer columns) until all are done.
// The lm_head stays per-stream ([hidden,1] mul_mat): token_embd is F16 in
// deployed models and ggml's llamafile sgemm would take a different (tiled)
// kernel for >=4 columns, breaking bit-identity with sequential decoding.
struct BatchStream {
    const std::vector<float>* fused = nullptr;  // seq token-major embedding rows
    int seq = 0;
    KvEvictOpts ev;   // ev.window_s <= 0 (default): no eviction for this stream
};
// Optional observation hooks (for streaming wrappers; the CLI passes neither).
// They observe only — they do not alter the decode:
// - on_prefill(stream): immediately BEFORE stream s's prefill graph runs.
// - on_token(stream, ids, done): immediately AFTER stream s appended a token;
//   ids = that stream's generated ids so far (INCLUDING the new token, and
//   including a terminal EOS); done = this token ended the stream (EOS or
//   max_new), i.e. the stream will not be in the next decode_batch step.
using BatchPrefillHook = std::function<void(int stream)>;
using BatchTokenHook   = std::function<void(int stream,
                                            const std::vector<int32_t>& ids,
                                            bool done)>;
std::vector<std::vector<int32_t>> greedy_generate_batch(
    Qwen3Decoder& dec, ModelLoader& m,
    const std::vector<BatchStream>& streams, int max_new, int eos,
    const BatchPrefillHook& on_prefill = {},
    const BatchTokenHook& on_token = {});

// Build the fused input embeddings [hidden x seq] (feature-fastest, token-major
// flat: out[p*hidden + h]).
//
// - input_ids:    seq token ids.
// - audio_embeds: n_audio audio rows, token-major flat (audio_embeds[k*hidden+h]).
// - n_audio:      number of audio rows available (== count of audio_token_id
//                 positions in input_ids for a well-formed input).
// - hidden:       model hidden size.
// - audio_token_id: placeholder token id marking audio positions.
//
// The base embedding of token t is column t of token_embd.weight
// (ne=[hidden,vocab]) i.e. token_embd_data[t*hidden + h]. The k-th audio-token
// position (in increasing index) is overwritten with audio_embeds row k.
std::vector<float> fuse_embeds(ModelLoader& m,
                               const std::vector<int32_t>& input_ids,
                               const std::vector<float>& audio_embeds,
                               int n_audio, int hidden, int audio_token_id);

}  // namespace mt

#endif  // MT_GENERATE_HPP
