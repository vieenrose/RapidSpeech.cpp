// moss_td_stream.cpp -- streaming C API for MOSS-TD live transcripts.
//
// A WRAPPER, not an engine change (same rule as moss_td_profile.cpp): the
// vendored moss_td/ tree stays byte-for-byte upstream. This file replicates
// transcribe_pcm16k()'s exact call sequence against the same unmodified
// library, but drives the greedy loop itself so it can hand the accumulated
// partial transcript to a caller-supplied callback after every decoded token,
// plus coarse phase events ("encode", "prefill", "decode") so a UI can show
// liveness through the long silent stages.
//
// Faithfulness contract: with a NULL callback the call DELEGATES to the
// vendored mt::transcribe_pcm16k -- the exact byte-validated path. With a
// callback, the replicated loop must produce the same final text; this is
// gated by diffing both paths on the golden clips before any deploy (same
// methodology as the profiler tool).
//
// The callback receives the FULL partial transcript each time (not a token
// fragment): UTF-8 splits, marker fragments and retokenization are handled
// here, so consumers just replace their provisional text wholesale.

#include "model_loader.hpp"
#include "audio_encoder.hpp"
#include "tokenizer.hpp"
#include "audio_span.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "transcribe.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

// kind: 0 = token (text = full partial transcript so far, cur = tokens emitted)
//       1 = phase (text = "encode" | "prefill" | "decode", cur/total context-dependent)
typedef void (*mtd_stream_event_cb)(int kind, const char* text, int cur,
                                    int total, void* user);

struct mtd_stream_ctx {
    mt::ModelLoader loader;
    std::string last;   // owns the last returned transcript
};

mtd_stream_ctx* mtd_stream_load(const char* gguf_path) {
    auto* ctx = new (std::nothrow) mtd_stream_ctx();
    if (!ctx) return nullptr;
    if (!gguf_path || !ctx->loader.load(gguf_path)) { delete ctx; return nullptr; }
    return ctx;
}

void mtd_stream_free(mtd_stream_ctx* ctx) { delete ctx; }

static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (unsigned char)s[b] <= ' ') ++b;
    while (e > b && (unsigned char)s[e - 1] <= ' ') --e;
    return s.substr(b, e - b);
}

// pcm: 16 kHz mono float PCM. Returns the final transcript (valid until the
// next call on this ctx; do not free). NULL on failure.
const char* mtd_stream_transcribe_pcm(mtd_stream_ctx* ctx, const float* pcm,
                                      int n_samples, int max_new,
                                      mtd_stream_event_cb cb, void* user) {
    if (!ctx || !pcm || n_samples <= 0) return nullptr;
    std::vector<float> samples(pcm, pcm + n_samples);
    mt::ModelLoader& m = ctx->loader;
    const mt::Config& c = m.config();
    if (max_new <= 0)
        max_new = c.default_max_new_tokens > 0 ? c.default_max_new_tokens : 5120;

    if (!cb) {  // exact vendored path, no replication involved
        ctx->last = mt::transcribe_pcm16k(m, samples, max_new);
        return ctx->last.c_str();
    }

    const int hidden = c.text_hidden;
    cb(1, "encode", 0, 0, user);
    mt::AudioEncoder aenc(m);
    int n_tokens = 0;
    std::vector<float> audio_embeds = aenc.encode(samples, n_tokens, hidden);
    if (audio_embeds.empty() || n_tokens <= 0) return nullptr;

    mt::Tokenizer tok;
    if (!tok.load(m)) return nullptr;
    std::vector<int32_t> input_ids =
        mt::build_input_ids(tok, c, c.default_prompt, n_tokens);
    if (input_ids.empty()) return nullptr;
    std::vector<float> fused =
        mt::fuse_embeds(m, input_ids, audio_embeds, n_tokens, hidden,
                        c.audio_token_id);
    if (fused.empty()) return nullptr;

    const int seq = (int)input_ids.size();
    mt::Qwen3Decoder dec;
    if (!dec.load(m, seq + max_new + 16)) return nullptr;

    cb(1, "prefill", seq, 0, user);
    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid) || (int)hid.size() < hidden * seq)
        return nullptr;
    std::vector<float> last(hid.end() - hidden, hid.end());
    std::vector<float> logits = dec.logits_from_hidden(last);
    if (logits.empty()) return nullptr;

    // MT_KV_EVICT_S: mirror the vendored transcribe.cpp opt-in exactly, so the
    // streaming and blocking paths stay in agreement whether or not eviction
    // is enabled (gated by byte-comparing both on the golden clips).
    double evict_w = 0.0, evict_tps = 0.0;
    int span_lo = -1, span_end = -1;
    if (const char* e = std::getenv("MT_KV_EVICT_S"); e && atof(e) > 0.0) {
        int lo = -1, hi = -1;
        for (int i = 0; i < seq; ++i)
            if (input_ids[i] == c.audio_token_id) { if (lo < 0) lo = i; hi = i; }
        const double audio_s = (double)samples.size() / 16000.0;
        if (lo >= 0 && hi > lo && audio_s > 0.0) {
            evict_w = atof(e);
            span_lo = lo; span_end = hi + 1;
            evict_tps = (double)(span_end - span_lo) / audio_s;
        }
    }
    std::string ev_tail;
    double max_ts = 0.0;
    int span_end_phys = span_end, evicted_total = 0;
    auto harvest_ts = [&]() {
        size_t close;
        while ((close = ev_tail.find(']')) != std::string::npos) {
            const size_t open = ev_tail.rfind('[', close);
            if (open != std::string::npos && close > open + 1) {
                bool num = true; int digits = 0;
                for (size_t i = open + 1; i < close; ++i) {
                    const char ch = ev_tail[i];
                    if (ch >= '0' && ch <= '9') { ++digits; }
                    else if (ch != '.') { num = false; break; }
                }
                if (num && digits > 0) {
                    const double ts = atof(ev_tail.substr(open + 1, close - open - 1).c_str());
                    if (ts > max_ts) max_ts = ts;
                }
            }
            ev_tail.erase(0, close + 1);
        }
        if (ev_tail.size() > 32) ev_tail.erase(0, ev_tail.size() - 32);
    };

    cb(1, "decode", 0, max_new, user);
    // Greedy loop -- replicates mt::greedy_generate (argmax FIRST-index-on-tie)
    // with a per-token partial-detokenize + callback.
    std::vector<int32_t> new_ids;
    new_ids.reserve((size_t)max_new);
    for (;;) {
        int best = 0;
        for (int i = 1; i < (int)logits.size(); ++i)
            if (logits[i] > logits[best]) best = i;
        new_ids.push_back(best);
        if (best == c.eos_token_id) break;
        if ((int)new_ids.size() >= max_new) break;

        // Full re-decode keeps UTF-8 and marker fragments correct; cost is
        // trivial next to a decode step.
        const std::string partial = tok.decode(new_ids);
        cb(0, partial.c_str(), (int)new_ids.size(), max_new, user);

        if (evict_w > 0.0) {
            ev_tail += tok.decode({new_ids.back()});
            harvest_ts();
            if (max_ts > evict_w) {
                int keep_from = span_lo +
                    (int)((max_ts - evict_w) * evict_tps) - evicted_total;
                if (keep_from > span_end_phys) keep_from = span_end_phys;
                if (keep_from - span_lo >= 256) {
                    const int delta = dec.evict(span_lo, keep_from);
                    evicted_total += delta;
                    span_end_phys -= delta;
                }
            }
        }

        std::vector<float> emb = mt::embed_token(m, best, hidden);
        if (emb.empty()) break;
        std::vector<float> h1 = dec.decode_one(emb);
        if ((int)h1.size() < hidden) break;
        logits = dec.logits_from_hidden(h1);
        if (logits.empty()) break;
    }
    if (new_ids.empty()) return nullptr;
    ctx->last = trim(tok.decode(new_ids));
    return ctx->last.c_str();
}

}  // extern "C"
