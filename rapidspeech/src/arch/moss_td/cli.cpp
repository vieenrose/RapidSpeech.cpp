#include "moss_transcribe.h"
#include "model_loader.hpp"
#include "transcribe.hpp"
#include "subtitle.hpp"
#include "audio_io.hpp"
#include "audio_encoder.hpp"
#include "audio_span.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "tokenizer.hpp"

#include <chrono>
#include <future>
#include <sys/resource.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int cmd_info(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-transcribe info <model.gguf>\n");
        return 2;
    }
    mt::ModelLoader m;
    if (!m.load(argv[2])) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    const auto& c = m.config();
    std::printf("arch=%s text.hidden=%d text.layers=%d audio.d_model=%d audio.layers=%d\n",
                c.arch.c_str(), c.text_hidden, c.text_layers, c.audio_d_model, c.audio_layers);
    std::printf("audio_token_id=%d merge=%d adaptor_in=%d mel_bins=%d n_fft=%d hop=%d\n",
                c.audio_token_id, c.audio_merge_size, c.adaptor_input_dim,
                c.feat_size, c.feat_n_fft, c.feat_hop);
    return 0;
}

static int cmd_transcribe(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: moss-transcribe transcribe <model.gguf> <audio.wav> "
            "[--max-new N] [--format text|srt|ass|json]\n");
        return 2;
    }
    const char* gguf = argv[2];
    const char* wav  = argv[3];
    int max_new = -1;  // resolved from config below if not overridden
    std::string format = "text";
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        }
    }
    if (format != "text" && format != "srt" && format != "ass" && format != "json") {
        std::fprintf(stderr, "unknown --format '%s' (want text|srt|ass|json)\n",
                     format.c_str());
        return 2;
    }
    mt::ModelLoader m;
    if (!m.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }
    m.promote_small_f16_to_f32();
    if (max_new <= 0) {
        max_new = m.config().default_max_new_tokens > 0
                      ? m.config().default_max_new_tokens : 5120;
    }
    std::string text = mt::transcribe_wav(m, wav, max_new);
    if (text.empty()) { std::fprintf(stderr, "transcription failed\n"); return 1; }

    if (format == "text") {
        // Byte-identical to the historical raw output.
        std::printf("%s\n", text.c_str());
    } else if (format == "json") {
        // Raw parsed segments (no timing normalization).
        auto segs = mt::subtitle_segments_from_transcript(text, /*postprocess=*/false);
        std::string out = mt::to_json(segs);
        std::fwrite(out.data(), 1, out.size(), stdout);
    } else {
        // srt / ass: normalize (merge/split/overlap) then export.
        auto segs = mt::subtitle_segments_from_transcript(text, /*postprocess=*/true);
        std::string out = (format == "srt") ? mt::to_srt(segs)
                                            : mt::to_ass(segs);
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// quantize: re-quantize the SAME allowlisted linear weights as the converter's
// f16/q8_0 path to a target ggml type -- including the K-quants (Q4_K/Q5_K/Q6_K)
// that the Python gguf writer can't produce. Every non-allowlisted tensor (and
// any allowlisted tensor that isn't shape-eligible) is copied verbatim in its
// stored type. All KV metadata is copied unchanged.
//
// Ported from parakeet.cpp's cmd_quantize; the only change is the allowlist,
// which mirrors the converter's _QUANTIZABLE_PATTERNS (token_embd + qwen3
// attn/ffn + whisper-enc attn/ffn + adaptor fc).
// ---------------------------------------------------------------------------

static bool is_quantizable_name(const std::string& n) {
    if (n == "token_embd.weight") return true;
    if (n == "adaptor.fc1.w" || n == "adaptor.fc2.w") return true;
    // qwen3.blk.<d>.<rest> / enc.blk.<d>.<rest>
    auto blk_rest = [](const std::string& s, const char* prefix, std::string& rest) -> bool {
        size_t p = std::strlen(prefix);
        if (s.rfind(prefix, 0) != 0) return false;
        size_t i = p, start = p;
        while (i < s.size() && std::isdigit((unsigned char)s[i])) ++i;
        if (i == start || i >= s.size() || s[i] != '.') return false;
        rest = s.substr(i + 1);
        return true;
    };
    std::string rest;
    if (blk_rest(n, "qwen3.blk.", rest)) {
        return rest == "attn_q.weight" || rest == "attn_k.weight" ||
               rest == "attn_v.weight" || rest == "attn_o.weight" ||
               rest == "ffn_gate.weight" || rest == "ffn_up.weight" || rest == "ffn_down.weight";
    }
    if (blk_rest(n, "enc.blk.", rest)) {
        return rest == "attn_q.w" || rest == "attn_k.w" ||
               rest == "attn_v.w" || rest == "attn_out.w" ||
               rest == "ffn_1.w" || rest == "ffn_2.w";
    }
    return false;
}

static bool parse_quant_type(const std::string& s, ggml_type& out) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (t == "iq2_xs") { out = GGML_TYPE_IQ2_XS; return true; }
    if (t == "iq2_s")  { out = GGML_TYPE_IQ2_S;  return true; }
    if (t == "q2_k") { out = GGML_TYPE_Q2_K; return true; }
    if (t == "q3_k") { out = GGML_TYPE_Q3_K; return true; }
    if (t == "q4_0") { out = GGML_TYPE_Q4_0; return true; }
    if (t == "q5_0") { out = GGML_TYPE_Q5_0; return true; }
    if (t == "q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (t == "q4_k") { out = GGML_TYPE_Q4_K; return true; }
    if (t == "q5_k") { out = GGML_TYPE_Q5_K; return true; }
    if (t == "q6_k") { out = GGML_TYPE_Q6_K; return true; }
    return false;
}

static bool is_k_quant(ggml_type t) {
    return t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

static int cmd_quantize(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: moss-transcribe quantize <in.gguf> <out.gguf> "
            "<iq2_xs|iq2_s|q2_k|q3_k|q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n");
        return 2;
    }
    const std::string in_path  = argv[0];
    const std::string out_path = argv[1];
    ggml_type qtype;
    if (!parse_quant_type(argv[2], qtype)) {
        std::fprintf(stderr,
            "moss-transcribe quantize: unknown type '%s' "
            "(want iq2_xs|iq2_s|q2_k|q3_k|q4_0|q5_0|q8_0|q4_k|q5_k|q6_k)\n", argv[2]);
        return 2;
    }

    // Load the source GGUF with a backing ggml_context so tensor data is read in.
    // Optional: --imatrix FILE (records: u32 name_len, name, u32 n, n floats
    // = per-input-channel importance) required by IQ2 types; --only SUBSTR
    // restricts quantization to tensors whose name contains SUBSTR (others
    // are copied verbatim).
    std::map<std::string, std::vector<float>> imatrix;
    std::vector<std::string> only;   // comma-separated substrings
    for (int ai = 3; ai < argc; ++ai) {
        if (!std::strcmp(argv[ai], "--imatrix") && ai + 1 < argc) {
            FILE* f = std::fopen(argv[++ai], "rb");
            if (!f) { std::fprintf(stderr, "cannot open imatrix\n"); return 1; }
            while (true) {
                uint32_t nl = 0;
                if (std::fread(&nl, 4, 1, f) != 1) break;
                std::string nm(nl, '\0');
                if (std::fread(nm.data(), 1, nl, f) != nl) break;
                uint32_t n = 0;
                if (std::fread(&n, 4, 1, f) != 1) break;
                std::vector<float> v(n);
                if (std::fread(v.data(), 4, n, f) != n) break;
                imatrix[nm] = std::move(v);
            }
            std::fclose(f);
            std::fprintf(stderr, "imatrix: %zu tensors\n", imatrix.size());
        } else if (!std::strcmp(argv[ai], "--only") && ai + 1 < argc) {
            std::string v = argv[++ai];
            size_t pos = 0;
            while (pos != std::string::npos) {
                size_t c = v.find(',', pos);
                only.push_back(v.substr(pos, c == std::string::npos ? c : c - pos));
                pos = c == std::string::npos ? c : c + 1;
            }
        }
    }

    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p{ /*no_alloc*/ false, /*ctx*/ &src_ctx };
    struct gguf_context* src = gguf_init_from_file(in_path.c_str(), p);
    if (!src) {
        std::fprintf(stderr, "failed to open %s\n", in_path.c_str());
        return 1;
    }

    const int64_t block = ggml_blck_size(qtype);  // 32 for q*_0, 256 for K-quants

    // Destination: empty gguf, copy all KV verbatim, then add tensors.
    struct gguf_context* dst = gguf_init_empty();
    gguf_set_kv(dst, src);  // copies every KV pair unchanged

    const int64_t nt = gguf_get_n_tensors(src);
    // Holds quantized buffers alive until gguf_write_to_file copies them out.
    std::vector<std::vector<uint8_t>> quant_bufs;
    quant_bufs.reserve(static_cast<size_t>(nt));
    struct ggml_init_params meta_p{ /*mem_size*/ ggml_tensor_overhead() * (nt + 1),
                                    /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context* meta = ggml_init(meta_p);

    int n_quant = 0, n_kept = 0, n_skipped = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(src, i);
        struct ggml_tensor* t = ggml_get_tensor(src_ctx, name);

        bool do_quant = false;
        bool only_ok = only.empty();
        for (const auto& o : only)
            if (std::string(name).find(o) != std::string::npos) { only_ok = true; break; }
        if (only_ok && is_quantizable_name(name) && t->type == GGML_TYPE_F32) {
            const bool two_d  = (t->ne[2] == 1 && t->ne[3] == 1 &&
                                 t->ne[0] >= 32 && t->ne[1] >= 32);
            const bool blk_ok = (t->ne[0] % block == 0);
            if (two_d && blk_ok) {
                do_quant = true;
            } else if (two_d && !blk_ok) {
                std::fprintf(stderr,
                    "  keep F32: %-48s ne0=%lld not divisible by %s block %lld\n",
                    name, (long long)t->ne[0],
                    is_k_quant(qtype) ? "K-quant superblock" : "block",
                    (long long)block);
                ++n_skipped;
            }
        }

        if (do_quant) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = t->ne[1];
            const size_t  out_bytes = ggml_row_size(qtype, n_per_row) * (size_t)nrows;
            quant_bufs.emplace_back(out_bytes);
            const float* fsrc = (const float*)t->data;
            const float* im = nullptr;
            auto it = imatrix.find(name);
            if (it != imatrix.end() && (int64_t)it->second.size() == n_per_row)
                im = it->second.data();
            ggml_quantize_chunk(qtype, fsrc, quant_bufs.back().data(),
                                /*start*/ 0, nrows, n_per_row, im);
            struct ggml_tensor* q = ggml_new_tensor_2d(meta, qtype, n_per_row, nrows);
            ggml_set_name(q, name);
            q->data = quant_bufs.back().data();
            gguf_add_tensor(dst, q);
            ++n_quant;
        } else {
            gguf_add_tensor(dst, t);
            ++n_kept;
        }
    }

    const bool ok = gguf_write_to_file(dst, out_path.c_str(), /*only_meta*/ false);

    std::printf("quantize: %s -> %s [%s]\n", in_path.c_str(), out_path.c_str(), argv[2]);
    std::printf("  quantized %d tensor(s), copied %d verbatim, %d allowlisted kept F32 (block).\n",
                n_quant, n_kept, n_skipped);

    ggml_quantize_free();
    ggml_free(meta);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(src_ctx);

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    return 0;
}


// transcribe-windowed: the deployed long-audio shape (fixed windows, fresh KV
// per window), with an optional PIPELINED mode: window N+1's audio encoder
// runs concurrently with window N's prefill+decode. Legal because windows are
// independent decodes; safe because backend.cpp keeps one backend+allocator
// per thread. Per-window output is printed so sequential vs pipelined runs
// can be diffed (they must be identical -- scheduling only, no numerics).
static int cmd_windowed(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: moss-transcribe transcribe-windowed "
                             "<model.gguf> <audio.wav> [--window S] "
                             "[--pipeline] [--max-new N] [--batch N]\n");
        return 2;
    }
    double window_s = 90.0;
    bool pipeline = false;
    int max_new = 0;
    int batch = 0;   // 0: resolve from MTD_BATCH below, default 1
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--window") && i + 1 < argc) window_s = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--pipeline")) pipeline = true;
        else if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
    }
    if (batch <= 0) {
        const char* be = std::getenv("MTD_BATCH");
        batch = be ? atoi(be) : 0;
    }
    if (batch <= 0) batch = 1;
    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count(); };

    mt::ModelLoader m;
    if (!m.load(argv[2])) { std::fprintf(stderr, "load failed\n"); return 1; }
    const mt::Config& c = m.config();
    const int hidden = c.text_hidden;
    if (max_new <= 0) max_new = 1200;

    mt::Audio audio;
    if (!mt::load_audio_16k_mono(argv[3], audio)) {
        std::fprintf(stderr, "audio load failed\n"); return 1;
    }
    const std::vector<float>& pcm = audio.samples;
    const int win_n = (int)(window_s * 16000.0);
    std::vector<std::pair<int,int>> wins;      // [begin, end) sample ranges
    for (int b = 0; b < (int)pcm.size(); b += win_n) {
        int e = std::min((int)pcm.size(), b + win_n);
        if (e - b >= 16000) wins.push_back({b, e});
    }
    std::fprintf(stderr, "windows: %zu x %.0fs, mode=%s, batch=%d\n",
                 wins.size(), window_s, pipeline ? "pipelined" : "sequential",
                 batch);

    struct Enc { std::vector<float> embeds; int n_tokens = 0; double secs = 0; };
    auto encode_win = [&](int wi) {
        auto t0 = Clock::now();
        Enc r;
        mt::AudioEncoder aenc(m);
        std::vector<float> seg(pcm.begin() + wins[wi].first,
                               pcm.begin() + wins[wi].second);
        r.embeds = aenc.encode(seg, r.n_tokens, hidden);
        r.secs = secs(t0, Clock::now());
        return r;
    };

    mt::Tokenizer tok;
    if (!tok.load(m)) { std::fprintf(stderr, "tokenizer load failed\n"); return 1; }

    // MT_KV_EVICT_S (opt-in): audio-KV eviction during decode, honored at
    // EVERY batch size so --batch 1 and --batch N stay comparable (batch=1
    // routes through greedy_generate_evict, batch>1 through the per-stream
    // eviction state of greedy_generate_batch).
    const char* evict_env = std::getenv("MT_KV_EVICT_S");
    const double evict_s = evict_env ? atof(evict_env) : 0.0;
    auto make_evict_opts = [&](const std::vector<int32_t>& ids,
                               double audio_secs) {
        mt::KvEvictOpts ev;
        if (evict_s <= 0.0) return ev;
        ev.window_s = evict_s;
        int lo = -1, hi = -1;
        for (int i = 0; i < (int)ids.size(); ++i)
            if (ids[i] == c.audio_token_id) { if (lo < 0) lo = i; hi = i; }
        if (lo >= 0 && hi > lo && audio_secs > 0.0) {
            ev.span_lo   = lo;
            ev.span_end  = hi + 1;
            ev.tok_per_s = (double)(ev.span_end - ev.span_lo) / audio_secs;
            ev.tok       = &tok;
        }
        return ev;
    };

    auto t_all = Clock::now();
    double enc_total = 0, dec_total = 0;
    std::future<Enc> ahead;
    if (pipeline) ahead = std::async(std::launch::async, encode_win, 0);
    for (int g0 = 0; g0 < (int)wins.size(); g0 += batch) {
        const int gn = std::min(batch, (int)wins.size() - g0);
        // Encode the group's windows (the --pipeline lookahead-1 machinery
        // composes unchanged: it runs over the global window index).
        std::vector<Enc> encs(gn);
        for (int j = 0; j < gn; ++j) {
            const int wi = g0 + j;
            if (pipeline) {
                encs[j] = ahead.get();
                if (wi + 1 < (int)wins.size())
                    ahead = std::async(std::launch::async, encode_win, wi + 1);
            } else {
                encs[j] = encode_win(wi);
            }
            enc_total += encs[j].secs;
            if (encs[j].embeds.empty() || encs[j].n_tokens <= 0) {
                std::fprintf(stderr, "window %d: encode failed\n", wi); return 1;
            }
        }
        auto t0 = Clock::now();
        std::vector<std::vector<int32_t>> ids(gn);
        std::vector<std::vector<float>> fused(gn);
        int max_seq_group = 0;
        for (int j = 0; j < gn; ++j) {
            ids[j] = mt::build_input_ids(tok, c, c.default_prompt,
                                         encs[j].n_tokens);
            fused[j] = mt::fuse_embeds(m, ids[j], encs[j].embeds,
                                       encs[j].n_tokens, hidden,
                                       c.audio_token_id);
            if (ids[j].empty() || fused[j].empty()) {
                std::fprintf(stderr, "window %d: input build failed\n", g0 + j);
                return 1;
            }
            max_seq_group = std::max(max_seq_group, (int)ids[j].size());
        }
        // One decoder per group: gn independent cache sets (gn==1 is the
        // exact sequential configuration).
        mt::Qwen3Decoder dec;
        if (!dec.load(m, max_seq_group + max_new + 16, gn)) {
            std::fprintf(stderr, "dec load failed\n"); return 1;
        }
        std::vector<std::vector<int32_t>> outs(gn);
        if (gn == 1) {
            const double asecs = (wins[g0].second - wins[g0].first) / 16000.0;
            outs[0] = (evict_s > 0.0)
                ? mt::greedy_generate_evict(dec, m, fused[0],
                                            (int)ids[0].size(), max_new,
                                            c.eos_token_id,
                                            make_evict_opts(ids[0], asecs))
                : mt::greedy_generate(dec, m, fused[0], (int)ids[0].size(),
                                      max_new, c.eos_token_id);
        } else {
            std::vector<mt::BatchStream> bs(gn);
            for (int j = 0; j < gn; ++j) {
                bs[j].fused = &fused[j];
                bs[j].seq   = (int)ids[j].size();
                const double asecs =
                    (wins[g0 + j].second - wins[g0 + j].first) / 16000.0;
                bs[j].ev = make_evict_opts(ids[j], asecs);
            }
            outs = mt::greedy_generate_batch(dec, m, bs, max_new,
                                             c.eos_token_id);
        }
        dec_total += secs(t0, Clock::now());
        // Print in WINDOW ORDER regardless of per-stream completion order.
        for (int j = 0; j < gn; ++j) {
            const int wi = g0 + j;
            std::printf("=== window %d [%.1fs..%.1fs] ===\n%s\n", wi,
                        wins[wi].first / 16000.0, wins[wi].second / 16000.0,
                        tok.decode(outs[j]).c_str());
            std::fflush(stdout);
        }
    }
    const double wall = secs(t_all, Clock::now());
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    std::fprintf(stderr,
                 "WINDOWED mode=%s batch=%d wall=%.1fs enc_sum=%.1fs dec_sum=%.1fs "
                 "audio=%.1fs rtf=%.3f peak_rss=%.0fMB\n",
                 pipeline ? "pipelined" : "sequential", batch, wall, enc_total,
                 dec_total, pcm.size() / 16000.0,
                 wall / (pcm.size() / 16000.0), ru.ru_maxrss / 1024.0);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: moss-transcribe <subcommand>\n"); return 2; }
    if (std::strcmp(argv[1], "version") == 0) { std::printf("%s\n", mt::version()); return 0; }
    if (std::strcmp(argv[1], "info") == 0) { return cmd_info(argc, argv); }
    if (std::strcmp(argv[1], "transcribe") == 0) { return cmd_transcribe(argc, argv); }
    if (std::strcmp(argv[1], "transcribe-windowed") == 0) { return cmd_windowed(argc, argv); }
    if (std::strcmp(argv[1], "quantize") == 0) { return cmd_quantize(argc - 2, argv + 2); }
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
