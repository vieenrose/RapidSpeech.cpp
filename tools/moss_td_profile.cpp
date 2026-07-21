// moss_td_profile.cpp -- per-stage latency tree for MOSS-TD CPU inference.
//
// A WRAPPER, not an engine change: this replicates transcribe_pcm16k()'s exact
// call sequence (see moss_td/transcribe.cpp) against the same unmodified
// vendored library, timing each stage. moss_td/ stays byte-for-byte upstream,
// which is the rule for that directory -- so profiling lives out here instead
// of as instrumentation inside it.
//
// Because it re-implements the sequence rather than calling transcribe_pcm16k,
// it prints its transcript so callers can diff against the real CLI's output
// and confirm the replication is faithful. If those ever diverge, the timings
// describe something that is not the shipped pipeline and must not be trusted.
//
// Usage: rs-moss-td-profile MODEL.gguf AUDIO.wav [--max-new N] [--repeat N]

#include "model_loader.hpp"
#include "audio_encoder.hpp"
#include "audio_io.hpp"
#include "tokenizer.hpp"
#include "audio_span.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "transcribe.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s MODEL.gguf AUDIO.wav [--max-new N] [--repeat N]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const char* wav_path = argv[2];
    int max_new = 0, repeat = 1;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }

    auto t_all = Clock::now();

    auto t0 = Clock::now();
    mt::ModelLoader m;
    if (!m.load(model_path)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const double t_model_load = ms_since(t0);

    const mt::Config& c = m.config();
    const int hidden = c.text_hidden;
    if (max_new <= 0) max_new = c.default_max_new_tokens > 0 ? c.default_max_new_tokens : 5120;

    t0 = Clock::now();
    mt::Audio audio;
    if (!mt::load_audio_16k_mono(wav_path, audio)) {
        std::fprintf(stderr, "audio load failed\n"); return 1;
    }
    const std::vector<float>& samples = audio.samples;
    const double t_audio_io = ms_since(t0);
    const double audio_seconds = samples.size() / 16000.0;

    double t_encode = 0, t_tok_load = 0, t_input_ids = 0, t_fuse = 0,
           t_dec_load = 0, t_generate = 0, t_detok = 0;
    std::string text;
    int n_tokens = 0, seq = 0, n_generated = 0;

    for (int r = 0; r < repeat; r++) {
        t0 = Clock::now();
        mt::AudioEncoder aenc(m);
        n_tokens = 0;
        std::vector<float> audio_embeds = aenc.encode(samples, n_tokens, hidden);
        t_encode = ms_since(t0);
        if (audio_embeds.empty() || n_tokens <= 0) { std::fprintf(stderr, "encode failed\n"); return 1; }

        t0 = Clock::now();
        mt::Tokenizer tok;
        if (!tok.load(m)) { std::fprintf(stderr, "tokenizer load failed\n"); return 1; }
        t_tok_load = ms_since(t0);

        t0 = Clock::now();
        std::vector<int32_t> input_ids = mt::build_input_ids(tok, c, c.default_prompt, n_tokens);
        t_input_ids = ms_since(t0);
        if (input_ids.empty()) { std::fprintf(stderr, "build_input_ids empty\n"); return 1; }

        t0 = Clock::now();
        std::vector<float> fused =
            mt::fuse_embeds(m, input_ids, audio_embeds, n_tokens, hidden, c.audio_token_id);
        t_fuse = ms_since(t0);
        if (fused.empty()) { std::fprintf(stderr, "fuse failed\n"); return 1; }

        seq = (int)input_ids.size();
        t0 = Clock::now();
        mt::Qwen3Decoder dec;
        if (!dec.load(m, seq + max_new + 16)) { std::fprintf(stderr, "decoder load failed\n"); return 1; }
        t_dec_load = ms_since(t0);

        t0 = Clock::now();
        std::vector<int32_t> new_ids =
            mt::greedy_generate(dec, m, fused, seq, max_new, c.eos_token_id);
        t_generate = ms_since(t0);
        n_generated = (int)new_ids.size();
        if (new_ids.empty()) { std::fprintf(stderr, "no tokens generated\n"); return 1; }

        t0 = Clock::now();
        text = tok.decode(new_ids);
        t_detok = ms_since(t0);
    }

    const double t_total = ms_since(t_all);
    // Everything except the one-time model load -- i.e. what a warm, resident
    // server actually pays per request.
    const double t_warm = t_audio_io + t_encode + t_tok_load + t_input_ids +
                          t_fuse + t_dec_load + t_generate + t_detok;

    auto row = [&](const char* name, double ms) {
        std::printf("  %-22s %9.1f ms  %5.1f%% of warm  %5.1f%% of total\n",
                    name, ms, 100.0 * ms / t_warm, 100.0 * ms / t_total);
    };

    std::printf("\n=== MOSS-TD CPU latency tree ===\n");
    std::printf("model      : %s\n", model_path);
    std::printf("audio      : %s (%.1f s, %d audio tokens, prompt seq %d)\n",
                wav_path, audio_seconds, n_tokens, seq);
    std::printf("generated  : %d tokens (cap %d)\n\n", n_generated, max_new);

    std::printf("ONE-TIME (amortised by a resident server):\n");
    row("model load (mmap)", t_model_load);
    std::printf("\nPER-REQUEST (warm path):\n");
    row("audio io + resample", t_audio_io);
    row("audio encoder", t_encode);
    row("tokenizer load", t_tok_load);
    row("build_input_ids", t_input_ids);
    row("fuse_embeds", t_fuse);
    row("decoder alloc/load", t_dec_load);
    row("greedy_generate", t_generate);
    row("detokenize", t_detok);
    std::printf("  %-22s %9.1f ms\n", "-- warm subtotal", t_warm);
    std::printf("  %-22s %9.1f ms\n", "-- total incl. load", t_total);

    if (n_generated > 0 && t_generate > 0)
        std::printf("\ndecode throughput   : %.2f tok/s (%.1f ms/token)\n",
                    1000.0 * n_generated / t_generate, t_generate / n_generated);
    if (audio_seconds > 0) {
        std::printf("RTF (warm)          : %.3f\n", (t_warm / 1000.0) / audio_seconds);
        std::printf("RTF (incl. load)    : %.3f\n", (t_total / 1000.0) / audio_seconds);
    }

    std::printf("\n--- transcript (diff vs `rs-moss-td transcribe` to verify "
                "this wrapper replicates the pipeline) ---\n%s\n", text.c_str());
    return 0;
}
