/**
 * rs-tts-split — Sentence-level TTS synthesis with auto text splitting.
 *
 * Splits input text into sentences, synthesizes each independently, and
 * concatenates the results.  Short sentences produce fewer acoustic frames
 * so each diffusion run is faster and more stable than one long utterance.
 *
 * Usage:
 *   rs-tts-split -m <model.gguf> -t "text" [same options as rs-tts-offline]
 */

#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define LOG_INFO(fmt, ...)  std::printf("[tts-split] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[tts-split] ERROR: " fmt "\n", ##__VA_ARGS__)

struct SplitArgs {
    const char *model_path = nullptr;
    const char *text = nullptr;
    const char *output_path = "output.wav";
    const char *instruct = "male";
    const char *language = "English";
    const char *ref_path = nullptr;
    const char *ref_text = nullptr;
    int seed = 42;
    int n_steps = 32;
    int n_threads = 4;
    bool use_gpu = true;
    float silence_gap_s = 0.15f;  // silence between sentences
};

static bool parse_bool(const std::string &v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static void print_usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -m <model.gguf> -t \"text\" [options]\n\n"
        << "Options:\n"
        << "  -m, --model <path>       TTS model path (required)\n"
        << "  -t, --text <text>        Text to synthesize (required)\n"
        << "  -o, --output <path>      Output WAV file path (default: output.wav)\n"
        << "      --ref <path>         Reference audio for voice cloning (WAV)\n"
        << "      --ref-text <text>    Transcript of the reference audio\n"
        << "      --instruct <text>    Voice description (default: male)\n"
        << "      --lang <lang>        Target language (default: English)\n"
        << "      --seed <n>           Random seed (default: 42)\n"
        << "      --n-steps <n>        Diffusion steps 1-128 (default: 32)\n"
        << "      --threads <n>        CPU threads (default: 4)\n"
        << "      --gpu <true|false>   Enable GPU acceleration (default: true)\n"
        << "      --gap <seconds>      Silence between sentences (default: 0.15)\n"
        << "  -h, --help               Show this help\n"
        << std::endl;
}

static bool parse_args(int argc, char **argv, SplitArgs &args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((a == "-t" || a == "--text") && i + 1 < argc) {
            args.text = argv[++i];
        } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (a == "--ref" && i + 1 < argc) {
            args.ref_path = argv[++i];
        } else if (a == "--ref-text" && i + 1 < argc) {
            args.ref_text = argv[++i];
        } else if (a == "--instruct" && i + 1 < argc) {
            args.instruct = argv[++i];
        } else if (a == "--lang" && i + 1 < argc) {
            args.language = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if (a == "--n-steps" && i + 1 < argc) {
            args.n_steps = std::stoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            args.n_threads = std::stoi(argv[++i]);
        } else if (a == "--gpu" && i + 1 < argc) {
            args.use_gpu = parse_bool(argv[++i]);
        } else if (a == "--gap" && i + 1 < argc) {
            args.silence_gap_s = std::stof(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    if (!args.model_path) { std::cerr << "Error: --model is required\n"; return false; }
    if (!args.text) { std::cerr << "Error: --text is required\n"; return false; }
    return true;
}

static bool write_wav_file(const char *filename, const std::vector<float> &pcm,
                           int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    int num_samples = (int)pcm.size();
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = num_samples * block_align;
    int chunk_size = 36 + data_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char *>(&chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int subchunk1_size = 16;
    short audio_format = 1;
    file.write(reinterpret_cast<const char *>(&subchunk1_size), 4);
    file.write(reinterpret_cast<const char *>(&audio_format), 2);
    file.write(reinterpret_cast<const char *>(&num_channels), 2);
    file.write(reinterpret_cast<const char *>(&sample_rate), 4);
    file.write(reinterpret_cast<const char *>(&byte_rate), 4);
    file.write(reinterpret_cast<const char *>(&block_align), 2);
    file.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char *>(&data_size), 4);

    for (float s : pcm) {
        float clipped = std::max(-1.0f, std::min(1.0f, s));
        int16_t sample = static_cast<int16_t>(clipped * 32767.0f);
        file.write(reinterpret_cast<const char *>(&sample), sizeof(int16_t));
    }
    return true;
}

// Split text into sentences.
static std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> result;
    std::string current;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];

        // Detect UTF-8 Chinese punctuation: 。！？
        bool is_zh_punct = false;
        if (c >= 0xE0 && i + 2 < text.size()) {
            // UTF-8 3-byte: E3 80 82 = 。, EF BC 81 = ！, EF BC 9F = ？
            unsigned char b1 = (unsigned char)text[i+1];
            unsigned char b2 = (unsigned char)text[i+2];
            if ((c == 0xE3 && b1 == 0x80 && b2 == 0x82) ||  // 。
                (c == 0xEF && b1 == 0xBC && (b2 == 0x81 || b2 == 0x9F))) {  // ！？
                is_zh_punct = true;
            }
        }

        if (is_zh_punct) {
            current += text.substr(i, 3);
            i += 3;
        } else {
            current += text[i];
            i += 1;
        }

        // Sentence boundary characters
        bool is_boundary = (c == '.' || c == '!' || c == '?' || c == '\n' || is_zh_punct);

        if (is_boundary) {
            // Trim whitespace
            while (!current.empty() && (current.front() == ' ' ||
                   current.front() == '\n' || current.front() == '\r'))
                current.erase(0, 1);
            while (!current.empty() && (current.back() == ' ' ||
                   current.back() == '\n' || current.back() == '\r'))
                current.pop_back();
            if (!current.empty()) result.push_back(current);
            current.clear();
        }
    }
    // Last segment
    while (!current.empty() && (current.front() == ' ' ||
           current.front() == '\n' || current.front() == '\r'))
        current.erase(0, 1);
    while (!current.empty() && (current.back() == ' ' ||
           current.back() == '\n' || current.back() == '\r'))
        current.pop_back();
    if (!current.empty()) result.push_back(current);

    return result;
}

// Synthesize a single sentence.  Returns empty vector on failure.
static std::vector<float> synthesize_one(rs_context_t *ctx, const char *text,
                                         const SplitArgs &args, int sentence_idx) {
    if (rs_push_text(ctx, text) != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("sent %d: push_text failed: %s", sentence_idx, err.message);
        return {};
    }

    std::vector<float> pcm;
    int ret;
    while ((ret = rs_process(ctx)) >= 0) {
        float *chunk = nullptr;
        int n = rs_get_audio_output(ctx, &chunk);
        if (n > 0 && chunk) pcm.insert(pcm.end(), chunk, chunk + n);
        if (ret == 0) break;
    }
    if (ret < 0) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("sent %d: process failed: %s", sentence_idx, err.message);
        return {};
    }
    if (pcm.empty()) {
        LOG_ERROR("sent %d: no audio produced", sentence_idx);
    }
    return pcm;
}

// Push voice cloning reference (if provided).
static bool push_reference(rs_context_t *ctx, const SplitArgs &args) {
    if (!args.ref_path || !args.ref_path[0]) return true;

    std::vector<float> ref_pcm;
    int ref_sr = 0;
    if (!load_wav_file(args.ref_path, ref_pcm, &ref_sr)) {
        LOG_ERROR("Failed to load reference WAV: %s", args.ref_path);
        return false;
    }
    if (rs_push_reference_audio(ctx, ref_pcm.data(), (int32_t)ref_pcm.size(),
                                ref_sr) != 0) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("rs_push_reference_audio failed: %s", err.message);
        return false;
    }
    if (args.ref_text && args.ref_text[0]) {
        if (rs_push_reference_text(ctx, args.ref_text) != RS_OK) {
            rs_error_info_t err = rs_get_last_error();
            LOG_ERROR("rs_push_reference_text failed: %s", err.message);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    SplitArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    // Split text
    auto sentences = split_sentences(args.text);
    if (sentences.empty()) {
        LOG_ERROR("No sentences found in input text");
        return 1;
    }
    LOG_INFO("Split into %zu sentences", sentences.size());

    // Init TTS model once
    rs_init_params_t params = rs_default_params();
    params.model_path = args.model_path;
    params.n_threads = args.n_threads;
    params.use_gpu = args.use_gpu;
    params.task_type = RS_TASK_TTS_ONLINE;

    rs_context_t *ctx = rs_init_from_file(params);
    if (!ctx) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("TTS init failed: %s", err.message);
        return 1;
    }

    rs_model_meta_t meta = rs_get_model_meta(ctx);
    LOG_INFO("Arch: %s  SampleRate: %d  Backend: %s",
             meta.arch_name, meta.audio_sample_rate, rs_get_backend_name(ctx));

    rs_set_tts_params(ctx, args.instruct, args.language, args.seed);
    rs_set_tts_diffusion_steps(ctx, args.n_steps);
    LOG_INFO("Instruct: %s  Lang: %s  Steps: %d  Seed: %d",
             args.instruct, args.language, args.n_steps, args.seed);

    // Pre-load reference audio to memory (avoid re-reading file per sentence)
    std::vector<float> ref_pcm;
    int ref_sr = 0;
    if (args.ref_path && args.ref_path[0]) {
        if (!load_wav_file(args.ref_path, ref_pcm, &ref_sr)) {
            LOG_ERROR("Failed to load reference WAV: %s", args.ref_path);
            rs_free(ctx);
            return 1;
        }
        LOG_INFO("Reference: %zu samples @ %d Hz", ref_pcm.size(), ref_sr);
    }

    // Silence gap samples
    int gap_samples = (int)(args.silence_gap_s * meta.audio_sample_rate);

    // Synthesize each sentence
    std::vector<float> all_pcm;
    auto t0 = std::chrono::steady_clock::now();
    int n_ok = 0;

    for (size_t i = 0; i < sentences.size(); ++i) {
        LOG_INFO("[%zu/%zu] \"%s\"", i + 1, sentences.size(), sentences[i].c_str());

        // Reset state for clean sentence
        rs_reset(ctx);

        // Push voice reference (re-encoded each sentence; HuBERT encode is fast)
        if (!ref_pcm.empty()) {
            if (rs_push_reference_audio(ctx, ref_pcm.data(), (int32_t)ref_pcm.size(),
                                        ref_sr) != 0) {
                LOG_ERROR("sent %zu: push_ref_audio failed", i + 1);
                continue;
            }
            if (args.ref_text && args.ref_text[0]) {
                rs_push_reference_text(ctx, args.ref_text);
            }
        }

        auto sent_pcm = synthesize_one(ctx, sentences[i].c_str(), args, (int)i + 1);
        if (sent_pcm.empty()) continue;

        // Append sentence audio + silence gap
        all_pcm.insert(all_pcm.end(), sent_pcm.begin(), sent_pcm.end());
        if (gap_samples > 0 && i + 1 < sentences.size()) {
            all_pcm.resize(all_pcm.size() + gap_samples, 0.0f);
        }
        n_ok++;
    }

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0f;

    if (all_pcm.empty()) {
        LOG_ERROR("No audio produced from any sentence");
        rs_free(ctx);
        return 1;
    }

    float audio_dur = (float)all_pcm.size() / meta.audio_sample_rate;
    float rtf = audio_dur > 0 ? elapsed_s / audio_dur : 0.f;

    LOG_INFO("Done: %d/%zu sentences, %zu samples (%.2f s), %.1f s, RTF %.3f",
             n_ok, sentences.size(), all_pcm.size(), audio_dur, elapsed_s, rtf);

    if (!write_wav_file(args.output_path, all_pcm, meta.audio_sample_rate)) {
        LOG_ERROR("Failed to write WAV: %s", args.output_path);
        rs_free(ctx);
        return 1;
    }
    LOG_INFO("Wrote: %s", args.output_path);

    rs_free(ctx);
    rs_clear_error();
    return 0;
}
