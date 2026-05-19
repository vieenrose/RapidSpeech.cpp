/**
 * rs-imatrix — Importance Matrix collection tool for activation-aware quantization.
 *
 * Runs TTS calibration sentences through the OmniVoice model and collects
 * activation statistics (sum of squared activations per weight column) for
 * use as importance weights during quantization (AWQ technique).
 *
 * Usage:
 *   rs-imatrix -m <model.gguf> -o <output.dat> [options]
 *
 * Options:
 *   -m, --model  <path>   Input GGUF model (required)
 *   -o, --output <path>   Output importance matrix .dat file (required)
 *       --n-steps <n>     Diffusion steps for calibration (default: 8)
 *       --threads <n>     CPU threads (default: 4)
 *       --gpu             Enable GPU acceleration (default: off)
 *   -h, --help            Show help
 */

#include "imatrix_collector.h"
#include "rapidspeech.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#define LOG_INFO(fmt, ...)  std::printf("[imatrix] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[imatrix] ERROR: " fmt "\n", ##__VA_ARGS__)

struct ImatrixArgs {
    const char *model_path = nullptr;
    const char *output_path = nullptr;
    int n_steps = 8;
    int n_threads = 4;
    bool use_gpu = false;
};

// Calibration sentences covering diverse phonemes and patterns.
static const char *kCalibrationTexts[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis technology has advanced rapidly.",
    "Hello world, this is a test of the text to speech system.",
    "Machine learning models can generate natural sounding speech.",
    "The weather today is sunny with a chance of rain.",
    "Please say something so we can verify the audio quality.",
    "Artificial intelligence is transforming how we interact with computers.",
    "Once upon a time there was a beautiful princess.",
    "One two three four five six seven eight nine ten.",
    "The conference on natural language processing starts tomorrow.",
    "Good morning everyone, thank you for attending.",
    "Music and art are essential parts of human culture.",
    "Scientific research requires careful observation and measurement.",
    "The package should arrive by the end of the week.",
    "Reading books helps improve vocabulary and comprehension.",
};

static void print_usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -m <model.gguf> -o <output.dat> [options]\n\n"
        << "Options:\n"
        << "  -m, --model  <path>   Input GGUF model (required)\n"
        << "  -o, --output <path>   Output importance matrix .dat file (required)\n"
        << "      --n-steps <n>     Diffusion steps for calibration (default: 8)\n"
        << "      --threads <n>     CPU threads (default: 4)\n"
        << "      --gpu             Enable GPU acceleration (default: off)\n"
        << "  -h, --help            Show this help\n"
        << std::endl;
}

static bool parse_args(int argc, char **argv, ImatrixArgs &args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (a == "--n-steps" && i + 1 < argc) {
            args.n_steps = std::stoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            args.n_threads = std::stoi(argv[++i]);
        } else if (a == "--gpu") {
            args.use_gpu = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    if (!args.model_path) { std::cerr << "Error: --model is required\n"; return false; }
    if (!args.output_path) { std::cerr << "Error: --output is required\n"; return false; }
    return true;
}

// Static collector and callback for C API
static IMatrixCollector *g_collector = nullptr;

static void imatrix_callback(void *userdata, struct ggml_cgraph *gf) {
    (void)userdata;
    if (g_collector) {
        g_collector->collect_from_graph(gf);
    }
}

int main(int argc, char **argv) {
    ImatrixArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    LOG_INFO("Importance Matrix Collection Tool");
    LOG_INFO("Model: %s", args.model_path);
    LOG_INFO("Output: %s", args.output_path);
    LOG_INFO("Threads: %d  Steps: %d  GPU: %s",
             args.n_threads, args.n_steps, args.use_gpu ? "ON" : "OFF");

    // ---- Init via C API ----
    rs_init_params_t params = rs_default_params();
    params.model_path = args.model_path;
    params.n_threads = args.n_threads;
    params.use_gpu = args.use_gpu;
    params.task_type = RS_TASK_TTS_ONLINE;

    rs_context_t *ctx = rs_init_from_file(params);
    if (!ctx) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("Init failed: %s", err.message);
        return 1;
    }

    rs_model_meta_t meta = rs_get_model_meta(ctx);
    LOG_INFO("Arch: %s  SampleRate: %d  Backend: %s",
             meta.arch_name, meta.audio_sample_rate, rs_get_backend_name(ctx));

    // ---- Setup collector and hook ----
    IMatrixCollector collector;
    g_collector = &collector;

    rs_set_imatrix_callback(ctx, imatrix_callback, nullptr);
    rs_set_tts_params(ctx, "male", "English", 42);
    rs_set_tts_diffusion_steps(ctx, args.n_steps);
    LOG_INFO("IMatrix collector hooked (%d diffusion steps)", args.n_steps);

    // ---- Run calibration ----
    int n_texts = sizeof(kCalibrationTexts) / sizeof(kCalibrationTexts[0]);
    LOG_INFO("Running %d calibration texts...", n_texts);

    auto t_start = std::chrono::steady_clock::now();
    int n_success = 0;

    for (int i = 0; i < n_texts; i++) {
        LOG_INFO("[%2d/%2d] \"%s\"", i + 1, n_texts, kCalibrationTexts[i]);

        rs_reset(ctx);

        if (rs_push_text(ctx, kCalibrationTexts[i]) != RS_OK) {
            LOG_ERROR("PushText failed for text %d", i);
            continue;
        }

        bool ok = true;
        while (true) {
            int ret = rs_process(ctx);
            if (ret < 0) { ok = false; break; }
            if (ret == 0) break;
            float *chunk = nullptr;
            while (rs_get_audio_output(ctx, &chunk) > 0) {}
        }
        if (ok) n_success++;
    }

    auto t_end = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0f;

    LOG_INFO("Calibration done: %d/%d texts succeeded in %.1f s", n_success, n_texts, elapsed);

    if (n_success == 0) {
        LOG_ERROR("No calibration texts succeeded - nothing to save");
        rs_free(ctx);
        return 1;
    }

    collector.save(args.output_path);
    LOG_INFO("Importance matrix saved to %s", args.output_path);

    rs_set_imatrix_callback(ctx, nullptr, nullptr);
    rs_free(ctx);
    return 0;
}
