// rs-speaker-id.cpp — CAMPPlus-based speaker enrollment + cosine query.
//
// Usage:
//   rs-speaker-id --model campplus.gguf
//                 --enroll alice=alice.wav --enroll bob=bob.wav
//                 --query unknown.wav
//                 [--threads N] [--cpu] [--dump-emb FILE]
//
// Each `--enroll NAME=PATH` registers one reference; the WAV is read,
// resampled to 16 kHz mono if needed, then embedded. The query WAV is
// embedded the same way and the cosine similarity against every enrolled
// embedding is printed; the best match is highlighted.
//
// `--cpu` forces the CPU backend (default: GPU when available).
// `--dump-emb FILE` writes the raw query embedding (192 × float32, native
//                   endianness) to FILE — used by the numpy diff harness.

#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --enroll NAME=WAV [--enroll NAME=WAV ...] "
        "--query WAV [--threads N] [--cpu]\n", prog);
}

bool load_pcm_16k(const std::string& path, std::vector<float>& pcm) {
    int src_sr = 0;
    if (!load_wav_file_resampled(path.c_str(), pcm, 16000, &src_sr)) {
        std::fprintf(stderr, "Failed to load WAV '%s'\n", path.c_str());
        return false;
    }
    if (pcm.empty()) {
        std::fprintf(stderr, "WAV '%s' decoded to 0 samples\n", path.c_str());
        return false;
    }
    if (src_sr != 16000) {
        std::fprintf(stdout, "  '%s' resampled %d → 16000 Hz\n",
                     path.c_str(), src_sr);
    }
    return true;
}

struct EnrollEntry {
    std::string name;
    std::string path;
    std::vector<float> embedding;
};

bool parse_enroll_arg(const std::string& arg, EnrollEntry& out) {
    auto eq = arg.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= arg.size()) {
        std::fprintf(stderr, "Bad --enroll '%s' (expected NAME=PATH)\n",
                     arg.c_str());
        return false;
    }
    out.name = arg.substr(0, eq);
    out.path = arg.substr(eq + 1);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string query_path;
    std::string dump_path;
    std::vector<EnrollEntry> enrolls;
    int n_threads = 4;
    bool use_gpu = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") {
            model_path = need_arg("--model");
        } else if (a == "--enroll" || a == "-e") {
            EnrollEntry ee;
            if (!parse_enroll_arg(need_arg("--enroll"), ee)) return 2;
            enrolls.push_back(std::move(ee));
        } else if (a == "--query" || a == "-q") {
            query_path = need_arg("--query");
        } else if (a == "--threads" || a == "-t") {
            n_threads = std::atoi(need_arg("--threads"));
            if (n_threads <= 0) n_threads = 4;
        } else if (a == "--cpu") {
            use_gpu = false;
        } else if (a == "--dump-emb") {
            dump_path = need_arg("--dump-emb");
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument '%s'\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || query_path.empty() || enrolls.empty()) {
        usage(argv[0]);
        return 2;
    }

    rs_speaker_t* sp = rs_speaker_init_from_file(model_path.c_str(),
                                                 n_threads, use_gpu);
    if (!sp) {
        rs_error_info_t e = rs_get_last_error();
        std::fprintf(stderr, "rs_speaker_init_from_file failed: %s\n",
                     e.message);
        return 1;
    }

    const int dim = rs_speaker_dim(sp);
    std::fprintf(stdout, "CAMPPlus loaded: dim=%d sr=%d threads=%d gpu=%d\n",
                 dim, rs_speaker_sample_rate(sp), n_threads, (int)use_gpu);

    auto embed_wav = [&](const std::string& path,
                         std::vector<float>& out) -> bool {
        std::vector<float> pcm;
        if (!load_pcm_16k(path, pcm)) return false;
        out.assign((size_t)dim, 0.0f);
        rs_error_t rc = rs_speaker_embed(sp, pcm.data(), (int)pcm.size(),
                                         out.data(), (int)out.size());
        if (rc != RS_OK) {
            rs_error_info_t e = rs_get_last_error();
            std::fprintf(stderr, "embed failed for '%s': %s\n",
                         path.c_str(), e.message);
            return false;
        }
        return true;
    };

    // Enroll
    std::fprintf(stdout, "\nEnrolling %d reference(s):\n", (int)enrolls.size());
    for (auto& en : enrolls) {
        if (!embed_wav(en.path, en.embedding)) {
            rs_speaker_free(sp);
            return 1;
        }
        std::fprintf(stdout, "  + %-16s  ← %s\n", en.name.c_str(),
                     en.path.c_str());
    }

    // Query
    std::fprintf(stdout, "\nQuery: %s\n", query_path.c_str());
    std::vector<float> q_emb;
    if (!embed_wav(query_path, q_emb)) {
        rs_speaker_free(sp);
        return 1;
    }

    if (!dump_path.empty()) {
        FILE* f = std::fopen(dump_path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "Failed to open dump file '%s'\n",
                         dump_path.c_str());
            rs_speaker_free(sp);
            return 1;
        }
        std::fwrite(q_emb.data(), sizeof(float), q_emb.size(), f);
        std::fclose(f);
        std::fprintf(stdout, "Wrote %d floats to %s\n", (int)q_emb.size(),
                     dump_path.c_str());
    }

    // Score
    std::fprintf(stdout, "\nCosine similarity:\n");
    int best_idx = -1;
    float best_score = -2.0f;
    for (size_t i = 0; i < enrolls.size(); ++i) {
        float c = rs_speaker_cosine(q_emb.data(), enrolls[i].embedding.data(),
                                    dim);
        std::fprintf(stdout, "  %-16s  %+.4f\n", enrolls[i].name.c_str(), c);
        if (c > best_score) {
            best_score = c;
            best_idx = (int)i;
        }
    }
    if (best_idx >= 0) {
        std::fprintf(stdout, "\n→ best match: %s (cos=%+.4f)\n",
                     enrolls[best_idx].name.c_str(), best_score);
    }

    rs_speaker_free(sp);
    return 0;
}
