#include "frontend/seamless_mel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
void rdft(int n, int isgn, double *a, int *ip, double *w);
}

namespace {

// Kaldi mel scale: mel = 1127 * ln(1 + hz/700)
inline double hz_to_mel_kaldi(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

// Inverse Kaldi mel scale: hz = 700 * (exp(mel/1127) - 1)
inline double mel_to_hz_kaldi(double mel) {
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

// Symmetric Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))
inline double hann_symmetric(int n, int N) {
    return 0.5 - 0.5 * std::cos(2.0 * M_PI * n / (double)(N - 1));
}

bool debug_enabled() {
    const char *v = std::getenv("RS_SEAMLESS_MEL_DEBUG");
    return v && *v && std::strcmp(v, "0") != 0;
}

template <typename T>
void print_stats(const char *name, const std::vector<T> &v) {
    if (v.empty()) {
        std::fprintf(stderr, "[seamless_mel] %s: empty\n", name);
        return;
    }
    double sum = 0.0;
    double sq = 0.0;
    double mn = std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    size_t nz = 0;
    for (T x0 : v) {
        double x = (double)x0;
        sum += x;
        sq += x * x;
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        if (x != 0.0) nz++;
    }
    const double n = (double)v.size();
    std::fprintf(stderr,
                 "[seamless_mel] %s: n=%zu min=%.8g max=%.8g mean=%.8g rms=%.8g nonzero=%zu\n",
                 name, v.size(), mn, mx, sum / n, std::sqrt(sq / n), nz);
}

} // namespace

SeamlessMelExtractor::SeamlessMelExtractor(const SeamlessMelConfig &config)
    : cfg_(config) {
    InitWindow();
    InitMelFilters();
}

SeamlessMelExtractor::~SeamlessMelExtractor() = default;

void SeamlessMelExtractor::InitWindow() {
    const int frame_len = cfg_.frame_length;
    const int n_fft     = cfg_.n_fft;

    // Povey window: symmetric Hann raised to power 0.85, zero-padded to n_fft.
    povey_window_.assign(n_fft, 0.0);
    for (int i = 0; i < frame_len; ++i) {
        double h = hann_symmetric(i, frame_len);
        povey_window_[i] = std::pow(h, 0.85);
    }

    // Ooura rdft workspace for n_fft.
    fft_ip_.assign(2 + (int)std::sqrt((double)n_fft / 2) + 1, 0);
    fft_w_.assign(n_fft / 2, 0.0);
    fft_ip_[0] = 0;
    std::vector<double> dummy(n_fft, 0.0);
    rdft(n_fft, 1, dummy.data(), fft_ip_.data(), fft_w_.data());
}

void SeamlessMelExtractor::InitMelFilters() {
    const int n_mels  = cfg_.n_mels;
    const int n_bins  = cfg_.n_fft / 2 + 1;  // 257
    mel_filters_.assign((size_t)n_mels * n_bins, 0.0f);

    // mel-space center points, equally spaced.
    const double mel_min = hz_to_mel_kaldi(cfg_.f_min);
    const double mel_max = hz_to_mel_kaldi(cfg_.f_max);
    std::vector<double> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (double)(n_mels + 1);
    }

    // triangularize_in_mel_space=True: fft bins in mel space, filter centers in mel space.
    const double fft_bin_hz = cfg_.sample_rate / ((n_bins - 1) * 2.0);
    std::vector<double> fft_mel(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        fft_mel[k] = hz_to_mel_kaldi(fft_bin_hz * k);
    }

    for (int m = 0; m < n_mels; ++m) {
        const double m_left   = mel_pts[m];
        const double m_center = mel_pts[m + 1];
        const double m_right  = mel_pts[m + 2];
        for (int k = 0; k < n_bins; ++k) {
            const double mel = fft_mel[k];
            const double up   = (mel - m_left) / (m_center - m_left);
            const double down = (m_right - mel) / (m_right - m_center);
            mel_filters_[(size_t)m * n_bins + k] = (float)std::max(0.0, std::min(up, down));
        }
    }
}

int SeamlessMelExtractor::Compute(const std::vector<float> &pcm,
                                   std::vector<float> &out) const {
    const int n_fft   = cfg_.n_fft;
    const int hop     = cfg_.hop_length;
    const int frm     = cfg_.frame_length;
    const int n_mels  = cfg_.n_mels;
    const int n_bins  = n_fft / 2 + 1;
    const int stride  = cfg_.stride;
    const bool debug = debug_enabled();

    if (pcm.empty()) {
        out.clear();
        return 0;
    }

    if (debug) {
        print_stats("pcm", pcm);
        print_stats("mel_filters", mel_filters_);
        std::fprintf(stderr,
                     "[seamless_mel] cfg: sr=%d frame=%d hop=%d n_fft=%d n_mels=%d stride=%d\n",
                     cfg_.sample_rate, cfg_.frame_length, cfg_.hop_length,
                     cfg_.n_fft, cfg_.n_mels, cfg_.stride);
    }

    // 1. Copy to double, multiply by 2^15 (Kaldi compliance).
    const int N = (int)pcm.size();
    std::vector<double> waveform(N);
    const double scale = 32768.0;  // 2^15
    for (int i = 0; i < N; ++i) {
        waveform[i] = (double)pcm[i] * scale;
    }

    // 2. Frame count (center=False).
    if (N < frm) {
        out.clear();
        return 0;
    }
    const int n_frames = (N - frm) / hop + 1;

    // 3. Per-frame: DC removal → preemphasis → window → FFT → power → mel → log.
    //    Layout: [n_bins, n_frames] transposed, but we store as
    //    log_mel[m * n_frames + t] (mel-bin slow, frame fast) for vectorization.
    std::vector<float> log_mel((size_t)n_mels * n_frames, 0.0f);

    auto frame_worker = [&](int t_start, int t_end) {
        std::vector<double> buf(n_fft);
        std::vector<double> power_spec(n_bins);
        for (int t = t_start; t < t_end; ++t) {
            const int offset = t * hop;

            // Copy frame samples.
            for (int j = 0; j < frm; ++j) {
                buf[j] = waveform[offset + j];
            }
            // Zero-pad rest of FFT buffer.
            for (int j = frm; j < n_fft; ++j) {
                buf[j] = 0.0;
            }

            // Remove DC offset.
            double mean = 0.0;
            for (int j = 0; j < frm; ++j) mean += buf[j];
            mean /= frm;
            for (int j = 0; j < frm; ++j) buf[j] -= mean;

            // Pre-emphasis.
            for (int j = frm - 1; j > 0; --j) {
                buf[j] -= cfg_.preemphasis * buf[j - 1];
            }
            buf[0] *= (1.0 - cfg_.preemphasis);

            // Apply window.
            for (int j = 0; j < n_fft; ++j) {
                buf[j] *= povey_window_[j];
            }

            // Real DFT via Ooura rdft.
            rdft(n_fft, 1, buf.data(),
                 const_cast<int *>(fft_ip_.data()),
                 const_cast<double *>(fft_w_.data()));

            // Ooura layout: a[0]=Re(0), a[1]=Re(N/2), a[2k]=Re(k), a[2k+1]=Im(k).
            power_spec[0]        = buf[0] * buf[0];
            power_spec[n_bins - 1] = buf[1] * buf[1];
            for (int k = 1; k < n_bins - 1; ++k) {
                const double re = buf[2 * k];
                const double im = buf[2 * k + 1];
                power_spec[k] = re * re + im * im;
            }

            // Mel projection + log.
            for (int m = 0; m < n_mels; ++m) {
                const float *row = &mel_filters_[(size_t)m * n_bins];
                double acc = 0.0;
                for (int k = 0; k < n_bins; ++k) {
                    acc += (double)row[k] * power_spec[k];
                }
                if (acc < (double)cfg_.mel_floor) acc = (double)cfg_.mel_floor;
                log_mel[(size_t)m * n_frames + t] = (float)std::log(acc);
            }
        }
    };

#if defined(_OPENMP)
#pragma omp parallel
    {
        int nthr = omp_get_num_threads();
        int tid  = omp_get_thread_num();
        int per  = (n_frames + nthr - 1) / nthr;
        int s    = std::min(tid * per, n_frames);
        int e    = std::min(s + per, n_frames);
        frame_worker(s, e);
    }
#else
    frame_worker(0, n_frames);
#endif

    if (debug) {
        print_stats("log_mel_pre_norm", log_mel);
    }

    // 4. Per-bin z-score normalization (ddof=1).
    for (int m = 0; m < n_mels; ++m) {
        float *row = &log_mel[(size_t)m * n_frames];
        double sum = 0.0;
        for (int t = 0; t < n_frames; ++t) sum += (double)row[t];
        double mean = sum / n_frames;

        double var_sum = 0.0;
        for (int t = 0; t < n_frames; ++t) {
            double d = (double)row[t] - mean;
            var_sum += d * d;
        }
        double std_dev = std::sqrt(var_sum / (n_frames - 1) + 1e-7);

        for (int t = 0; t < n_frames; ++t) {
            row[t] = (float)(((double)row[t] - mean) / std_dev);
        }
    }

    if (debug) {
        print_stats("log_mel_post_norm", log_mel);
    }

    // 5. Stride-2 frame stacking: [n_mels=80, n_frames] → [T_out, 160].
    const int T_remainder = n_frames % stride;
    const int T_in        = n_frames - T_remainder;
    const int T_out       = T_in / stride;

    out.assign((size_t)T_out * n_mels * stride, 0.0f);

    for (int t = 0; t < T_out; ++t) {
        for (int s = 0; s < stride; ++s) {
            const int src_t = t * stride + s;
            for (int m = 0; m < n_mels; ++m) {
                // Python does np.reshape([num_frames, n_mels],
                // [num_frames / stride, n_mels * stride]) in C order:
                // [frame0_mels..., frame1_mels...].
                out[(size_t)t * n_mels * stride + s * n_mels + m] =
                    log_mel[(size_t)m * n_frames + src_t];
            }
        }
    }

    if (debug) {
        print_stats("input_features", out);
        std::fprintf(stderr,
                     "[seamless_mel] frames: input=%d stacked=%d dim=%d\n",
                     n_frames, T_out, n_mels * stride);
    }

    return T_out;
}
