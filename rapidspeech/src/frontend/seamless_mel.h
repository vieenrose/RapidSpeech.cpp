#pragma once

// SeamlessM4T fbank-160 feature extractor.
//
// Mirrors transformers SeamlessM4TFeatureExtractor:
//   1. Multiply PCM by 2^15 (Kaldi 16-bit integer compliance)
//   2. Remove DC offset per frame
//   3. Pre-emphasis (0.97)
//   4. Povey window (symmetric Hann^0.85) + STFT (n_fft=512, hop=160, center=False)
//   5. Power spectrogram (|X|^2)
//   6. 80-bin Kaldi-scale mel filterbank (20-8000 Hz, triangularize_in_mel_space)
//   7. Natural log, clamp to mel_floor (1.192092955078125e-07)
//   8. Per-bin z-score normalization (ddof=1)
//   9. Stride-2 frame stacking → output [T, 160] (C-order)
//
// Usage:
//   SeamlessMelConfig cfg;  // defaults match SeamlessM4TFeatureExtractor
//   SeamlessMelExtractor ext(cfg);
//   std::vector<float> out;
//   int T = ext.Compute(pcm_16k_mono, out);  // out.size() == T * 160

#include <vector>

struct SeamlessMelConfig {
    int sample_rate    = 16000;
    int frame_length   = 400;    // 25 ms
    int hop_length     = 160;    // 10 ms
    int n_fft          = 512;
    int n_mels         = 80;
    float f_min        = 20.0f;
    float f_max        = 8000.0f;
    float preemphasis  = 0.97f;
    float mel_floor    = 1.192092955078125e-07f;  // 2^-23
    int stride         = 2;      // frame stacking factor
    bool center        = false;
};

class SeamlessMelExtractor {
public:
    explicit SeamlessMelExtractor(const SeamlessMelConfig &config = SeamlessMelConfig());
    ~SeamlessMelExtractor();

    // Convert 16kHz mono PCM float [-1,1] to fbank-160 features.
    // Returns T (number of output frames); out is resized to T * 160.
    int Compute(const std::vector<float> &pcm, std::vector<float> &out) const;

    int n_mels() const { return cfg_.n_mels; }

private:
    SeamlessMelConfig cfg_;

    std::vector<double> povey_window_;    // n_fft (zero-padded from frame_length)
    std::vector<float>  mel_filters_;     // [n_mels, n_freq_bins] row-major
    std::vector<int>    fft_ip_;          // Ooura rdft workspace
    std::vector<double> fft_w_;           // Ooura rdft workspace

    void InitWindow();
    void InitMelFilters();
};
