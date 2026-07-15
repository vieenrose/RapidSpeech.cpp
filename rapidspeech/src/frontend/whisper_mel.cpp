#include "frontend/whisper_mel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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

// Convert Hz to Slaney mel (matches librosa.filters.mel(htk=False)).
//   Below 1000 Hz: linear ramp at 200/3 Hz per mel.
//   Above 1000 Hz: log ramp through 6.4 octaves spanning the remainder.
inline double hz_to_mel_slaney(double hz) {
  const double f_min        = 0.0;
  const double f_sp         = 200.0 / 3.0;
  const double min_log_hz   = 1000.0;
  const double min_log_mel  = (min_log_hz - f_min) / f_sp; // 15
  const double logstep      = std::log(6.4) / 27.0;
  if (hz >= min_log_hz) {
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  return (hz - f_min) / f_sp;
}

inline double mel_to_hz_slaney(double mel) {
  const double f_min        = 0.0;
  const double f_sp         = 200.0 / 3.0;
  const double min_log_hz   = 1000.0;
  const double min_log_mel  = (min_log_hz - f_min) / f_sp;
  const double logstep      = std::log(6.4) / 27.0;
  if (mel >= min_log_mel) {
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
  }
  return f_min + f_sp * mel;
}

} // namespace

WhisperMelExtractor::WhisperMelExtractor(const WhisperMelConfig &config)
    : cfg_(config) {
  InitTables();
  InitMelFilters();
}

WhisperMelExtractor::~WhisperMelExtractor() = default;

void WhisperMelExtractor::InitTables() {
  // Periodic Hann window (torch.hann_window default periodic=True) — this is
  // what WhisperFeatureExtractor uses: denominator is n_fft, NOT n_fft-1.
  hann_window_.resize(cfg_.n_fft);
  for (int i = 0; i < cfg_.n_fft; ++i) {
    hann_window_[i] =
        0.5 - 0.5 * std::cos(2.0 * M_PI * i / (double)cfg_.n_fft);
  }

  // Direct DFT basis [n_bins][n_fft] (Whisper n_fft=400 is not a power of 2, so
  // Ooura's rdft cannot be used; a direct DFT matches WhisperFeatureExtractor).
  const int n_bins = cfg_.n_fft / 2 + 1;
  dft_cos_.assign((size_t)n_bins * cfg_.n_fft, 0.0f);
  dft_sin_.assign((size_t)n_bins * cfg_.n_fft, 0.0f);
  for (int k = 0; k < n_bins; ++k) {
    for (int n = 0; n < cfg_.n_fft; ++n) {
      const double a = 2.0 * M_PI * k * n / (double)cfg_.n_fft;
      dft_cos_[(size_t)k * cfg_.n_fft + n] = (float)std::cos(a);
      dft_sin_[(size_t)k * cfg_.n_fft + n] = (float)std::sin(a);
    }
  }
}

void WhisperMelExtractor::InitMelFilters() {
  const int n_mels  = cfg_.n_mels;
  const int n_bins  = cfg_.n_fft / 2 + 1;
  mel_filters_.assign((size_t)n_mels * n_bins, 0.0f);

  const double mel_lo = hz_to_mel_slaney(cfg_.f_min);
  const double mel_hi = hz_to_mel_slaney(cfg_.f_max);
  std::vector<double> mel_points(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    mel_points[i] = mel_lo + (mel_hi - mel_lo) * i / (double)(n_mels + 1);
  }
  std::vector<double> hz_points(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    hz_points[i] = mel_to_hz_slaney(mel_points[i]);
  }

  const double fft_bin_hz = (double)cfg_.sample_rate / (double)cfg_.n_fft;
  for (int m = 0; m < n_mels; ++m) {
    const double f_left   = hz_points[m];
    const double f_center = hz_points[m + 1];
    const double f_right  = hz_points[m + 2];
    // Slaney area-normalization factor.
    const double enorm = 2.0 / (f_right - f_left);
    for (int k = 0; k < n_bins; ++k) {
      const double hz       = fft_bin_hz * k;
      const double up       = (hz - f_left) / (f_center - f_left);
      const double down     = (f_right - hz) / (f_right - f_center);
      const double weight   = std::max(0.0, std::min(up, down)) * enorm;
      mel_filters_[(size_t)m * n_bins + k] = (float)weight;
    }
  }
}

int WhisperMelExtractor::Compute(const std::vector<float> &pcm,
                                 std::vector<float> &out) const {
  const int n_fft   = cfg_.n_fft;
  const int hop     = cfg_.hop_length;
  const int n_mels  = cfg_.n_mels;
  const int n_bins  = n_fft / 2 + 1;
  const int chunk   = cfg_.chunk_size;
  const int pad     = n_fft / 2;

  if (pcm.empty()) {
    out.clear();
    return 0;
  }

  // 1. Reflection padding (matches WhisperFeatureExtractor(center=True)).
  std::vector<float> padded;
  padded.resize((size_t)pcm.size() + 2 * pad);
  // Reflect-without-endpoint: mirror across index 0 / index N-1.
  for (int i = 0; i < pad; ++i) {
    int src = pad - i;             // 1..pad
    if (src >= (int)pcm.size()) src = (int)pcm.size() - 1;
    padded[i] = pcm[src];
  }
  std::memcpy(padded.data() + pad, pcm.data(), pcm.size() * sizeof(float));
  for (int i = 0; i < pad; ++i) {
    int src = (int)pcm.size() - 2 - i; // N-2, N-3, ...
    if (src < 0) src = 0;
    padded[pad + pcm.size() + i] = pcm[src];
  }

  // 2. Frame count after centering: floor((N + 2*pad - n_fft) / hop) + 1
  //    With center=True this is equivalent to ceil(N / hop).
  const int n_padded = (int)padded.size();
  if (n_padded < n_fft) {
    out.clear();
    return 0;
  }
  const int n_frames = (n_padded - n_fft) / hop + 1;

  // 3. STFT + |X|^2 + mel projection + log10 for every frame.
  //    Output layout: mel-bin slow, frame fast. log_mel[m * n_frames + t].
  std::vector<float> log_mel((size_t)n_mels * n_frames, 0.0f);

  auto frame_worker = [&](int t_start, int t_end) {
    std::vector<double> buf(n_fft);
    std::vector<double> power(n_bins);
    for (int t = t_start; t < t_end; ++t) {
      const int offset = t * hop;
      for (int j = 0; j < n_fft; ++j) {
        buf[j] = (double)padded[offset + j] * hann_window_[j];
      }
      // Direct DFT |X[k]|^2 for k in [0, n_bins) (matches HF; n_fft=400).
      for (int k = 0; k < n_bins; ++k) {
        const float *cs = &dft_cos_[(size_t)k * n_fft];
        const float *sn = &dft_sin_[(size_t)k * n_fft];
        double re = 0.0, im = 0.0;
        for (int j = 0; j < n_fft; ++j) {
          re += (double)cs[j] * buf[j];
          im -= (double)sn[j] * buf[j];  // e^{-i...} convention
        }
        power[k] = re * re + im * im;
      }
      for (int m = 0; m < n_mels; ++m) {
        const float *row = &mel_filters_[(size_t)m * n_bins];
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k) {
          acc += (double)row[k] * power[k];
        }
        log_mel[(size_t)m * n_frames + t] =
            (float)std::log10(std::max(acc, 1e-10));
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

  // 4. Whisper-style normalization: clamp(log_mel, max-8) shifted to [0, 1+].
  //    Computed across the whole utterance once (so a longer wav stays
  //    self-consistent across chunks).
  float mmax = -1e30f;
  for (float v : log_mel) {
    if (v > mmax) mmax = v;
  }
  const float floor_v = mmax - 8.0f;
  for (float &v : log_mel) {
    if (v < floor_v) v = floor_v;
    v = (v + 4.0f) / 4.0f;
  }

  // 5. Pad the frame count. Output layout: frame fast, mel-bin slow, so the
  //    encoder graph can view it as [n_out_frames, n_mel, 1, n_chunks].
  //    For audio that fits in a single chunk (<= chunk frames), pad only to the
  //    next multiple of 8 rather than the full chunk: the MOSS encoder is
  //    variable-length, and 8-alignment satisfies conv2 (stride 2) + merge-4 so
  //    it produces exactly ceil(real/8) tokens without processing silence.
  //    For longer audio keep uniform chunk-sized chunks.
  // ceil, NOT +1: for exact-multiple durations (e.g. 180.00 s) a spurious
  // extra frame rounded the chunk count up, appending an entire 30 s
  // all-silence chunk (375 dead audio tokens the HF reference never sees).
  const int n_eff   = std::min(n_frames, (int)((pcm.size() + hop - 1) / hop));
  const int ENC_ALIGN = 8;   // conv2 stride-2 (even) * merge-4
  // RS_NO_ENC_TRUNC=1 restores full chunk-sized padding (30 s) for A/B accuracy
  // comparison against the encoder-truncation optimization.
  const bool no_trunc = std::getenv("RS_NO_ENC_TRUNC") != nullptr;
  // Trailing context margin (frames) past the last speech. The MOSS timestamp/EOS
  // head was trained with the full 30 s Whisper chunk, so more trailing context
  // = timestamps closer to the full-30 s (== ORT) reference (native: ~800 frames
  // reproduces it byte-for-byte). On WASM, Q4_K/ggml numerics still diverge from
  // native/ORT regardless, so the default trims aggressively for speed and the
  // margin is an env-tunable accuracy knob (RS_ENC_MARGIN=800 for best timestamps).
  const char *menv = std::getenv("RS_ENC_MARGIN");
  const int margin = menv ? std::atoi(menv) : 800;
  int n_out_frames;
  if (n_eff <= chunk && !no_trunc) {
    int want = n_eff + margin;
    n_out_frames = std::min(chunk, std::max(ENC_ALIGN, ((want + ENC_ALIGN - 1) / ENC_ALIGN) * ENC_ALIGN));
  } else {
    const int n_chunks = (n_eff + chunk - 1) / chunk;
    n_out_frames = n_chunks * chunk;
  }

  // Pad with the NORMALIZED SILENCE FLOOR, not 0.0: the HF reference pads the
  // raw audio with zeros, whose log-mel clamps to (mmax-8) and normalizes to
  // (mmax-4)/4. The MOSS encoder is bidirectional within a chunk with NO
  // padding mask, so real frames in a partial last chunk attend to the pad
  // region — 0.0 there is out-of-distribution and corrupts that chunk's
  // tokens (broken timestamps, unstable quantized decodes).
  const float pad_v = (floor_v + 4.0f) / 4.0f;
  out.assign((size_t)n_mels * n_out_frames, pad_v);
  const int n_copy = std::min(n_eff, n_frames);
  for (int m = 0; m < n_mels; ++m) {
    std::memcpy(out.data() + (size_t)m * n_out_frames,
                log_mel.data() + (size_t)m * n_frames,
                (size_t)n_copy * sizeof(float));
  }
  return n_out_frames;
}
