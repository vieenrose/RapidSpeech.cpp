#pragma once

#include <vector>

/**
 * Whisper-style log-mel spectrogram extractor.
 *
 * Used by Qwen3-ASR and other models that share the HuggingFace
 * WhisperFeatureExtractor pipeline:
 *   - n_fft=400, hop=160, win=400 (matches sample_rate=16000)
 *   - Hann window (symmetric)
 *   - Reflection padding of n_fft/2 samples on each end (center=True)
 *   - 128-bin Slaney mel filterbank, [0, sr/2]
 *   - log10(max(mel, 1e-10))
 *   - Global clamp: mmax = max(log_mel) - 8;
 *     mel = (max(v, mmax) + 4) / 4
 *
 * Output is laid out as [n_mels * n_frames], with mel bins as the fast axis
 * (frame index slowest), so a graph view can reshape it to a 4D tensor
 * [chunk_size, n_mels, 1, n_chunks] directly.
 */
struct WhisperMelConfig {
  int sample_rate = 16000;
  int n_fft       = 400;
  int hop_length  = 160;
  int n_mels      = 128;
  float f_min     = 0.0f;
  float f_max     = 8000.0f;
  // Chunk size in mel frames. Output is padded to a multiple of this.
  // Qwen3-ASR: chunk_size = 100 (2 * n_window with n_window=50).
  int chunk_size  = 100;
};

class WhisperMelExtractor {
public:
  explicit WhisperMelExtractor(const WhisperMelConfig &config);
  ~WhisperMelExtractor();

  /**
   * Convert PCM into a padded log-mel spectrogram.
   *
   * @param pcm input audio, mono float in [-1, 1] at config.sample_rate
   * @param out output buffer, resized to n_mels * n_frames_padded (f32)
   * @return n_frames_padded (always a multiple of chunk_size; equals chunk_size
   *         * n_chunks). n_chunks can be derived by the caller via
   *         n_frames_padded / chunk_size.
   */
  int Compute(const std::vector<float> &pcm, std::vector<float> &out) const;

  int n_mels()    const { return cfg_.n_mels; }
  int chunk_size() const { return cfg_.chunk_size; }
  int hop_length() const { return cfg_.hop_length; }

private:
  WhisperMelConfig cfg_;

  // Precomputed tables.
  std::vector<double> hann_window_;        // n_fft
  std::vector<float>  mel_filters_;        // [n_mels, n_fft/2 + 1] (row-major)
  std::vector<int>    fft_ip_;             // Ooura rdft workspace
  std::vector<double> fft_w_;              // Ooura rdft workspace
  // Direct-DFT basis [n_bins * n_fft]. Ooura's rdft only supports power-of-2
  // lengths, but Whisper uses n_fft=400 — a direct DFT matches HF exactly.
  std::vector<float>  dft_cos_;
  std::vector<float>  dft_sin_;

  void InitTables();
  void InitMelFilters();
};
