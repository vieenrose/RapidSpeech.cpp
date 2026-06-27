#pragma once
#ifdef __cplusplus
extern "C" {
#endif
bool qwen3_mel_cufft(const float* samples_padded, int n_padded,
                     int compute_frames, int frame_size, int frame_step,
                     const float* hann_f, const float* mel_filters,
                     int n_mel, int n_bins, double* temp_data);
#ifdef __cplusplus
}
#endif
