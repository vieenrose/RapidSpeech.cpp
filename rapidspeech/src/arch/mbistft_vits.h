#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// =====================================================================
// MB-iSTFT-VITS TTS ("mbistft-vits", PrimeTTS v2)
//
// Single-speaker VITS variant with a multiband-iSTFT generator head:
//   (phone,tone,lang) ids -> TextEncoder (6x windowed rel-pos transformer)
//     -> deterministic DurationPredictor -> length regulate
//     -> ResidualCouplingBlock flow (reverse)
//     -> Multiband iSTFT generator (HiFiGAN upsample + per-subband iSTFT
//        + PQMF synthesis) -> 16 kHz waveform
//
// No BERT, no speaker embedding, no SDP (g = NULL everywhere).
// Text frontend is out of scope for now: callers push precomputed,
// blank-interleaved id streams directly into the state (see the
// mbistft_parity tool).  Backend-agnostic ggml graphs (standard ops only).
// =====================================================================

struct MBIstftVitsHParams {
  int32_t n_vocab          = 88;
  int32_t num_tones        = 6;
  int32_t num_langs        = 2;
  int32_t hidden_channels  = 192;
  int32_t inter_channels   = 192;
  int32_t filter_channels  = 768;
  int32_t n_heads          = 2;
  int32_t n_layers         = 6;
  int32_t kernel_size      = 3;   // FFN conv kernel
  int32_t window_size      = 4;   // rel-pos attention window
  int32_t k_channels       = 96;
  int32_t subbands         = 4;
  int32_t gen_istft_n_fft  = 16;
  int32_t gen_istft_hop    = 4;
  int32_t ups_initial_ch   = 512;
  int32_t sample_rate      = 16000;
  int32_t hop_length       = 256;
  std::vector<int32_t> upsample_rates        = {4, 4};
  std::vector<int32_t> upsample_kernel_sizes = {16, 16};
  std::vector<int32_t> resblock_kernel_sizes = {3, 7, 11};
  // dp filter channels (from tensor shapes at load)
  int32_t dp_filter_channels = 256;
  int32_t flow_n_layers      = 4;  // WN layers per coupling layer
  int32_t flow_kernel_size   = 5;
  int32_t n_flows            = 4;  // coupling layers (flow.flows.{0,2,4,6})
};

struct MBIstftVitsState : public RSState {
  // Blank-interleaved input id streams (all three, same length).
  std::vector<int32_t> phone_ids;
  std::vector<int32_t> tone_ids;
  std::vector<int32_t> lang_ids;

  float noise_scale  = 0.667f;  // parity harness sets 0.0 (deterministic)
  float length_scale = 1.0f;

  // Optional teacher-forced per-phone durations (models.py infer's
  // `durations` arg).  When set (same length as phone_ids), the expansion
  // uses these instead of ceil(exp(logw)); `w_ceil` still holds the model's
  // own prediction.  The parity harness uses this to keep downstream module
  // comparisons aligned when a knife-edge ceil() boundary flips by 1 frame.
  std::vector<int32_t> duration_override;

  // Host-side intermediates (ggml [C,T] layout: channel varies fastest).
  int T_text   = 0;
  int T_frames = 0;
  std::vector<float>   m_p, logs_p;   // [192*T_text]
  std::vector<float>   logw;          // [T_text]
  std::vector<int32_t> w_ceil;        // per-phone frame counts
  std::vector<float>   z_p;           // [192*T_frames] expanded prior (+noise)
  std::vector<float>   z;             // [192*T_frames] flow output

  // Audio output
  std::vector<float> audio_output;
  int audio_read_cursor = 0;

  // Parity capture: when true, all reference intermediates are stored in
  // `dumps` using the PyTorch memory order of the parity refs.
  bool capture = false;
  std::map<std::string, std::vector<float>>   dumps;
  std::map<std::string, std::vector<int64_t>> dump_shapes;

  MBIstftVitsState() {}
  ~MBIstftVitsState() override = default;
};

class MBIstftVitsModel : public ISpeechModel {
public:
  MBIstftVitsModel();
  ~MBIstftVitsModel() override;

  bool Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  // Encode: TextEncoder + DurationPredictor + length regulate + flow (reverse)
  bool Encode(const std::vector<float>& input_frames, RSState& state,
              ggml_backend_sched_t sched) override;

  // Decode: multiband iSTFT generator on the flow output
  bool Decode(RSState& state, ggml_backend_sched_t sched) override;

  std::string GetTranscription(RSState& state) override { (void)state; return ""; }
  const RSModelMeta& GetMeta() const override { return meta_; }

  // Text frontend not implemented yet; succeeds only if ids were pre-set.
  bool PushText(RSState& state, const char* text, const char* language = "zh",
                const char* instruct = nullptr) override;

  int GetAudioOutput(RSState& state, float** out_data) override;

private:
  RSModelMeta meta_;
  MBIstftVitsHParams hparams_;
  std::map<std::string, struct ggml_tensor*> tensors_;
  std::vector<float> istft_window_;  // host copy of istft.window

  struct ggml_tensor* Get(const std::string& name) const;

  bool RunEncoderAndDP(MBIstftVitsState& state, ggml_backend_sched_t sched);
  bool RunFlowReverse(MBIstftVitsState& state, ggml_backend_sched_t sched);
  bool RunDecoder(MBIstftVitsState& state, ggml_backend_sched_t sched);
};
