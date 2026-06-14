#pragma once
// Matcha-TTS (conditional flow-matching) arch for RapidSpeech.cpp.
//
// text (phoneme+tone IDs) -> TextEncoder (emb -> ConvReluNorm prenet -> 6x RoPE
//   transformer -> proj_m=mu, proj_w=durations) -> length-regulate -> CFM decoder
//   (1-D UNet, 3-step Euler ODE) -> mel -> Vocos vocoder (ConvNeXt + iSTFT) -> audio.
//
// Forward graph numerically validated against ONNX component-by-component
// (see docs/MATCHA_TTS_GGML_PORT.md): encoder rel<=1e-4, vocos rel 3e-4, iSTFT corr 1.0.
//
// Weights come from scripts/convert_matcha_onnx_to_gguf.py (474 tensors, arch="matcha-tts").
#include "core/rs_context.h"
#include "core/rs_model.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;

struct MatchaHParams {
  int32_t hidden       = 192;   // encoder hidden
  int32_t n_heads      = 2;
  int32_t head_dim     = 96;
  int32_t rotary_dim   = 48;    // partial rotary (first 48 of 96)
  int32_t n_layers     = 6;
  int32_t filter       = 768;   // FFN inner
  int32_t n_mels       = 80;
  int32_t dec_hidden   = 256;   // CFM decoder channels
  int32_t time_dim     = 1024;
  int32_t num_ode_steps = 3;
  int32_t sample_rate  = 8000;
  int32_t n_fft        = 512;
  int32_t hop_length   = 128;
  int32_t pad_id       = 0;
  int32_t n_vocab      = 0;     // from emb
};

struct MatchaState : public RSState {
  std::vector<int32_t> phoneme_ids;     // input token IDs (from text frontend)
  std::vector<int32_t> tone_ids;        // (unused by this acoustic model — kept for parity)
  float noise_scale  = 0.667f;
  float length_scale = 1.0f;
  std::vector<float> audio_output;
  int audio_read_cursor = 0;
};

class MatchaModel : public ISpeechModel {
public:
  bool Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float>&, RSState&, ggml_backend_sched_t) override { return true; }
  bool Decode(RSState&, ggml_backend_sched_t) override { return true; }
  std::string GetTranscription(RSState&) override { return ""; }
  bool PushText(RSState& state, const char* text, const char* language, const char* instruct) override;
  int GetAudioOutput(RSState& state, float** out_data) override;
  const RSModelMeta& GetMeta() const override { return meta_; }

private:
  MatchaHParams hp_;
  RSModelMeta meta_;
  std::map<std::string, ggml_tensor*> w_;   // weights from gguf_data
  rs_context_t* rsctx_ = nullptr;

  ggml_tensor* W(const std::string& name) const;
  // forward stages (defined in matcha.cpp); ctx is a graph-build context
  ggml_tensor* build_encoder(ggml_context* ctx, ggml_tensor* ids, int L,
                             ggml_tensor* cos_t, ggml_tensor* sin_t, ggml_tensor** logw_out);
  ggml_tensor* build_cfm(ggml_context* ctx, ggml_tensor* mu, int T, float noise_scale);
  ggml_tensor* build_vocos(ggml_context* ctx, ggml_tensor* mel, int T);  // -> head [514,T]
  std::vector<float> istft(const float* head, int T) const;             // -> waveform
};
