#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h"   // RS_API
#include <vector>
#include <string>

/**
 * Silero VAD hyperparameters (loaded from GGUF metadata).
 */
struct SileroVadHParams {
    int sample_rate = 16000;
    int window_size_samples = 512;   // 32ms at 16kHz
    int context_samples = 64;
    int state_size = 256;            // 2 * 1 * 128 flattened
    float threshold = 0.5f;
    int min_silence_ms = 200;
    int min_speech_ms = 250;
    int speech_pad_ms = 30;
};

/**
 * Runtime state for Silero VAD inference.
 * Implements VADIterator algorithm with hysteresis and temp_end mechanism.
 *
 * LSTM h/c state stored in host memory (h_state: first 128 = h, next 128 = c).
 * Each frame, h/c are uploaded to graph input tensors via ggml_backend_tensor_set,
 * and the new h/c are read back from graph output tensors via ggml_backend_tensor_get.
 */
struct SileroVadState : public RSState {
    // Hidden state: first 128 = LSTM h, next 128 = LSTM c
    std::vector<float> h_state;
    // Context buffer: last 64 samples from previous chunk
    std::vector<float> context;
    // Last computed speech probability
    float speech_prob = 0.0f;

    // VADIterator state machine
    bool triggered = false;           // True when speech is detected
    int temp_end_sample = 0;         // Sample index when prob dropped below threshold
    int current_sample = 0;          // Total samples processed
    int speech_pad_samples = 480;    // 30ms * 16000 / 1000

    SileroVadState() {
        h_state.resize(256, 0.0f);
        context.resize(64, 0.0f);
    }
};

/**
 * Silero VAD weights mapped from GGUF tensors.
 * The model is a small STFT + Conv + LSTM network.
 *
 * GGUF data layout:
 *   For 3D conv weights (PT shape [OC,IC,K]):
 *     Stored as [OC, IC, K] in C-order. GGML reads ne=[K, IC, OC].
 *     Row-major [K,IC,OC] IS contiguous with [1,IC*K,OC] — no transpose needed.
 *   For 2D LSTM weights (PT shape [4H,H]):
 *     Stored as [H,4H] in C-order (transpose from PyTorch).
 *     GGML reads ne=[4H, H]; ggml_mul_mat computes A^T @ B = F.linear result.
 */
struct SileroVadWeights {
    // STFT / feature extraction convolutions
    struct ggml_tensor* stft_conv_weight = nullptr;

    // Encoder convolution blocks
    struct ggml_tensor* enc_conv1_weight = nullptr;
    struct ggml_tensor* enc_conv1_bias = nullptr;
    struct ggml_tensor* enc_conv2_weight = nullptr;
    struct ggml_tensor* enc_conv2_bias = nullptr;
    struct ggml_tensor* enc_conv3_weight = nullptr;
    struct ggml_tensor* enc_conv3_bias = nullptr;
    struct ggml_tensor* enc_conv4_weight = nullptr;
    struct ggml_tensor* enc_conv4_bias = nullptr;

    // LSTM layers
    struct ggml_tensor* lstm_weight_ih = nullptr;
    struct ggml_tensor* lstm_weight_hh = nullptr;
    struct ggml_tensor* lstm_bias_ih = nullptr;
    struct ggml_tensor* lstm_bias_hh = nullptr;

    // Output projection
    struct ggml_tensor* out_weight = nullptr;
    struct ggml_tensor* out_bias = nullptr;
};

/**
 * Silero VAD model — ggml graph computation implementation.
 *
 * Uses ggml computation graph for model forward pass:
 *   - ggml_conv_1d for STFT and encoder convolutions
 *   - ggml_mul_mat for LSTM gates
 *   - LSTM h/c state passed via set_input/set_output + backend_tensor_set/get
 *   - Uses the caller-provided ggml_backend_sched for graph allocation and compute
 */
class RS_API SileroVadModel : public ISpeechModel {
public:
    SileroVadModel();
    ~SileroVadModel() override;

    bool Load(const std::unique_ptr<rs_context_t>& ctx,
              ggml_backend_t backend) override;

    /**
     * Alternative load path: directly accepts raw ggml/gguf contexts.
     * Use this when you don't have a full rs_context_t (e.g. in CLI tools).
     */
    bool LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                    ggml_backend_t backend);

    std::shared_ptr<RSState> CreateState() override;
    bool Encode(const std::vector<float>& input_frames, RSState& state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState& state, ggml_backend_sched_t sched) override;
    std::string GetTranscription(RSState& state) override;
    const RSModelMeta& GetMeta() const override { return meta_; }

    // VAD-specific: get speech probability from last Encode call
    float GetSpeechProbability(RSState& state);

    // Check if speech is currently triggered
    bool IsSpeechTriggered(RSState& state) {
        return dynamic_cast<SileroVadState&>(state).triggered;
    }

    // Set VAD threshold at runtime
    void SetThreshold(float threshold) {
        hparams_.threshold = threshold;
    }

    float GetThreshold() const { return hparams_.threshold; }

    ggml_backend_t GetBackend() { return backend_; }

private:
    RSModelMeta meta_;
    SileroVadHParams hparams_;
    SileroVadWeights weights_;
    ggml_backend_t backend_ = nullptr;

    bool MapTensors(ggml_context* gguf_data);
    bool LoadHParams(gguf_context* ctx_gguf);
};
