#include "arch/silero_vad.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstring>
#include <cmath>
#include <algorithm>

#define VAD_MAX_NODES 1024

// ============================================================
// Silero VAD — ggml graph computation
//
// Architecture (verified against Python Silero V6):
//   Input: 576 samples = 64 context + 512 window
//   -> pad_reflect_1d(right=64) -> 640 samples
//   -> STFT: conv1d(IC=1, OC=258, K=256, stride=128) -> magnitude
//   -> Encoder: 4x Conv1d(ReLU), strides=[1,2,2,1], pad=1
//   -> Squeeze: [1,128,1] -> [128,1]
//   -> LSTM: gates = W_ih@x + b_ih + W_hh@h + b_hh; split i,f,g,o
//   -> ReLU(h) -> Unsqueeze -> final conv1d -> sigmoid -> prob
//
// Weight layout:
//   3D conv: stored [OC,IC,K] C-order, GGML ne=[K,IC,OC]. Used directly.
//   2D LSTM: stored [4H,H] C-order, GGML ne=[H,4H].
//     ggml_mul_mat computes W^T @ x = [4H,H] @ [H,1] = [4H,1]. Correct!
// ============================================================

SileroVadModel::SileroVadModel() {
    meta_.arch_name = "silero-vad";
    meta_.audio_sample_rate = 16000;
    meta_.n_mels = 0;
    meta_.vocab_size = 0;
}

SileroVadModel::~SileroVadModel() = default;

bool SileroVadModel::MapTensors(ggml_context* gguf_data) {
    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(gguf_data, name);
        if (!t) RS_LOG_WARN("VAD tensor not found: %s", name);
        return t;
    };

    weights_.stft_conv_weight = get("stft_conv.weight");
    weights_.enc_conv1_weight = get("conv1.weight");
    weights_.enc_conv1_bias   = get("conv1.bias");
    weights_.enc_conv2_weight = get("conv2.weight");
    weights_.enc_conv2_bias   = get("conv2.bias");
    weights_.enc_conv3_weight = get("conv3.weight");
    weights_.enc_conv3_bias   = get("conv3.bias");
    weights_.enc_conv4_weight = get("conv4.weight");
    weights_.enc_conv4_bias   = get("conv4.bias");
    weights_.lstm_weight_ih   = get("lstm_cell.weight_ih");
    weights_.lstm_weight_hh   = get("lstm_cell.weight_hh");
    weights_.lstm_bias_ih     = get("lstm_cell.bias_ih");
    weights_.lstm_bias_hh     = get("lstm_cell.bias_hh");
    weights_.out_weight       = get("final_conv.weight");
    weights_.out_bias         = get("final_conv.bias");

    if (!weights_.stft_conv_weight || !weights_.lstm_weight_ih || !weights_.out_weight) {
        RS_LOG_ERR("Critical VAD tensors missing. Check GGUF conversion.");
        return false;
    }
    return true;
}

bool SileroVadModel::LoadHParams(gguf_context* ctx_gguf) {
    if (!ctx_gguf) return true;
    auto ki = [&](const char* k, int def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? (int)gguf_get_val_i32(ctx_gguf, key) : def;
    };
    auto kf = [&](const char* k, float def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? gguf_get_val_f32(ctx_gguf, key) : def;
    };
    hparams_.sample_rate         = ki("vad.sample_rate", 16000);
    hparams_.window_size_samples = ki("vad.window_size", 512);
    hparams_.threshold           = kf("vad.threshold", 0.5f);
    hparams_.min_silence_ms      = ki("vad.min_silence_ms", 100);
    hparams_.min_speech_ms       = ki("vad.min_speech_ms", 250);
    hparams_.context_samples     = ki("vad.context_size_samples", 64);
    return true;
}

bool SileroVadModel::LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                                ggml_backend_t backend) {
    if (!gguf_data) return false;
    backend_ = backend;
    LoadHParams(ctx_gguf);
    meta_.audio_sample_rate = hparams_.sample_rate;
    if (!MapTensors(gguf_data)) return false;
    RS_LOG_INFO("Silero VAD loaded: sr=%d, window=%d, threshold=%.2f",
                hparams_.sample_rate, hparams_.window_size_samples,
                hparams_.threshold);
    return true;
}

bool SileroVadModel::Load(const std::unique_ptr<rs_context_t>& ctx,
                          ggml_backend_t backend) {
    if (!ctx || !ctx->gguf_data) return false;
    return LoadDirect(ctx->gguf_data, ctx->ctx_gguf, backend);
}

std::shared_ptr<RSState> SileroVadModel::CreateState() {
    return std::make_shared<SileroVadState>();
}

// ============================================================
// Build computation graph for a single VAD frame
// Input tensors (set via ggml_backend_tensor_set before compute):
//   padded_audio  — F32 [640] (576 samples + 64 reflect-pad right)
//   in_lstm_h     — F32 [128] (LSTM hidden state)
//   in_lstm_c     — F32 [128] (LSTM cell state)
// Output tensors (read back after compute):
//   speech_prob   — F32 scalar
//   out_lstm_h    — F32 [128]
//   out_lstm_c    — F32 [128]
// ============================================================
static ggml_cgraph* silero_vad_build_graph(
        const SileroVadWeights& w,
        ggml_context* ctx0,
        ggml_tensor* padded_audio,
        ggml_tensor* h_in,
        ggml_tensor* c_in)
{
    // Cast conv weights to F16 (ggml_conv_1d -> im2col requires F16 kernel)
    auto f16 = [&](ggml_tensor* t) { return ggml_cast(ctx0, t, GGML_TYPE_F16); };

    // ---- STFT conv1d ----
    // Input: [640] -> reshape to [640, 1] for conv1d (IC=1)
    ggml_tensor* cur = ggml_reshape_2d(ctx0, padded_audio, padded_audio->ne[0], 1);
    cur = ggml_conv_1d(ctx0, f16(w.stft_conv_weight), cur, 128, 0, 1);
    // cur: ne=[OL=4, OC=258, N=1]

    // ---- Magnitude from STFT ----
    const int n_fft = (int)cur->ne[1] / 2;  // 129
    ggml_tensor* real_part = ggml_view_2d(ctx0, cur, cur->ne[0], n_fft, cur->nb[1], 0);
    ggml_tensor* imag_part = ggml_view_2d(ctx0, cur, cur->ne[0], n_fft, cur->nb[1],
                                          (size_t)n_fft * cur->nb[1]);
    cur = ggml_sqrt(ctx0,
        ggml_add(ctx0,
            ggml_mul(ctx0, real_part, real_part),
            ggml_mul(ctx0, imag_part, imag_part)));
    // cur: ne=[4, 129, 1]

    // ---- Helper: add 1D bias to conv1d output [OL, OC, N] ----
    auto add_bias = [&](ggml_tensor* out, ggml_tensor* bias) -> ggml_tensor* {
        ggml_tensor* b = ggml_reshape_2d(ctx0, bias, 1, bias->ne[0]);
        return ggml_add(ctx0, out, b);
    };

    // ---- Encoder: 4x Conv1d + ReLU ----
    cur = ggml_conv_1d(ctx0, f16(w.enc_conv1_weight), cur, 1, 1, 1);
    cur = add_bias(cur, w.enc_conv1_bias);
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_1d(ctx0, f16(w.enc_conv2_weight), cur, 2, 1, 1);
    cur = add_bias(cur, w.enc_conv2_bias);
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_1d(ctx0, f16(w.enc_conv3_weight), cur, 2, 1, 1);
    cur = add_bias(cur, w.enc_conv3_bias);
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_1d(ctx0, f16(w.enc_conv4_weight), cur, 1, 1, 1);
    cur = add_bias(cur, w.enc_conv4_bias);
    cur = ggml_relu(ctx0, cur);
    // cur: ne=[1, 128, 1]

    // ---- Squeeze: [1, 128, 1] -> [128, 1] ----
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[1], cur->ne[2]);

    // ---- LSTM cell ----
    ggml_tensor* gates = ggml_mul_mat(ctx0, w.lstm_weight_ih, cur);
    gates = ggml_add(ctx0, gates, w.lstm_bias_ih);
    ggml_tensor* hh = ggml_mul_mat(ctx0, w.lstm_weight_hh, h_in);
    hh = ggml_add(ctx0, hh, w.lstm_bias_hh);
    gates = ggml_add(ctx0, gates, hh);
    // gates: ne=[4H=512, 1]

    // Split gates via 1D views
    const int H = 128;
    const size_t stride = ggml_row_size(gates->type, H);
    ggml_tensor* i_gate = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 0 * stride));
    ggml_tensor* f_gate = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 1 * stride));
    ggml_tensor* g_gate = ggml_tanh(   ctx0, ggml_view_1d(ctx0, gates, H, 2 * stride));
    ggml_tensor* o_gate = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 3 * stride));

    // c' = f * c + i * g
    ggml_tensor* c_new = ggml_add(ctx0,
        ggml_mul(ctx0, f_gate, c_in),
        ggml_mul(ctx0, i_gate, g_gate));
    ggml_set_name(c_new, "out_lstm_c");
    ggml_set_output(c_new);

    // h' = o * tanh(c')
    ggml_tensor* h_new = ggml_mul(ctx0, o_gate, ggml_tanh(ctx0, c_new));
    ggml_set_name(h_new, "out_lstm_h");
    ggml_set_output(h_new);

    // ---- ReLU(h) ----
    cur = ggml_relu(ctx0, h_new);  // [128, 1]

    // ---- Unsqueeze: [128,1] -> [1,128] ----
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    // ---- Final conv1d: K=1, IC=128, OC=1 ----
    cur = ggml_conv_1d(ctx0, f16(w.out_weight), cur, 1, 0, 1);
    cur = ggml_add(ctx0, cur, w.out_bias);
    cur = ggml_reshape_1d(ctx0, cur, 1);
    cur = ggml_sigmoid(ctx0, cur);
    ggml_set_name(cur, "speech_prob");
    ggml_set_output(cur);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, VAD_MAX_NODES, false);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ============================================================
// Encode: one VAD forward pass
// ============================================================
bool SileroVadModel::Encode(const std::vector<float>& input_frames,
                            RSState& state,
                            ggml_backend_sched_t /*sched*/) {
    auto& vad_state = dynamic_cast<SileroVadState&>(state);

    if ((int)input_frames.size() < hparams_.window_size_samples) {
        RS_LOG_WARN("VAD: input too short (%zu < %d)", input_frames.size(), hparams_.window_size_samples);
        return false;
    }

    // ---- 1. Build padded input: context(64) + window(512) + pad-reflect(64) = 640 ----
    const int ctx_len = hparams_.context_samples;
    const int win_len = hparams_.window_size_samples;
    const int pad_right = 64;

    std::vector<float> padded(win_len + ctx_len + pad_right, 0.f);

    // Copy context
    std::copy(vad_state.context.begin(), vad_state.context.end(), padded.begin());
    // Copy window
    std::copy(input_frames.begin(), input_frames.begin() + win_len, padded.begin() + ctx_len);
    // Reflect-pad right
    for (int i = 0; i < pad_right; i++)
        padded[ctx_len + win_len + i] = padded[ctx_len + win_len - 1 - i];

    // ---- 2. Setup backend ----
    ggml_backend_t compute_backend = backend_;
    bool local_backend = false;
    if (!compute_backend) {
        compute_backend = ggml_backend_cpu_init();
        local_backend = true;
    }

    // ---- 3. Build graph ----
    size_t meta_size = ggml_tensor_overhead() * VAD_MAX_NODES + ggml_graph_overhead_custom(VAD_MAX_NODES, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params params = { meta_size, meta.data(), true };
    ggml_context* meta_ctx = ggml_init(params);

    ggml_tensor* tensor_padded = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, (int64_t)padded.size());
    ggml_set_name(tensor_padded, "padded_audio");
    ggml_tensor* tensor_h_in = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, 128);
    ggml_set_name(tensor_h_in, "in_lstm_h");
    ggml_tensor* tensor_c_in = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, 128);
    ggml_set_name(tensor_c_in, "in_lstm_c");

    ggml_cgraph* gf = silero_vad_build_graph(weights_, meta_ctx,
                                              tensor_padded, tensor_h_in, tensor_c_in);

    // ---- 4. Allocate and compute ----
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(compute_backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        RS_LOG_ERR("VAD: gallocr alloc failed");
        ggml_gallocr_free(alloc);
        ggml_free(meta_ctx);
        if (local_backend) ggml_backend_free(compute_backend);
        return false;
    }

    ggml_backend_tensor_set(tensor_padded, padded.data(), 0, padded.size() * sizeof(float));
    ggml_backend_tensor_set(tensor_h_in, vad_state.h_state.data(), 0, 128 * sizeof(float));
    ggml_backend_tensor_set(tensor_c_in, vad_state.h_state.data() + 128, 0, 128 * sizeof(float));

    if (ggml_backend_graph_compute(compute_backend, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("VAD: graph compute failed");
        ggml_gallocr_free(alloc);
        ggml_free(meta_ctx);
        if (local_backend) ggml_backend_free(compute_backend);
        return false;
    }

    // ---- 5. Read outputs ----
    ggml_tensor* prob_out = ggml_graph_get_tensor(gf, "speech_prob");
    float speech_prob = 0.f;
    ggml_backend_tensor_get(prob_out, &speech_prob, 0, sizeof(float));

    ggml_tensor* h_out = ggml_graph_get_tensor(gf, "out_lstm_h");
    ggml_tensor* c_out = ggml_graph_get_tensor(gf, "out_lstm_c");
    if (h_out && c_out) {
        ggml_backend_tensor_get(h_out, vad_state.h_state.data(), 0, 128 * sizeof(float));
        ggml_backend_tensor_get(c_out, vad_state.h_state.data() + 128, 0, 128 * sizeof(float));
    }

    vad_state.speech_prob = speech_prob;

    // ---- 6. Cleanup ----
    ggml_gallocr_free(alloc);
    ggml_free(meta_ctx);
    if (local_backend) ggml_backend_free(compute_backend);

    // ---- 7. Update context ----
    int ctx_start = win_len - ctx_len;
    std::copy(input_frames.begin() + ctx_start,
              input_frames.begin() + win_len,
              vad_state.context.begin());
    vad_state.current_sample += win_len;

    // ---- 8. VADIterator hysteresis ----
    float threshold = hparams_.threshold;
    float end_threshold = threshold - 0.15f;
    if (end_threshold < 0.01f) end_threshold = 0.01f;
    int min_silence_samples = (hparams_.min_silence_ms * hparams_.sample_rate) / 1000;

    if (speech_prob >= threshold && vad_state.temp_end_sample > 0) {
        vad_state.temp_end_sample = 0;
    }

    if (speech_prob >= threshold) {
        if (!vad_state.triggered) {
            vad_state.triggered = true;
        }
    }

    if (speech_prob < end_threshold && vad_state.triggered) {
        if (vad_state.temp_end_sample == 0) {
            vad_state.temp_end_sample = vad_state.current_sample;
        }
        if (vad_state.current_sample - vad_state.temp_end_sample >= min_silence_samples) {
            vad_state.triggered = false;
            vad_state.temp_end_sample = 0;
        }
    }

    return true;
}

bool SileroVadModel::Decode(RSState& /*state*/, ggml_backend_sched_t /*sched*/) {
    return true;
}

std::string SileroVadModel::GetTranscription(RSState& state) {
    auto& vad_state = dynamic_cast<SileroVadState&>(state);
    return vad_state.triggered ? "SPEECH" : "SILENCE";
}

float SileroVadModel::GetSpeechProbability(RSState& state) {
    return dynamic_cast<SileroVadState&>(state).speech_prob;
}

static struct SileroVadRegistrar {
    SileroVadRegistrar() {
        rs_register_model_arch("silero-vad", []() -> std::shared_ptr<ISpeechModel> {
            return std::make_shared<SileroVadModel>();
        });
    }
} s_silero_vad_registrar;
