#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations to keep this header ObjC-free.
struct ggml_tensor;

namespace indextts2 {

struct BigVGANWeights;
struct HParams;

// -----------------------------------------------------------------------------
// Custom Metal GPU backend for the IndexTTS-2 BigVGAN-v2 vocoder.
//
// The ggml-graph path builds ~12-15k nodes and the Metal scheduler fragments it
// into ~467 CPU/Metal splits (the anti-aliased Snake activations pad/cont/view a
// lot), so a single vocoder forward costs ~36-54s — dominated by CPU<->GPU sync,
// not compute.  This backend bypasses ggml entirely: it encodes the whole
// forward into ONE command buffer with memory barriers between dependent
// dispatches, so there is exactly one CPU sync for the whole vocoder.
//
// Kernels (T-major layout throughout: element(t,c) at t + c*T):
//   bv_conv1d, bv_conv_transpose1d, bv_aa_up_snake (FIR-up2x + snakebeta),
//   bv_aa_down (FIR-down2x), bv_add_into (residual), bv_axpy3 (sum/3).
// -----------------------------------------------------------------------------
class BigVGANMetalDecoder {
public:
    BigVGANMetalDecoder();
    ~BigVGANMetalDecoder();

    // One-time init: compile shaders, create pipelines, upload weights.
    bool init(const BigVGANWeights &bv, const HParams &hp);

    // Full BigVGAN forward: mel [n_mels, T_mel] (row-major, T fastest) → audio.
    // use_mps: conv1d via im2col+MPS GEMM (fast). Set false to use the naive
    // Metal conv kernel (slower but robust) — used as a fallback if MPS emits NaN.
    bool decode(const float *mel, int n_mels, int T_mel,
                std::vector<float> &audio_out, bool use_mps = true);

    bool is_valid() const { return valid_; }

private:
    bool valid_ = false;

    // Metal objects (void* to keep header ObjC-free).
    void *device_ = nullptr;
    void *library_ = nullptr;
    void *queue_ = nullptr;
    void *pipe_conv1d_ = nullptr;
    void *pipe_conv1d_simd_ = nullptr;
    void *pipe_conv_t_ = nullptr;
    void *pipe_im2col_ = nullptr;
    void *pipe_add_bias_ = nullptr;
    void *pipe_gemm_ = nullptr;
    void *pipe_up_snake_ = nullptr;
    void *pipe_down_ = nullptr;
    void *pipe_add_into_ = nullptr;
    void *pipe_axpy3_ = nullptr;

    // Anti-aliasing FIR filters (shared by every Activation1d), K taps each.
    void *buf_fir_up_ = nullptr;
    void *buf_fir_down_ = nullptr;
    int   fir_K_ = 12;

    struct Conv {           // Conv1d / ConvTranspose1d weight+bias on GPU
        void *w = nullptr, *b = nullptr;
        void *w_mps = nullptr;           // [OC, wmps_row] row-major (im2col+MPS GEMM)
        int wmps_row = 0;                // padded KIC row stride (floats, 16B-aligned)
        int IC = 0, OC = 0, K = 0, stride = 1, pad = 0, dilation = 1;
    };
    struct Snake {          // precomputed exp(alpha), exp(-beta) per channel
        void *a_exp = nullptr, *b_inv = nullptr;
        int C = 0;
    };
    struct ResBlock {       // AMPBlock1: 3 (act→conv1→act→conv2) with dilations
        int K = 0;                       // 3 / 7 / 11
        Snake a1[3], a2[3];              // activations[2*idx], [2*idx+1]
        Conv  c1[3], c2[3];              // convs1[idx] (dilated), convs2[idx]
    };
    struct UpStage {
        Conv ct;                         // ups[i][0] ConvTranspose1d
        ResBlock rb[3];                  // resblocks[i*3 + 0..2]
    };

    Conv conv_pre_;
    std::vector<UpStage> stages_;
    Snake act_post_;
    Conv conv_post_;
    bool use_tanh_final_ = false;

    // Helpers.
    void *get_fn(const char *name);
    void *alloc_buffer(size_t bytes, const void *data = nullptr);
    static std::vector<float> read_host(const ggml_tensor *t);
    void *upload_vec(const std::vector<float> &v);
    void *upload_conv_w(const ggml_tensor *t, int K, int IC, int OC);
    void *upload_conv_w_mps(const ggml_tensor *t, int K, int IC, int OC);
    void *upload_conv_t_w(const ggml_tensor *t, int K, int IC, int OC);
    bool  load_snake(const BigVGANWeights &bv, const std::string &alpha_key,
                     const std::string &beta_key, Snake &out);
    bool  load_conv(const BigVGANWeights &bv, const std::string &wkey,
                    const std::string &bkey, int K, Conv &out, bool transpose);
};

} // namespace indextts2
