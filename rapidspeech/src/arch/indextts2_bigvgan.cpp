// IndexTTS-2 BigVGAN-v2 vocoder forward path.
//
// Mirrors NVIDIA/BigVGAN bigvgan.BigVGAN.forward + alias_free_activation.torch
// (UpSample1d / DownSample1d / LowPassFilter1d / Activation1d) + SnakeBeta with
// alpha_logscale=True.
//
//   mel [n_mels, T] (innermost = T)
//       ↓ conv_pre (Conv1d 80→1536, K=7, pad=3)
//   x [1536, T]
//       ↓ stage_i (i=0..5):
//          x = ups[i][0](x)          (ConvTranspose1d, k/u from cfg)
//          xs = sum_j resblocks[3i+j](x)
//          x  = xs / 3
//   x [24, T*256]
//       ↓ activation_post (Activation1d ⟨SnakeBeta⟩)
//       ↓ conv_post (Conv1d 24→1, K=7, pad=3, no bias)
//   wav [T*256]
//
// AMPBlock1.forward (per resblock, K∈{3,7,11}, dilations=[1,3,5]):
//   acts1, acts2 = activations[::2], activations[1::2]
//   for c1,c2,a1,a2 in zip(convs1,convs2,acts1,acts2):
//       xt = a1(x); xt = c1(xt); xt = a2(xt); xt = c2(xt); x = xt + x
//
// Activation1d (anti-aliased nonlinearity):
//   y = upsample_2x(x)   (Kaiser-sinc FIR, replicate-pad 5/5, ConvT k=12 s=2,
//                         scale by 2, crop 15/15)
//   y = snakebeta(y, exp(alpha), exp(beta))
//   y = downsample_2x(y) (Kaiser-sinc FIR, replicate-pad 5/6, Conv k=12 s=2)
//
// SnakeBeta (alpha_logscale=True):
//   y = x + 1/exp(beta) * sin(exp(alpha) * x)**2
//
// Data layout: ne[0]=T (innermost), ne[1]=C, ne[2]=B (always 1 here).
// Weight layout (ggml convention):
//   Conv1d.weight          [K, Cin, Cout]
//   ConvTranspose1d.weight [K, Cout, Cin]
//   FIR filter             [K, 1, 1]  (shared across channels via expand)
//
// The Kaiser FIR is applied depthwise (groups=C). ggml has no depthwise
// conv_transpose_1d and the depthwise conv_1d takes a per-channel kernel, so
// we open-code both passes as a sum of `K` time-shifted scaled views — cheap
// (K=12) and avoids broadcasting a 1-tap filter to C copies.

#include "indextts2.h"

#include "core/rs_context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"
#ifdef RS_USE_METAL_BIGVGAN
#include "indextts2_bigvgan_metal.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace indextts2 {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void bv_log_stats(const char *name, const float *data, size_t n) {
    if (!data || n == 0) {
        RS_LOG_INFO("%s stats: empty", name);
        return;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    float mn = data[0];
    float mx = data[0];
    size_t finite = 0;
    for (size_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) continue;
        finite++;
        sum += v;
        sumsq += (double)v * (double)v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    const double denom = finite ? (double)finite : 1.0;
    RS_LOG_INFO("%s stats: n=%zu finite=%zu min=%.5f max=%.5f "
                "mean=%.5f rms=%.5f",
                name, n, finite, mn, mx, sum / denom,
                std::sqrt(sumsq / denom));
}

static ggml_tensor *bv_get(const BigVGANWeights &bv, const std::string &k) {
    auto it = bv.tensors.find(k);
    return it == bv.tensors.end() ? nullptr : it->second;
}

// Read an F32 1-D tensor of length n out of the backend, into a host buffer.
static std::vector<float> bv_read_f32_1d(ggml_tensor *t) {
    if (!t) return {};
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        if (t->type == GGML_TYPE_F16) {
            auto *src = reinterpret_cast<ggml_fp16_t *>(raw.data());
            for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
    return out;
}

// Plain Conv1d on layout [T, Cin, B] with weight [K, Cin, Cout].
// Pads with `pad` zeros on each end of the time axis before the conv.
static ggml_tensor *bv_conv1d(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                              ggml_tensor *x, int pad, int dilation) {
    ggml_tensor *y = ggml_conv_1d(ctx, w, x, /*s0=*/1, pad, dilation);
    if (b) {
        ggml_tensor *bv = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, bv);
    }
    return y;
}

// ConvTranspose1d on layout [T, Cin, B] with weight [K, Cout, Cin]. ggml only
// supports padding=0, so we run with p=0 and crop `pad` samples from each end
// of the time axis to match PyTorch's `padding=pad` semantics.
static ggml_tensor *bv_conv_transpose1d(ggml_context *ctx, ggml_tensor *w,
                                        ggml_tensor *b, ggml_tensor *x,
                                        int stride, int pad) {
    ggml_tensor *y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    const int64_t T_unpad = y->ne[0];
    const int64_t Cout    = y->ne[1];
    const int64_t B       = y->ne[2];
    if (pad > 0) {
        const int64_t T_out = T_unpad - 2 * pad;
        y = ggml_view_3d(ctx, y, T_out, Cout, B,
                         y->nb[1], y->nb[2],
                         (size_t)pad * y->nb[0]);
        y = ggml_cont(ctx, y);
    }
    if (b) {
        ggml_tensor *bv = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, bv);
    }
    return y;
}

// Replicate-pad along ne[0] (time). Output shape [T+lp+rp, C, B].
static ggml_tensor *bv_replicate_pad_T(ggml_context *ctx, ggml_tensor *x,
                                       int lp, int rp) {
    if (lp == 0 && rp == 0) return x;
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    const int64_t B = x->ne[2];

    ggml_tensor *first = ggml_view_3d(ctx, x, 1, C, B,
                                      x->nb[1], x->nb[2], 0);
    first = ggml_cont(ctx, first);
    ggml_tensor *last  = ggml_view_3d(ctx, x, 1, C, B,
                                      x->nb[1], x->nb[2],
                                      (size_t)(T - 1) * x->nb[0]);
    last = ggml_cont(ctx, last);

    ggml_tensor *y = x;
    if (lp > 0) {
        ggml_tensor *fL = ggml_repeat_4d(ctx, first, lp, C, B, 1);
        y = ggml_concat(ctx, fL, y, 0);
    }
    if (rp > 0) {
        ggml_tensor *lR = ggml_repeat_4d(ctx, last, rp, C, B, 1);
        y = ggml_concat(ctx, y, lR, 0);
    }
    return y;
}

// Build a 12-tap depthwise FIR (Kaiser-sinc) as a sum of time-shifted scaled
// views. PyTorch's conv1d/conv_transpose1d treat the kernel as channel-shared
// via `expand(C, -1, -1)` + groups=C, so the same `filter_f32[K]` is applied
// to every channel.
//
// Forward (DownSample1d): conv1d(filter, x, stride=2, groups=C) over a padded
// input of length L. Output length:  L_out = (L - K) / 2 + 1.
//   y[t,c] = sum_{k=0..K-1} filter[k] * x_pad[t*2 + k, c]
//
// Transpose (UpSample1d): conv_transpose1d(filter, x, stride=2, groups=C) over
// a padded input of length L. ggml output length: L_unpad = (L-1)*2 + K.
//   y[t,c] = sum_{(j,k) : j*2 + k = t} filter[k] * x[j, c]
// Equivalent to "zero-stuff x by inserting 1 zero between samples, then a
// stride-1 conv with the same filter".

static ggml_tensor *bv_fir_downsample_2x(ggml_context *ctx, ggml_tensor *x,
                                         const std::vector<float> &h, int K) {
    // x: [L, C, B]  (L may be odd — the upstream replicate-pad on DownSample1d
    //                makes it 2*L_act + 11, which is odd.)
    //   y[t, c] = sum_{k=0..K-1} h[k] * x[2t + k, c],  t in [0, T_out)
    //   T_out   = (L - K) / 2 + 1                          (PyTorch conv1d s=2)
    //
    // We reshape x into [2, L_even/2] rows so taps split by parity:
    //   k even (= 2m):  source row 0, column off + m
    //   k odd  (= 2m+1): source row 1, column off + m, but the absolute
    //                     index into the unreshaped buffer is 2*(col)+1.
    // The reshape requires L_even = L rounded UP to next even number. We
    // right-pad with one zero when L is odd; this never affects valid taps
    // (max index accessed is 2*(T_out-1) + (K-1) = L - 1).
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    const int64_t B = x->ne[2];
    const int64_t T_out = (L - K) / 2 + 1;

    ggml_tensor *xp = x;
    int64_t L_even = L;
    if (L % 2) {
        xp = ggml_pad_ext(ctx, x, /*lp0=*/0, /*rp0=*/1, 0,0, 0,0, 0,0);
        xp = ggml_cont(ctx, xp);
        L_even = L + 1;
    }
    ggml_tensor *x2 = ggml_reshape_4d(ctx, xp, 2, L_even / 2, C, B);

    ggml_tensor *acc = nullptr;
    for (int k = 0; k < K; ++k) {
        const int rk = k & 1;
        const int sk = k >> 1;
        ggml_tensor *row = ggml_view_4d(ctx, x2, 1, T_out, C, B,
                                        x2->nb[1], x2->nb[2], x2->nb[3],
                                        (size_t)rk * x2->nb[0] +
                                        (size_t)sk * x2->nb[1]);
        row = ggml_cont(ctx, row);
        row = ggml_reshape_3d(ctx, row, T_out, C, B);
        ggml_tensor *scaled = ggml_scale(ctx, row, h[k]);
        acc = (acc == nullptr) ? scaled : ggml_add(ctx, acc, scaled);
    }
    return acc;
}

static ggml_tensor *bv_fir_upsample_2x(ggml_context *ctx, ggml_tensor *x,
                                       const std::vector<float> &h, int K) {
    // x: [L, C, B]  → output length L_unpad = (L-1)*2 + K   (PyTorch
    //                  conv_transpose1d stride=2 padding=0 dilation=1)
    //
    // Equivalent formulation:
    //   stuffed[i] = x[i/2] if i even and i < 2L-1, else 0   (length 2L-1)
    //   stuffed_pad = [0]*(K-1)  ++  stuffed  ++  [0]*(K-1)  (length 2L+2K-3)
    //   y[t]       = sum_{k=0..K-1} h[k] * stuffed_pad[t + k]
    //                  for t in [0, L_unpad)
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    const int64_t B = x->ne[2];
    const int64_t L_unpad = (L - 1) * 2 + K;

    // 1. Zero-stuff to length 2L-1.
    //    reshape [L, C, B] → [1, L, C, B], interleave with a zeros tensor along
    //    ne0 → [2, L, C, B] (row 0 = original, row 1 = zero). Reshape → [2L,C,B],
    //    then drop the trailing zero with a view of length 2L-1.
    //    NOTE: we use ggml_concat (not ggml_pad_ext) here on purpose. The CUDA
    //    pad kernel maps ne1 → gridDim.y (max 65535); at activation_post L is the
    //    full audio length (>100k) and would overflow the grid ("PAD failed:
    //    invalid argument"). ggml_concat's contiguous path uses a flat
    //    grid-stride kernel with no such limit, and produces the identical
    //    interleaved buffer.
    ggml_tensor *x4    = ggml_reshape_4d(ctx, x, 1, L, C, B);
    ggml_tensor *zeros = ggml_scale(ctx, x4, 0.0f);
    ggml_tensor *stuffed = ggml_concat(ctx, x4, zeros, /*dim=*/0);
    stuffed = ggml_cont(ctx, stuffed);
    stuffed = ggml_reshape_3d(ctx, stuffed, 2 * L, C, B);
    stuffed = ggml_view_3d(ctx, stuffed, 2 * L - 1, C, B,
                           stuffed->nb[1], stuffed->nb[2], 0);
    stuffed = ggml_cont(ctx, stuffed);

    // 2. Pad K-1 on each side along time. Total length 2L + 2K - 3.
    stuffed = ggml_pad_ext(ctx, stuffed,
                           /*lp0=*/K - 1, /*rp0=*/K - 1,
                           0,0, 0,0, 0,0);
    stuffed = ggml_cont(ctx, stuffed);

    // 3. Sum of K shifted scaled views of length L_unpad.
    ggml_tensor *acc = nullptr;
    for (int k = 0; k < K; ++k) {
        ggml_tensor *shifted = ggml_view_3d(ctx, stuffed,
                                            L_unpad, C, B,
                                            stuffed->nb[1], stuffed->nb[2],
                                            (size_t)k * stuffed->nb[0]);
        shifted = ggml_cont(ctx, shifted);
        ggml_tensor *scaled = ggml_scale(ctx, shifted, h[k]);
        acc = (acc == nullptr) ? scaled : ggml_add(ctx, acc, scaled);
    }
    return acc;
}

// SnakeBeta with alpha_logscale=True:  y = x + 1/exp(beta) * sin(exp(alpha)*x)^2
// alpha, beta: [C]; x: [T, C, B].
static ggml_tensor *bv_snakebeta(ggml_context *ctx, ggml_tensor *x,
                                 ggml_tensor *alpha, ggml_tensor *beta) {
    ggml_tensor *a = ggml_reshape_3d(ctx, alpha, 1, alpha->ne[0], 1);
    ggml_tensor *b = ggml_reshape_3d(ctx, beta,  1, beta->ne[0],  1);
    ggml_tensor *ea = ggml_exp(ctx, a);
    ggml_tensor *eb = ggml_exp(ctx, b);
    ggml_tensor *s  = ggml_sin(ctx, ggml_mul(ctx, x, ea));
    s = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_div(ctx, s, eb));
}

// Activation1d: replicate-pad → upsample 2x (FIR, ×2) → snakebeta →
// replicate-pad → downsample 2x (FIR).
//
// Filter parameters for ratio=2, K=12 (BigVGAN default):
//   UpSample1d:   pad=5, pad_left=15, pad_right=15
//   DownSample1d: kernel_size=12, even → pad_left=5, pad_right=6
struct AAFilters {
    std::vector<float> up;   // length K
    std::vector<float> down; // length K
    int K = 12;
};

static ggml_tensor *bv_activation1d(ggml_context *ctx, ggml_tensor *x,
                                    ggml_tensor *alpha, ggml_tensor *beta,
                                    const AAFilters &f) {
    // UpSample1d
    ggml_tensor *y = bv_replicate_pad_T(ctx, x, /*lp=*/5, /*rp=*/5);
    y = bv_fir_upsample_2x(ctx, y, f.up, f.K);   // L_unpad = (L_in - 1)*2 + K
    y = ggml_scale(ctx, y, 2.0f);                // ratio = 2
    // Crop pad_left=15 / pad_right=15
    {
        const int64_t L_unpad = y->ne[0];
        const int64_t C = y->ne[1];
        const int64_t B = y->ne[2];
        const int pad_l = 15, pad_r = 15;
        const int64_t T_out = L_unpad - pad_l - pad_r;
        y = ggml_view_3d(ctx, y, T_out, C, B,
                         y->nb[1], y->nb[2],
                         (size_t)pad_l * y->nb[0]);
        y = ggml_cont(ctx, y);
    }
    // Nonlinearity
    y = bv_snakebeta(ctx, y, alpha, beta);
    // DownSample1d
    y = bv_replicate_pad_T(ctx, y, /*lp=*/5, /*rp=*/6);
    y = bv_fir_downsample_2x(ctx, y, f.down, f.K);
    return y;
}

// AMPBlock1.forward
static ggml_tensor *bv_amp_block(ggml_context *ctx, ggml_tensor *x,
                                 const BigVGANWeights &bv,
                                 const std::string &prefix, int K,
                                 const AAFilters &f) {
    static const int dilations[3] = {1, 3, 5};
    for (int idx = 0; idx < 3; ++idx) {
        const int d = dilations[idx];
        // activations[2*idx]
        ggml_tensor *a1_alpha = bv_get(bv, prefix + ".activations." +
                                          std::to_string(2 * idx) + ".act.alpha");
        ggml_tensor *a1_beta  = bv_get(bv, prefix + ".activations." +
                                          std::to_string(2 * idx) + ".act.beta");
        ggml_tensor *c1w = bv_get(bv, prefix + ".convs1." + std::to_string(idx) + ".weight");
        ggml_tensor *c1b = bv_get(bv, prefix + ".convs1." + std::to_string(idx) + ".bias");
        ggml_tensor *a2_alpha = bv_get(bv, prefix + ".activations." +
                                          std::to_string(2 * idx + 1) + ".act.alpha");
        ggml_tensor *a2_beta  = bv_get(bv, prefix + ".activations." +
                                          std::to_string(2 * idx + 1) + ".act.beta");
        ggml_tensor *c2w = bv_get(bv, prefix + ".convs2." + std::to_string(idx) + ".weight");
        ggml_tensor *c2b = bv_get(bv, prefix + ".convs2." + std::to_string(idx) + ".bias");

        ggml_tensor *xt = bv_activation1d(ctx, x, a1_alpha, a1_beta, f);
        // Conv1d k=K dilation=d, pad = d*(K-1)/2
        xt = bv_conv1d(ctx, c1w, c1b, xt, /*pad=*/d * (K - 1) / 2, /*dilation=*/d);
        xt = bv_activation1d(ctx, xt, a2_alpha, a2_beta, f);
        // Conv1d k=K dilation=1, pad = (K-1)/2
        xt = bv_conv1d(ctx, c2w, c2b, xt, /*pad=*/(K - 1) / 2, /*dilation=*/1);
        x  = ggml_add(ctx, xt, x);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Public entry — Model::RunBigVGAN
// ---------------------------------------------------------------------------
bool Model::RunBigVGAN(State &s, ggml_backend_sched_t sched,
                       const float *mel, int n_mels, int T_mel) {
    if (!bigvgan_ready_) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: weights not loaded");
        return false;
    }
    if (!mel || n_mels <= 0 || T_mel <= 0) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: invalid input mel (%p, %d, %d)",
                   mel, n_mels, T_mel);
        return false;
    }
    bv_log_stats("[indextts2] BigVGAN input mel", mel,
                 (size_t)n_mels * T_mel);

#ifdef RS_USE_METAL_BIGVGAN
    // Fused Metal fast-path (bypasses the ggml graph's ~467 CPU/Metal splits).
    // conv1d uses im2col + our own tiled-GEMM Metal kernel (fast and robust for
    // all sizes). RS_INDEXTTS2_BIGVGAN_NAIVE=1 forces the simple strided kernel.
    if (bigvgan_metal_ && bigvgan_metal_->is_valid() &&
        !std::getenv("RS_INDEXTTS2_BIGVGAN_GGML")) {
        const bool use_gemm = std::getenv("RS_INDEXTTS2_BIGVGAN_NAIVE") == nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        if (bigvgan_metal_->decode(mel, n_mels, T_mel, s.audio_output,
                                   /*use_mps=*/use_gemm)) {
            auto is_finite = [](const std::vector<float> &v) {
                if (v.empty()) return false;
                for (float x : v) if (!std::isfinite(x)) return false;
                return true;
            };
            bool ok = is_finite(s.audio_output);
            if (!ok && use_gemm) {
                RS_LOG_WARN("[indextts2] BigVGAN GEMM output non-finite — retrying "
                            "with naive Metal conv");
                bigvgan_metal_->decode(mel, n_mels, T_mel, s.audio_output,
                                       /*use_mps=*/false);
                ok = is_finite(s.audio_output);
            }
            if (ok) {
                const double sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                bv_log_stats("[indextts2] BigVGAN(Metal) output wav",
                             s.audio_output.data(), s.audio_output.size());
                RS_LOG_INFO("[indextts2] RunBigVGAN(Metal): %d×%d mel → %zu "
                            "samples in %.2f s", T_mel, n_mels,
                            s.audio_output.size(), sec);
                return true;
            }
            RS_LOG_WARN("[indextts2] BigVGAN Metal non-finite — ggml fallback");
            s.audio_output.clear();
        } else {
            RS_LOG_WARN("[indextts2] BigVGAN Metal decode failed — ggml fallback");
        }
    }
#endif

    // Pull the two Kaiser-sinc FIR filters off the backend once and cache them
    // in a stack-local struct. They are identical for every Activation1d (the
    // upstream PyTorch UpSample1d and LowPassFilter1d both compute the same
    // kaiser_sinc_filter1d at the same cutoff/half_width/kernel_size — only
    // the pad geometry differs at forward time). We pick the resblocks.0
    // copy because every Activation1d shares the same filter buffer.
    AAFilters f;
    {
        ggml_tensor *u = bv_get(bigvgan_, "resblocks.0.activations.0.upsample.filter");
        ggml_tensor *d = bv_get(bigvgan_, "resblocks.0.activations.0.downsample.lowpass.filter");
        if (!u || !d) {
            RS_LOG_ERR("[indextts2] RunBigVGAN: missing Kaiser FIR weights");
            return false;
        }
        f.up   = bv_read_f32_1d(u);
        f.down = bv_read_f32_1d(d);
        f.K    = (int)u->ne[0];
        if ((int)f.up.size() != f.K || (int)f.down.size() != f.K) {
            RS_LOG_ERR("[indextts2] RunBigVGAN: unexpected FIR shape K=%d "
                       "(up.size=%zu down.size=%zu)",
                       f.K, f.up.size(), f.down.size());
            return false;
        }
    }

    // --- Tensor map sanity ---
    ggml_tensor *conv_pre_w  = bv_get(bigvgan_, "conv_pre.weight");
    ggml_tensor *conv_pre_b  = bv_get(bigvgan_, "conv_pre.bias");
    ggml_tensor *conv_post_w = bv_get(bigvgan_, "conv_post.weight");
    ggml_tensor *conv_post_b = bv_get(bigvgan_, "conv_post.bias"); // may be null
    ggml_tensor *ap_alpha    = bv_get(bigvgan_, "activation_post.act.alpha");
    ggml_tensor *ap_beta     = bv_get(bigvgan_, "activation_post.act.beta");
    if (!conv_pre_w || !conv_pre_b || !conv_post_w || !ap_alpha || !ap_beta) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: missing core weights");
        return false;
    }

    const std::vector<int> &ur = hp_.bigvgan_upsample_rates;        // [4,4,2,2,2,2]
    const std::vector<int> &uk = hp_.bigvgan_upsample_kernel_sizes; // [8,8,4,4,4,4]
    const std::vector<int> &rk = hp_.bigvgan_resblock_kernel_sizes; // [3,7,11]
    const int num_upsamples = (int)ur.size();
    const int num_kernels   = (int)rk.size();

    // --- Build the graph ---
    // ≈109 Activation1d × ~30 nodes + 6 stages × convs + 3 resblocks × 6 convs.
    // The FIR helpers emit K=12 cont/scale/add per direction → ~80 nodes each.
    // Empirically the full graph is ~12-15k nodes; round up to 32k.
    const size_t n_nodes = 32768;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * n_nodes +
                              ggml_graph_overhead_custom(n_nodes, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    if (!ctx) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: ggml_init failed");
        return false;
    }
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // Input: mel [n_mels, T_mel] → ggml tensor with ne=[T_mel, n_mels, 1].
    // The reference dump is stored row-major as (n_mels, T_mel), so the
    // contiguous numpy buffer is identical to a ggml tensor with ne[0]=T_mel
    // (fastest = T) and ne[1]=n_mels.
    ggml_tensor *x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_mel, n_mels, 1);
    ggml_set_name(x_in, "mel_in");
    ggml_set_input(x_in);

    ggml_tensor *x = bv_conv1d(ctx, conv_pre_w, conv_pre_b, x_in,
                               /*pad=*/3, /*dilation=*/1);

    for (int i = 0; i < num_upsamples; ++i) {
        // ups[i][0] — ConvTranspose1d, stride=ur[i], k=uk[i], padding=(k-u)/2
        const int u = ur[i];
        const int k = uk[i];
        const int pad = (k - u) / 2;
        ggml_tensor *ups_w = bv_get(bigvgan_, "ups." + std::to_string(i) + ".0.weight");
        ggml_tensor *ups_b = bv_get(bigvgan_, "ups." + std::to_string(i) + ".0.bias");
        if (!ups_w || !ups_b) {
            RS_LOG_ERR("[indextts2] RunBigVGAN: missing ups.%d.0", i);
            ggml_free(ctx);
            return false;
        }
        x = bv_conv_transpose1d(ctx, ups_w, ups_b, x, /*stride=*/u, /*pad=*/pad);

        // Sum 3 resblocks, divide by 3.
        ggml_tensor *xs = nullptr;
        for (int j = 0; j < num_kernels; ++j) {
            const std::string pfx = "resblocks." +
                                    std::to_string(i * num_kernels + j);
            ggml_tensor *r = bv_amp_block(ctx, x, bigvgan_, pfx, rk[j], f);
            xs = (xs == nullptr) ? r : ggml_add(ctx, xs, r);
        }
        x = ggml_scale(ctx, xs, 1.0f / (float)num_kernels);
    }

    // activation_post + conv_post
    x = bv_activation1d(ctx, x, ap_alpha, ap_beta, f);
    x = bv_conv1d(ctx, conv_post_w, conv_post_b, x, /*pad=*/3, /*dilation=*/1);
    if (hp_.bigvgan_use_tanh_at_final) {
        x = ggml_tanh(ctx, x);
    }
    ggml_set_name(x, "bigvgan_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // --- Allocate + compute ---
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: sched alloc failed");
        ggml_free(ctx);
        return false;
    }

    // Upload the mel input.
    ggml_backend_tensor_set(x_in, mel, 0,
                            (size_t)T_mel * n_mels * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] RunBigVGAN: compute failed");
        ggml_free(ctx);
        return false;
    }

    // Output shape: [T_mel * 256, 1, 1] for BigVGAN-v2 (256x upsampling).
    const int64_t n_out = ggml_nelements(x);
    s.audio_output.assign(n_out, 0.0f);
    ggml_backend_tensor_get(x, s.audio_output.data(), 0,
                            n_out * sizeof(float));
    bv_log_stats("[indextts2] BigVGAN output wav", s.audio_output.data(),
                 s.audio_output.size());

    RS_LOG_INFO("[indextts2] RunBigVGAN: %d×%d mel → %lld samples",
                T_mel, n_mels, (long long)n_out);
    ggml_free(ctx);
    return true;
}

} // namespace indextts2
