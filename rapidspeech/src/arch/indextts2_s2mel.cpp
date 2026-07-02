// IndexTTS-2 S2Mel forward path: lm_latent → mel-spectrogram.
//
// Mirrors upstream indextts/s2mel/modules/flow_matching.py +
// length_regulator.py + diffusion_transformer.py.
//
//   lm_latent [T_mel, 1280]                 ← from RunAR second pass
//        ↓ gpt_layer (1280 → 256 → 128 → 1024, NO activations)
//   gpt_emb [T_mel, 1024]
//        ↓ + S_infer (residual from semantic codec)
//        ↓ length_regulator
//   cond [T_out, 512]
//        ↓ cat([prompt_cond, cond])
//        ↓ CFM Euler solver (n_timesteps steps, classifier-free guidance):
//          z = randn(0, 1) * temperature
//          z[:, :T_prompt, :] = prompt          (prompt-on-z conditioning)
//          for t,dt in t_span:
//              dphi      = DiT(z, prompt, x_lens, t, style, cond)
//              cfg_dphi  = DiT(z, prompt, x_lens, t, style, zeros)
//              dphi_dt   = (1+cfg)*dphi - cfg*cfg_dphi
//              z += dt * dphi_dt
//   mel [T_total, n_mels=80], cropped to [T_out, n_mels]
//
// **Status (this CL):**
//   - gpt_layer:        bit-exact (3 chained Linears, no activations)
//   - length_regulator: bit-exact (content_in_proj → nearest interp → 4×
//                       [Conv1d/GN/Mish] + final 1×1 Conv)
//   - DiT estimator:    wired through diffusion_transformer + WaveNet head
//   - CFM Euler:        production path uses the real DiT estimator; set
//                       RS_INDEXTTS2_S2MEL_NSTEPS to shorten debug runs.
//
// Smoke harness: set RS_INDEXTTS2_S2MEL_TEST_DIR to a directory containing
// the .npy files produced by scripts/dump_s2mel_ref.py to bypass RunAR
// entirely. The wrapper dumps gpt_layer_out.f32 / lr_out.f32 / mel_out.f32
// for off-line diff against the PyTorch reference.

#include "indextts2.h"

#include "arch/semantic_codec.h"
#include "core/rs_context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace indextts2 {

extern "C" {
void rdft(int n, int isgn, double *a, int *ip, double *w);
}

namespace {

static void log_vec_stats(const char *name, const float *data, size_t n) {
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

inline double hz_to_mel_slaney(double hz) {
    const double f_sp        = 200.0 / 3.0;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    return hz >= min_log_hz
        ? min_log_mel + std::log(hz / min_log_hz) / logstep
        : hz / f_sp;
}

inline double mel_to_hz_slaney(double mel) {
    const double f_sp        = 200.0 / 3.0;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    return mel >= min_log_mel
        ? min_log_hz * std::exp(logstep * (mel - min_log_mel))
        : mel * f_sp;
}

inline int reflect_index(int idx, int n) {
    if (n <= 1) return 0;
    const int period = 2 * (n - 1);
    int m = ((idx % period) + period) % period;
    return m >= n ? period - m : m;
}

static int compute_indextts2_ref_mel(const HParams &hp,
                                      const std::vector<float> &pcm,
                                      std::vector<float> &out_TC) {
    out_TC.clear();
    if (pcm.empty()) return 0;

    const int sr     = hp.s2mel_sr;
    const int n_fft  = hp.s2mel_n_fft;
    const int hop    = hp.s2mel_hop;
    const int win    = hp.s2mel_win;
    const int n_mels = hp.s2mel_n_mels;
    const int n_bins = n_fft / 2 + 1;
    const int pad    = (n_fft - hop) / 2;
    if (sr <= 0 || n_fft <= 0 || hop <= 0 || win <= 0 ||
        n_mels <= 0 || pad < 0) {
        return 0;
    }

    std::vector<float> padded((size_t)pcm.size() + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        padded[i] = pcm[(size_t)reflect_index(i - pad, (int)pcm.size())];
    }
    std::memcpy(padded.data() + pad, pcm.data(), pcm.size() * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        padded[(size_t)pad + pcm.size() + i] =
            pcm[(size_t)reflect_index((int)pcm.size() + i, (int)pcm.size())];
    }

    const int n_padded = (int)padded.size();
    if (n_padded < n_fft) return 0;
    const int n_frames = (n_padded - n_fft) / hop + 1;
    if (n_frames <= 0) return 0;

    std::vector<double> hann(n_fft, 0.0);
    for (int i = 0; i < win; ++i) {
        hann[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (double)win);
    }

    std::vector<float> mel_fb((size_t)n_mels * n_bins, 0.0f);
    const double mel_lo = hz_to_mel_slaney(0.0);
    const double mel_hi = hz_to_mel_slaney((double)sr / 2.0);
    std::vector<double> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_lo + (mel_hi - mel_lo) * i / (double)(n_mels + 1);
    }
    std::vector<double> hz_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) hz_pts[i] = mel_to_hz_slaney(mel_pts[i]);
    const double fft_bin_hz = (double)sr / (double)n_fft;
    for (int m = 0; m < n_mels; ++m) {
        const double fl = hz_pts[m], fc = hz_pts[m + 1], fr = hz_pts[m + 2];
        const double enorm = 2.0 / (fr - fl);
        for (int k = 0; k < n_bins; ++k) {
            const double hz   = fft_bin_hz * k;
            const double up   = (hz - fl) / (fc - fl);
            const double down = (fr - hz) / (fr - fc);
            mel_fb[(size_t)m * n_bins + k] =
                (float)(std::max(0.0, std::min(up, down)) * enorm);
        }
    }

    std::vector<int> fft_ip(2 + (int)std::sqrt((double)n_fft / 2) + 1, 0);
    std::vector<double> fft_w(n_fft / 2, 0.0);
    std::vector<double> dummy(n_fft, 0.0);
    rdft(n_fft, 1, dummy.data(), fft_ip.data(), fft_w.data());

    out_TC.assign((size_t)n_frames * n_mels, 0.0f);
    std::vector<double> buf(n_fft);
    std::vector<double> mag(n_bins);
    for (int t = 0; t < n_frames; ++t) {
        const int off = t * hop;
        for (int j = 0; j < n_fft; ++j) {
            buf[j] = (double)padded[(size_t)off + j] * hann[j];
        }
        rdft(n_fft, 1, buf.data(), fft_ip.data(), fft_w.data());

        mag[0] = std::sqrt(buf[0] * buf[0] + 1e-9);
        mag[n_bins - 1] = std::sqrt(buf[1] * buf[1] + 1e-9);
        for (int k = 1; k < n_bins - 1; ++k) {
            const double re = buf[2 * k];
            const double im = buf[2 * k + 1];
            mag[k] = std::sqrt(re * re + im * im + 1e-9);
        }
        for (int m = 0; m < n_mels; ++m) {
            const float *row = &mel_fb[(size_t)m * n_bins];
            double acc = 0.0;
            for (int k = 0; k < n_bins; ++k) acc += (double)row[k] * mag[k];
            out_TC[(size_t)t * n_mels + m] = (float)std::log(std::max(acc, 1e-5));
        }
    }
    return n_frames;
}

} // namespace

// -----------------------------------------------------------------------------
// Small helpers (duplicated from indextts2_gpt.cpp on purpose — these are tiny
// and we want both TUs to be self-contained).
// -----------------------------------------------------------------------------
static ggml_tensor *try_t(const std::unordered_map<std::string, ggml_tensor *> &m,
                          const std::string &k) {
    auto it = m.find(k);
    return (it == m.end()) ? nullptr : it->second;
}

static ggml_tensor *linear(ggml_context *ctx, ggml_tensor *x,
                           ggml_tensor *w, ggml_tensor *b) {
    if (!w) return x;
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// -----------------------------------------------------------------------------
// gpt_layer: a 3-layer MLP that projects lm_latent from GPT hidden dim down to
// the S2Mel encoder dim. Upstream layout
//   (indextts/s2mel/modules/commons.py:MyModel.__init__):
//     'gpt_layer': nn.Sequential(
//          nn.Linear(1280, 256),
//          nn.Linear(256, 128),
//          nn.Linear(128, 1024))
// **No** activations between linears — verified against upstream source.
// -----------------------------------------------------------------------------
static bool run_gpt_layer(ggml_backend_sched_t sched, ggml_backend_t backend,
                          const S2MelWeights &w, const HParams &hp,
                          const std::vector<float> &lm_latent, int T,
                          std::vector<float> &out_1024) {
    (void)backend;
    if (!w.gpt_layer0_w || !w.gpt_layer1_w || !w.gpt_layer2_w) {
        out_1024.assign((size_t)T * hp.s2mel_lr_in_channels, 0.0f);
        return true;
    }
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 256 +
                              ggml_graph_overhead_custom(256, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 256, false);

    const int D_in = hp.n_embd;
    ggml_tensor *x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_in, T);
    ggml_set_name(x_in, "lm_latent");
    ggml_set_input(x_in);

    ggml_tensor *x = linear(ctx, x_in, w.gpt_layer0_w, w.gpt_layer0_b);
    x = linear(ctx, x, w.gpt_layer1_w, w.gpt_layer1_b);
    x = linear(ctx, x, w.gpt_layer2_w, w.gpt_layer2_b);
    ggml_set_name(x, "gpt_layer_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] gpt_layer alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, lm_latent.data(), 0,
                            lm_latent.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] gpt_layer compute failed");
        ggml_free(ctx);
        return false;
    }
    out_1024.assign((size_t)T * hp.s2mel_lr_in_channels, 0.0f);
    ggml_backend_tensor_get(x, out_1024.data(), 0,
                            out_1024.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

// -----------------------------------------------------------------------------
// length_regulator (InterpolateRegulator) forward (continuous, sampling_ratios
// = [1,1,1,1], no f0, no vq, no extra codebooks). Upstream
// (length_regulator.py:90):
//
//   x = self.content_in_proj(x)              # (B, T_in, 1024) → (B, T_in, 512)
//   x = F.interpolate(x.T, size=T_out, mode='nearest')   # (B, 512, T_out)
//   for k in [0,1,2,3]:                       # sampling_ratios stack
//       x = self.model[3k+0](x)               # Conv1d(512,512,3,pad=1)
//       x = self.model[3k+1](x)               # GroupNorm(num_groups=1, 512)
//       x = self.model[3k+2](x)               # Mish (functional)
//   x = self.model[12](x)                     # Conv1d(512,512,1)
//   out = x.transpose(1, 2) * mask            # (B, T_out, 512)
//
// We do the linear + nearest-interp on the host (one mul_mat graph + a memcpy
// gather), then build a second ggml graph for the model stack on the GPU /
// backend. Splitting graphs keeps GPU allocations small and lets us validate
// each stage independently against the reference dump.
// -----------------------------------------------------------------------------
static bool run_content_in_proj(ggml_backend_sched_t sched,
                                const S2MelWeights &w, const HParams &hp,
                                const std::vector<float> &gpt_emb_1024,
                                int T_in,
                                std::vector<float> &x_512) {
    ggml_tensor *ww = try_t(w.length_regulator, "content_in_proj.weight");
    ggml_tensor *bb = try_t(w.length_regulator, "content_in_proj.bias");
    const int C_in  = hp.s2mel_lr_in_channels; // 1024
    const int C_out = hp.s2mel_lr_channels;    // 512
    x_512.assign((size_t)T_in * C_out, 0.0f);
    if (!ww) {
        // No projection bound — degrade to a 1024→512 truncation.
        for (int t = 0; t < T_in; ++t)
            std::memcpy(&x_512[(size_t)t * C_out],
                        &gpt_emb_1024[(size_t)t * C_in],
                        C_out * sizeof(float));
        return true;
    }

    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 +
                              ggml_graph_overhead_custom(64, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor *x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C_in, T_in);
    ggml_set_name(x_in, "cip_in");
    ggml_set_input(x_in);
    ggml_tensor *y = linear(ctx, x_in, ww, bb);
    ggml_set_name(y, "cip_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] content_in_proj alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, gpt_emb_1024.data(), 0,
                            gpt_emb_1024.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] content_in_proj compute failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(y, x_512.data(), 0, x_512.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

// Host-side nearest-neighbour interpolation matching PyTorch F.interpolate
// mode='nearest' for 1D inputs. Maps out_idx i → in_idx = floor(i * T_in / T_out).
// Input layout: (T_in, C). Output layout: (T_out, C).
static void nearest_interp_1d(const std::vector<float> &x_in,
                              int T_in, int C, int T_out,
                              std::vector<float> &x_out) {
    x_out.assign((size_t)T_out * C, 0.0f);
    if (T_in <= 0 || T_out <= 0) return;
    for (int t = 0; t < T_out; ++t) {
        int64_t src = (int64_t)t * (int64_t)T_in / (int64_t)T_out;
        if (src >= T_in) src = T_in - 1;
        if (src < 0) src = 0;
        std::memcpy(&x_out[(size_t)t * C], &x_in[(size_t)src * C],
                    (size_t)C * sizeof(float));
    }
}

// Transpose a (rows, cols) row-major buffer into (cols, rows) row-major.
// We use this at the conv-stack boundary because mul_mat outputs (T, C) with
// C-fast memory, but ggml_conv_1d wants (T, C, B) with T-fast memory — i.e.
// the same bytes laid out as (C, T) row-major.
static void transpose_2d(const std::vector<float> &in, int rows, int cols,
                          std::vector<float> &out) {
    out.assign((size_t)rows * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float *src_row = &in[(size_t)r * cols];
        for (int c = 0; c < cols; ++c) {
            out[(size_t)c * rows + r] = src_row[c];
        }
    }
}

// GroupNorm(num_groups=1, num_channels=C) over a (T, C) layout (ne[0]=T, ne[1]=C).
// PyTorch normalises across the (C × T) elements per batch element, then
// applies per-channel affine transform.
//
// ggml_norm normalises across ne[0]. To get "all C×T together as one stat",
// we reshape (T, C) → flat 1D of length T*C, run ggml_norm there, then
// reshape back. The affine is applied per-channel afterwards (broadcast over
// T). The result is bit-equivalent to F.group_norm with 1 group.
static ggml_tensor *group_norm1(ggml_context *ctx, ggml_tensor *x_TC,
                                 ggml_tensor *gamma_C, ggml_tensor *beta_C,
                                 float eps) {
    const int64_t T = x_TC->ne[0];
    const int64_t C = x_TC->ne[1];
    const int64_t B = x_TC->ne[2];
    ggml_tensor *flat = ggml_reshape_3d(ctx, x_TC, T * C, 1, B);
    flat = ggml_norm(ctx, flat, eps);
    ggml_tensor *y = ggml_reshape_3d(ctx, flat, T, C, B);
    if (gamma_C) {
        // gamma/beta are (C,) — broadcast over T via reshape to (1, C, 1).
        ggml_tensor *g = ggml_reshape_3d(ctx, gamma_C, 1, C, 1);
        y = ggml_mul(ctx, y, g);
    }
    if (beta_C) {
        ggml_tensor *bt = ggml_reshape_3d(ctx, beta_C, 1, C, 1);
        y = ggml_add(ctx, y, bt);
    }
    return y;
}

// Mish: y = x * tanh(softplus(x)).
static ggml_tensor *mish(ggml_context *ctx, ggml_tensor *x) {
    return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
}

// Helper to fetch a Conv1d kernel cast to f16 if needed for ggml_conv_1d's
// im2col path.
static ggml_tensor *w_f16(ggml_context *ctx, ggml_tensor *w) {
    if (w->type == GGML_TYPE_F16) return w;
    return ggml_cast(ctx, w, GGML_TYPE_F16);
}

static bool run_lr_model_stack(ggml_backend_sched_t sched,
                                const S2MelWeights &w, const HParams &hp,
                                const std::vector<float> &x_in_512, int T_out,
                                std::vector<float> &cond_512) {
    const int C = hp.s2mel_lr_channels; // 512
    cond_512.assign((size_t)T_out * C, 0.0f);

    // Need: model.{0,3,6,9}.weight,bias  (Conv1d k=3,pad=1)
    //       model.{1,4,7,10}.weight,bias (GroupNorm gamma/beta)
    //       model.12.weight,bias         (Conv1d k=1)
    struct Stage { int conv_idx, gn_idx; };
    const Stage stages[] = {{0,1}, {3,4}, {6,7}, {9,10}};

    ggml_tensor *convs_w[4] = {0}, *convs_b[4] = {0};
    ggml_tensor *gns_g[4]  = {0}, *gns_b[4]  = {0};
    for (int i = 0; i < 4; ++i) {
        convs_w[i] = try_t(w.length_regulator,
                            "model." + std::to_string(stages[i].conv_idx) + ".weight");
        convs_b[i] = try_t(w.length_regulator,
                            "model." + std::to_string(stages[i].conv_idx) + ".bias");
        gns_g[i]   = try_t(w.length_regulator,
                            "model." + std::to_string(stages[i].gn_idx) + ".weight");
        gns_b[i]   = try_t(w.length_regulator,
                            "model." + std::to_string(stages[i].gn_idx) + ".bias");
        if (!convs_w[i] || !gns_g[i]) {
            RS_LOG_WARN("[indextts2] length_regulator stage %d weights missing — "
                        "passing through", i);
            cond_512 = x_in_512;
            return true;
        }
    }
    ggml_tensor *final_w = try_t(w.length_regulator, "model.12.weight");
    ggml_tensor *final_b = try_t(w.length_regulator, "model.12.bias");
    if (!final_w) {
        RS_LOG_WARN("[indextts2] length_regulator final 1×1 conv missing — "
                    "passing through");
        cond_512 = x_in_512;
        return true;
    }

    const int n_nodes = 4096;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * n_nodes +
                              ggml_graph_overhead_custom(n_nodes, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // Input layout for ggml_conv_1d: (T_out, C, B=1)  ne[0]=T_out, ne[1]=C
    ggml_tensor *x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T_out, C, 1);
    ggml_set_name(x_in, "lr_model_in");
    ggml_set_input(x_in);

    ggml_tensor *x = x_in;
    for (int i = 0; i < 4; ++i) {
        // Conv1d(C, C, k=3, stride=1, padding=1, dilation=1)
        ggml_tensor *y = ggml_conv_1d(ctx, w_f16(ctx, convs_w[i]), x,
                                       /*s=*/1, /*p=*/1, /*d=*/1);
        // Add bias: bias is (C,) → broadcast over T.
        // ggml_conv_1d returns (T_out, C, B); bias add via reshape (1, C, 1).
        ggml_tensor *bias_r = ggml_reshape_3d(ctx, convs_b[i], 1, C, 1);
        y = ggml_add(ctx, y, bias_r);
        // GroupNorm(1, C)
        y = group_norm1(ctx, y, gns_g[i], gns_b[i], 1e-5f);
        // Mish
        y = mish(ctx, y);
        x = y;
    }
    // Final 1×1 Conv1d
    ggml_tensor *y = ggml_conv_1d(ctx, w_f16(ctx, final_w), x,
                                   /*s=*/1, /*p=*/0, /*d=*/1);
    if (final_b) {
        ggml_tensor *bias_r = ggml_reshape_3d(ctx, final_b, 1, C, 1);
        y = ggml_add(ctx, y, bias_r);
    }
    ggml_set_name(y, "lr_model_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[indextts2] lr_model alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, x_in_512.data(), 0,
                            x_in_512.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[indextts2] lr_model compute failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(y, cond_512.data(), 0, cond_512.size() * sizeof(float));
    ggml_free(ctx);
    return true;
}

// Returns cond_512 in (T_out, 512) row-major layout (C-fast, numpy-compatible).
static bool run_length_regulator(ggml_backend_sched_t sched,
                                  const S2MelWeights &w, const HParams &hp,
                                  const std::vector<float> &x_in_1024,
                                  int T_in, int T_out,
                                  std::vector<float> &cond_512) {
    const int C_out = hp.s2mel_lr_channels;    // 512

    // 1) content_in_proj: 1024 → 512  (output is (T_in, 512) C-fast)
    std::vector<float> xc(T_in * C_out);
    if (!run_content_in_proj(sched, w, hp, x_in_1024, T_in, xc)) return false;

    // 2) Host-side nearest-neighbour interpolation T_in → T_out, still C-fast.
    std::vector<float> x_up;
    nearest_interp_1d(xc, T_in, C_out, T_out, x_up);

    // 3) Transpose into T-fast layout for the ggml conv stack. ggml_conv_1d
    //    expects (T, C, B=1) with ne[0]=T (i.e. T fastest in memory), which is
    //    the same byte pattern as (C, T_out) C-fast.
    std::vector<float> x_up_tfast;
    transpose_2d(x_up, T_out, C_out, x_up_tfast);

    // 4) Conv1d / GroupNorm / Mish stack + final 1×1.
    std::vector<float> cond_tfast;
    if (!run_lr_model_stack(sched, w, hp, x_up_tfast, T_out, cond_tfast))
        return false;

    // 5) Transpose back to (T_out, C_out) C-fast for numpy comparison.
    //    cond_tfast is (T_out, C_out) T-fast == (C_out, T_out) C-fast, so we
    //    transpose a (C_out, T_out) matrix into (T_out, C_out).
    transpose_2d(cond_tfast, C_out, T_out, cond_512);
    return true;
}

// -----------------------------------------------------------------------------
// DiT estimator forward (single batch). One forward = one Euler step pass.
//
// Inputs (host, f32):
//   x         [T_total, 80]   — noisy mel (z) at current timestep
//   prompt_x  [T_total, 80]   — prompt mel pasted at the head, zeros elsewhere
//   style     [192]
//   mu        [T_total, 512]  — output of length_regulator + prompt_condition
//   t                          — scalar timestep ∈ [0, 1]
//
// Output (host, f32): dphi [T_total, 80]
//
// Internally we lay tensors out time-fast: ggml tensors created with
// `ne[0] = inner_dim` (so e.g. (T_total, 512) means ne[0]=512, ne[1]=T_total,
// which matches numpy's (T, 512) C-order). This is the same convention we use
// in the LR `content_in_proj` graph above. The wavenet head, which operates
// in `(C, T)` PyTorch layout, needs an explicit transpose at its boundary —
// same caveat as the LR stack.
// -----------------------------------------------------------------------------

namespace {

struct DiTInputs {
    const float *x;        // (T, 80)  C-fast
    const float *prompt_x; // (T, 80)  C-fast
    const float *style;    // (192,)
    const float *mu;       // (T, 512) C-fast
    float t;
    int T;
};

// Build a (T, D) tensor in ggml-T-fast layout. The caller will memcpy a host
// buffer that is "(T, D) C-fast == (D, T) row-major" — i.e. T-fast in memory.
inline ggml_tensor *new_TD(ggml_context *ctx, int D, int T) {
    return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, 1);
}

// Apply standard nn.Linear (W in_features→out_features stored as ggml mul_mat
// kernel with ne[0]=in, ne[1]=out). Input x: ne[0]=in. Output ne[0]=out.
inline ggml_tensor *nn_linear(ggml_context *ctx, ggml_tensor *x,
                              ggml_tensor *w, ggml_tensor *b) {
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// SiLU
inline ggml_tensor *silu(ggml_context *ctx, ggml_tensor *x) {
    return ggml_silu(ctx, x);
}

// PyTorch RMSNorm with elementwise affine: y = x / sqrt(mean(x^2) + eps) * gamma
inline ggml_tensor *rms_norm(ggml_context *ctx, ggml_tensor *x,
                             ggml_tensor *gamma, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    if (gamma) x = ggml_mul(ctx, x, gamma);
    return x;
}

// AdaptiveLayerNorm (gpt_fast.model.AdaptiveLayerNorm). Given input (T, D) and
// embedding c (1, D):
//   w, b = project_layer(c).split(D, dim=-1)
//   return w * RMSNorm(input) + b           (gpt_fast uses RMSNorm internally)
//
// NOTE: although the class name is "AdaptiveLayerNorm", the underlying
// `self.norm` is a RMSNorm (constructed by Transformer.__init__ in
// gpt_fast/model.py), so the math is actually adaptive-RMSNorm.
inline ggml_tensor *adaptive_rms_norm(ggml_context *ctx, ggml_tensor *x_TD,
                                      ggml_tensor *c_1D,
                                      ggml_tensor *norm_gamma,
                                      ggml_tensor *proj_w, ggml_tensor *proj_b,
                                      int D, float eps) {
    // RMSNorm in float32 then cast back to x's dtype (matches PyTorch).
    ggml_tensor *y = rms_norm(ctx, x_TD, norm_gamma, eps);
    // project_layer(c): nn.Linear(D, 2D)
    ggml_tensor *wb = nn_linear(ctx, c_1D, proj_w, proj_b); // (1, 2D)
    // split [w | b] along ne[0]
    ggml_tensor *w = ggml_view_2d(ctx, wb, D, wb->ne[1], wb->nb[1], 0);
    ggml_tensor *b = ggml_view_2d(ctx, wb, D, wb->ne[1], wb->nb[1],
                                   (size_t)D * sizeof(float));
    // y broadcasts (D, T) * (D, 1) and adds (D, 1) — ggml broadcasts ne[1]=1.
    ggml_tensor *out = ggml_mul(ctx, y, w);
    out = ggml_add(ctx, out, b);
    return out;
}

// RoPE for gpt_fast: kernel applies cos/sin pair to last dim in (head_dim/2, 2)
// groups. ggml_rope_ext does the equivalent if we provide a position vector.
// gpt_fast uses standard interleaved-real-imag layout matching ggml's mode=0
// (head_dim/2 freqs, interleaved x[2i], x[2i+1] pairs).
inline ggml_tensor *apply_rope(ggml_context *ctx, ggml_tensor *qk_TBHD,
                               ggml_tensor *pos, int head_dim, float base) {
    return ggml_rope_ext(ctx, qk_TBHD, pos, /*freq_factors=*/nullptr,
                          /*n_dims=*/head_dim, /*mode=*/0,
                          /*n_ctx_orig=*/0,
                          /*freq_base=*/base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

// One TransformerBlock forward (gpt_fast.TransformerBlock).
// Inputs:
//   x_TD: (D, T) [T-fast]   - hidden state
//   c_1D: (D, 1)            - adaLN conditioning (t_emb broadcast)
//   skip_in_TD: optional skip-connection from earlier emitter layer
// Returns new x_TD.
struct LayerW {
    ggml_tensor *attn_norm_gamma, *attn_norm_pw, *attn_norm_pb;
    ggml_tensor *ffn_norm_gamma,  *ffn_norm_pw,  *ffn_norm_pb;
    ggml_tensor *wqkv, *wo;
    ggml_tensor *w1, *w2, *w3;
    ggml_tensor *skip_in_w, *skip_in_b; // optional
};

inline ggml_tensor *transformer_block(ggml_context *ctx,
                                      ggml_tensor *x_TD, ggml_tensor *c_1D,
                                      ggml_tensor *pos, ggml_tensor *mask_TT,
                                      const LayerW &lw,
                                      int D, int n_head, int head_dim,
                                      float rope_base, ggml_tensor *skip_in_TD,
                                      bool uvit_receive) {
    if (uvit_receive && skip_in_TD) {
        // skip_in_linear(cat([x, skip], dim=-1)): cat on ne[0] axis (D), then
        // Linear (2D → D).
        ggml_tensor *cat = ggml_concat(ctx, x_TD, skip_in_TD, /*dim=*/0);
        x_TD = nn_linear(ctx, cat, lw.skip_in_w, lw.skip_in_b);
    }
    // attention sub-block
    ggml_tensor *n1 = adaptive_rms_norm(ctx, x_TD, c_1D, lw.attn_norm_gamma,
                                         lw.attn_norm_pw, lw.attn_norm_pb,
                                         D, 1e-5f);
    // wqkv: (D → 3D)
    ggml_tensor *qkv = nn_linear(ctx, n1, lw.wqkv, nullptr);
    // qkv shape (3D, T, 1). Split along ne[0] into q, k, v.
    const int T = (int)x_TD->ne[1];
    ggml_tensor *q = ggml_view_3d(ctx, qkv, D, T, 1, qkv->nb[1], qkv->nb[2], 0);
    ggml_tensor *k = ggml_view_3d(ctx, qkv, D, T, 1, qkv->nb[1], qkv->nb[2],
                                   (size_t)D * sizeof(float));
    ggml_tensor *v = ggml_view_3d(ctx, qkv, D, T, 1, qkv->nb[1], qkv->nb[2],
                                   (size_t)(2 * D) * sizeof(float));
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);
    // Reshape to (head_dim, n_head, T, 1) then apply RoPE on ne[0].
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, T, 1);
    k = ggml_reshape_4d(ctx, k, head_dim, n_head, T, 1);
    v = ggml_reshape_4d(ctx, v, head_dim, n_head, T, 1);
    q = apply_rope(ctx, q, pos, head_dim, rope_base);
    k = apply_rope(ctx, k, pos, head_dim, rope_base);
    // Permute to (head_dim, T, n_head, 1) for ggml flash-attn-like layout —
    // ggml_mul_mat will compute Q @ K^T over ne[0] (head_dim).
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    // K^T @ Q via mul_mat: (head_dim, T, n_head) ⊗ (head_dim, T, n_head) along ne[0]
    ggml_tensor *kq = ggml_mul_mat(ctx, k, q);             // (T_k, T_q, n_head, 1)
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt((float)head_dim));
    if (mask_TT) {
        kq = ggml_add(ctx, kq, mask_TT);
    }
    kq = ggml_soft_max(ctx, kq);
    // Attn output: V^T @ softmax. Use ggml_mul_mat after transposing v on ne[0]/ne[1].
    ggml_tensor *v_t = ggml_cont(ctx, ggml_transpose(ctx, v)); // (T_v, head_dim, n_head)
    ggml_tensor *out_heads = ggml_mul_mat(ctx, v_t, kq);       // (head_dim, T_q, n_head)
    // Permute to (head_dim, n_head, T, 1) then reshape to (D, T).
    out_heads = ggml_cont(ctx, ggml_permute(ctx, out_heads, 0, 2, 1, 3));
    ggml_tensor *attn_out = ggml_reshape_3d(ctx, out_heads, D, T, 1);
    attn_out = nn_linear(ctx, attn_out, lw.wo, nullptr);

    ggml_tensor *h = ggml_add(ctx, x_TD, attn_out);

    // FFN sub-block (SwiGLU: w2(silu(w1(h)) * w3(h)))
    ggml_tensor *n2 = adaptive_rms_norm(ctx, h, c_1D, lw.ffn_norm_gamma,
                                         lw.ffn_norm_pw, lw.ffn_norm_pb,
                                         D, 1e-5f);
    ggml_tensor *gate = nn_linear(ctx, n2, lw.w1, nullptr);
    gate = silu(ctx, gate);
    ggml_tensor *up = nn_linear(ctx, n2, lw.w3, nullptr);
    ggml_tensor *gu = ggml_mul(ctx, gate, up);
    ggml_tensor *ff_out = nn_linear(ctx, gu, lw.w2, nullptr);
    h = ggml_add(ctx, h, ff_out);
    return h;
}

} // namespace

// Map helper: try to fetch from cfm map, prefixed.
static ggml_tensor *cfm_t(const S2MelWeights &w, const std::string &k) {
    auto it = w.cfm.find(k);
    return it == w.cfm.end() ? nullptr : it->second;
}

// Build LayerW from cfm map for layer index L.
static LayerW lw_for_layer(const S2MelWeights &w, int L) {
    LayerW lw{};
    auto p = std::string("estimator.transformer.layers.") + std::to_string(L) + ".";
    lw.attn_norm_gamma = cfm_t(w, p + "attention_norm.norm.weight");
    lw.attn_norm_pw    = cfm_t(w, p + "attention_norm.project_layer.weight");
    lw.attn_norm_pb    = cfm_t(w, p + "attention_norm.project_layer.bias");
    lw.ffn_norm_gamma  = cfm_t(w, p + "ffn_norm.norm.weight");
    lw.ffn_norm_pw     = cfm_t(w, p + "ffn_norm.project_layer.weight");
    lw.ffn_norm_pb     = cfm_t(w, p + "ffn_norm.project_layer.bias");
    lw.wqkv            = cfm_t(w, p + "attention.wqkv.weight");
    lw.wo              = cfm_t(w, p + "attention.wo.weight");
    lw.w1              = cfm_t(w, p + "feed_forward.w1.weight");
    lw.w2              = cfm_t(w, p + "feed_forward.w2.weight");
    lw.w3              = cfm_t(w, p + "feed_forward.w3.weight");
    lw.skip_in_w       = cfm_t(w, p + "skip_in_linear.weight");
    lw.skip_in_b       = cfm_t(w, p + "skip_in_linear.bias");
    return lw;
}

// Compute the timestep frequency embedding on the host (depends on `freqs`
// buffer which lives in cfm. The cfm `t_embedder.freqs` is a 1D buffer of
// length frequency_embedding_size/2 = 128).
//
// Output is f32 of length frequency_embedding_size = 256:
//   cat([cos(scale*t*freqs), sin(scale*t*freqs)], dim=-1)
static void timestep_freq_embed(const float *freqs, int half, float t,
                                 float scale, std::vector<float> &out) {
    out.assign((size_t)half * 2, 0.0f);
    for (int i = 0; i < half; ++i) {
        float a = scale * t * freqs[i];
        out[i] = std::cos(a);
        out[i + half] = std::sin(a);
    }
}

// One DiT estimator forward pass. Returns dphi (T, 80) C-fast in `out_dphi`.
//
// We allocate a fresh ggml_context + graph for each call. This is acceptable
// for n_steps=4 (8 graphs total with CFG); a fused-cache version is a future
// optimisation.
static bool dit_forward(ggml_backend_sched_t sched,
                         const S2MelWeights &w, const HParams &hp,
                         const DiTInputs &in,
                         std::vector<float> &out_dphi) {
    // Intermediate-dump gate for bisecting against PyTorch reference. When set
    // to a directory path, write each named gate as
    //   <dir>/dit_intern_<name>.f32 (raw float32, header-less)
    // PT reference dumper writes a matching dit_intern_<name>.npy.
    const char *dump_env = std::getenv("RS_INDEXTTS2_DIT_DUMP_DIR");
    std::string dump_dir = dump_env ? dump_env : "";
    auto dump_intern = [&](const std::string &name,
                            const float *data, size_t n) {
        if (dump_dir.empty()) return;
        std::string p = dump_dir + "/dit_intern_" + name + ".f32";
        FILE *f = std::fopen(p.c_str(), "wb");
        if (!f) return;
        std::fwrite(data, sizeof(float), n, f);
        std::fclose(f);
    };
    auto dump_tensor = [&](const std::string &name, ggml_tensor *t) {
        if (dump_dir.empty()) return;
        size_t n = ggml_nelements(t);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        dump_intern(name, buf.data(), n);
    };

    const int T   = in.T;
    const int D   = hp.s2mel_dit_hidden;     // 512
    const int H   = hp.s2mel_dit_heads;      // 8
    const int Hd  = D / H;                    // 64
    const int Mel = hp.s2mel_n_mels;          // 80
    const int Sty = hp.s2mel_style_dim;       // 192
    const int Cnd = hp.s2mel_lr_channels;     // 512  (cond_projection in)
    const int L   = hp.s2mel_dit_depth;       // 13
    const float ROPE_BASE = 10000.0f;
    const int FREQ_EMB    = 256;              // frequency_embedding_size
    const int half        = FREQ_EMB / 2;     // 128

    // Fetch input/output linear weights.
    ggml_tensor *cpw  = cfm_t(w, "estimator.cond_projection.weight");
    ggml_tensor *cpb  = cfm_t(w, "estimator.cond_projection.bias");
    ggml_tensor *xew  = cfm_t(w, "estimator.x_embedder.weight");
    ggml_tensor *xeb  = cfm_t(w, "estimator.x_embedder.bias");
    ggml_tensor *cmw  = cfm_t(w, "estimator.cond_x_merge_linear.weight");
    ggml_tensor *cmb  = cfm_t(w, "estimator.cond_x_merge_linear.bias");
    ggml_tensor *slw  = cfm_t(w, "estimator.skip_linear.weight");
    ggml_tensor *slb  = cfm_t(w, "estimator.skip_linear.bias");
    ggml_tensor *te0w = cfm_t(w, "estimator.t_embedder.mlp.0.weight");
    ggml_tensor *te0b = cfm_t(w, "estimator.t_embedder.mlp.0.bias");
    ggml_tensor *te2w = cfm_t(w, "estimator.t_embedder.mlp.2.weight");
    ggml_tensor *te2b = cfm_t(w, "estimator.t_embedder.mlp.2.bias");
    ggml_tensor *te0w2 = cfm_t(w, "estimator.t_embedder2.mlp.0.weight");
    ggml_tensor *te0b2 = cfm_t(w, "estimator.t_embedder2.mlp.0.bias");
    ggml_tensor *te2w2 = cfm_t(w, "estimator.t_embedder2.mlp.2.weight");
    ggml_tensor *te2b2 = cfm_t(w, "estimator.t_embedder2.mlp.2.bias");
    ggml_tensor *nw    = cfm_t(w, "estimator.transformer.norm.norm.weight");
    ggml_tensor *npw   = cfm_t(w, "estimator.transformer.norm.project_layer.weight");
    ggml_tensor *npb   = cfm_t(w, "estimator.transformer.norm.project_layer.bias");
    ggml_tensor *c1w   = cfm_t(w, "estimator.conv1.weight");
    ggml_tensor *c1b   = cfm_t(w, "estimator.conv1.bias");
    ggml_tensor *c2w   = cfm_t(w, "estimator.conv2.weight");
    ggml_tensor *c2b   = cfm_t(w, "estimator.conv2.bias");
    ggml_tensor *rpw   = cfm_t(w, "estimator.res_projection.weight");
    ggml_tensor *rpb   = cfm_t(w, "estimator.res_projection.bias");
    ggml_tensor *flnw  = cfm_t(w, "estimator.final_layer.linear.weight");
    ggml_tensor *flnb  = cfm_t(w, "estimator.final_layer.linear.bias");
    ggml_tensor *flmw  = cfm_t(w, "estimator.final_layer.adaLN_modulation.1.weight");
    ggml_tensor *flmb  = cfm_t(w, "estimator.final_layer.adaLN_modulation.1.bias");
    ggml_tensor *t_freqs_T = cfm_t(w, "estimator.t_embedder.freqs");
    ggml_tensor *t_freqs2_T = cfm_t(w, "estimator.t_embedder2.freqs");

    if (!cpw || !xew || !cmw || !slw || !te0w || !te2w ||
        !c1w || !c2w || !rpw || !flnw || !flmw || !nw ||
        !t_freqs_T || !t_freqs2_T) {
        RS_LOG_ERR("[dit] missing essential weights — DiT bindings incomplete");
        out_dphi.assign((size_t)T * Mel, 0.0f);
        return false;
    }

    // Read freqs to host so we can compute the timestep embedding on CPU.
    std::vector<float> freqs(half), freqs2(half);
    if ((int)ggml_nelements(t_freqs_T) != half ||
        (int)ggml_nelements(t_freqs2_T) != half) {
        RS_LOG_ERR("[dit] freqs len mismatch (%d/%d expected %d)",
                   (int)ggml_nelements(t_freqs_T),
                   (int)ggml_nelements(t_freqs2_T), half);
        return false;
    }
    ggml_backend_tensor_get(t_freqs_T, freqs.data(), 0, freqs.size() * sizeof(float));
    ggml_backend_tensor_get(t_freqs2_T, freqs2.data(), 0, freqs2.size() * sizeof(float));

    std::vector<float> tfe(FREQ_EMB), tfe2(FREQ_EMB);
    timestep_freq_embed(freqs.data(),  half, in.t, 1000.0f, tfe);
    timestep_freq_embed(freqs2.data(), half, in.t, 1000.0f, tfe2);

    // Build host-side composite input buffer for cond_x_merge_linear:
    //   cat([x(80), prompt_x(80), cond_projection(mu)(512), style.repeat(T)(192)])
    //   = (T, 864) C-fast
    // We do cond_projection on the GPU in a small mini-graph to avoid a host
    // mat-mul. Same for x_embedder (which is just a Linear too). So mini-graph
    // 1 computes:
    //   cond_proj_out (T, 512)
    //   x_embedder is applied later inside the main graph (it's optional —
    //   currently NOT used by upstream since x is concatenated raw, but the
    //   weights exist for ablation). We follow upstream verbatim — DiT.forward
    //   passes the raw `x` and `prompt_x` into the cat, not x_embedder. So we
    //   skip x_embedder here too. (Names retained above for completeness.)
    (void)xew; (void)xeb;

    // ---- Mini-graph 1: cond_projection (Linear 512 → 512) ----
    std::vector<float> cond_proj((size_t)T * D, 0.0f);
    {
        std::vector<uint8_t> meta(ggml_tensor_overhead() * 32 +
                                   ggml_graph_overhead_custom(32, false));
        ggml_init_params ip = { meta.size(), meta.data(), true };
        ggml_context *ctx = ggml_init(ip);
        ggml_cgraph *gf = ggml_new_graph_custom(ctx, 32, false);
        ggml_tensor *mu_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cnd, T);
        ggml_set_input(mu_in);
        ggml_tensor *y = nn_linear(ctx, mu_in, cpw, cpb);
        ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            RS_LOG_ERR("[dit] cond_projection alloc failed");
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(mu_in, in.mu, 0, (size_t)T * Cnd * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[dit] cond_projection compute failed");
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_get(y, cond_proj.data(), 0, cond_proj.size() * sizeof(float));
        ggml_free(ctx);
    }
    dump_intern("cond_proj", cond_proj.data(), cond_proj.size());

    // ---- Host-side cat: (T, 80+80+512+192=864) C-fast ----
    const int merge_in = Mel + Mel + D + Sty;  // 864
    std::vector<float> x_in_merge((size_t)T * merge_in);
    for (int t = 0; t < T; ++t) {
        float *dst = &x_in_merge[(size_t)t * merge_in];
        // x (80)
        std::memcpy(dst, in.x + (size_t)t * Mel, Mel * sizeof(float));
        // prompt_x (80)
        std::memcpy(dst + Mel, in.prompt_x + (size_t)t * Mel, Mel * sizeof(float));
        // cond (512)
        std::memcpy(dst + Mel + Mel, &cond_proj[(size_t)t * D], D * sizeof(float));
        // style.repeat(T): copy `style` (192,) into each row.
        std::memcpy(dst + Mel + Mel + D, in.style, Sty * sizeof(float));
    }
    dump_intern("x_in_merge", x_in_merge.data(), x_in_merge.size());

    // ---- Main graph: cond_x_merge_linear → transformer (13L) → wavenet head ----
    const int n_nodes = 16384;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * n_nodes +
                               ggml_graph_overhead_custom(n_nodes, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // x_in: (merge_in=864, T, 1)
    ggml_tensor *x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, merge_in, T, 1);
    ggml_set_name(x_in, "x_in_merge");
    ggml_set_input(x_in);
    // t_freq_emb: (256,)
    ggml_tensor *tfe_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FREQ_EMB);
    ggml_set_name(tfe_in, "tfe");
    ggml_set_input(tfe_in);
    ggml_tensor *tfe2_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, FREQ_EMB);
    ggml_set_name(tfe2_in, "tfe2");
    ggml_set_input(tfe2_in);
    // pos: (T,) i32
    ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    // x_raw: (80, T, 1) for long_skip_connection
    ggml_tensor *x_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Mel, T, 1);
    ggml_set_name(x_raw, "x_raw");
    ggml_set_input(x_raw);

    // t_emb path: tfe (1, 256) → mlp.0 → SiLU → mlp.2 → (1, D)
    auto t_mlp = [&](ggml_tensor *e, ggml_tensor *m0w, ggml_tensor *m0b,
                     ggml_tensor *m2w, ggml_tensor *m2b) {
        ggml_tensor *e2 = ggml_reshape_2d(ctx, e, FREQ_EMB, 1); // (256, 1)
        ggml_tensor *y  = nn_linear(ctx, e2, m0w, m0b);
        y = silu(ctx, y);
        y = nn_linear(ctx, y, m2w, m2b);                        // (D, 1)
        return y;
    };
    ggml_tensor *t_emb  = t_mlp(tfe_in,  te0w,  te0b,  te2w,  te2b);
    ggml_tensor *t_emb2 = t_mlp(tfe2_in, te0w2, te0b2, te2w2, te2b2);

    // cond_x_merge_linear: (864, T, 1) → (D, T, 1)
    ggml_tensor *h = nn_linear(ctx, x_in, cmw, cmb);
    ggml_tensor *h_post_merge_t = h;
    ggml_set_name(h_post_merge_t, "h_post_merge");
    ggml_set_output(h_post_merge_t);
    ggml_build_forward_expand(gf, h_post_merge_t);

    // 13-layer transformer with uvit_skip_connection.
    // emit layers: 0..(L/2-1) push their output (after the layer) to a stack.
    // recv layers: those with i > L/2 pop from the stack and feed via
    // skip_in_linear at the entry of the layer.
    std::vector<ggml_tensor *> skip_stack;
    ggml_tensor *layer0_t = nullptr;
    ggml_tensor *layer6_t = nullptr;
    ggml_tensor *layer12_t = nullptr;
    for (int li = 0; li < L; ++li) {
        bool recv = (li > L / 2);
        ggml_tensor *skip = nullptr;
        if (recv && !skip_stack.empty()) {
            skip = skip_stack.back();
            skip_stack.pop_back();
        }
        LayerW lw = lw_for_layer(w, li);
        h = transformer_block(ctx, h, t_emb, pos, /*mask=*/nullptr,
                               lw, D, H, Hd, ROPE_BASE, skip, recv);
        if (li == 0)  { layer0_t  = h; ggml_set_name(h, "layer0_out");  ggml_set_output(h); ggml_build_forward_expand(gf, h); }
        if (li == 6)  { layer6_t  = h; ggml_set_name(h, "layer6_out");  ggml_set_output(h); ggml_build_forward_expand(gf, h); }
        if (li == 12) { layer12_t = h; ggml_set_name(h, "layer12_out"); ggml_set_output(h); ggml_build_forward_expand(gf, h); }
        if (li < L / 2) skip_stack.push_back(h);
    }
    // Final adaLN over (D, T, 1) using transformer.norm.{norm,project_layer}
    h = adaptive_rms_norm(ctx, h, t_emb, nw, npw, npb, D, 1e-5f);
    ggml_tensor *xres_post_xfmr_t = h;
    ggml_set_name(xres_post_xfmr_t, "xres_post_xfmr");
    ggml_set_output(xres_post_xfmr_t);
    ggml_build_forward_expand(gf, xres_post_xfmr_t);

    // long_skip_connection: cat([h, x_raw], dim=-1) along ne[0] (D + Mel),
    // then skip_linear (D+Mel → D).
    ggml_tensor *h_skip = ggml_concat(ctx, h, x_raw, /*dim=*/0); // (D+Mel, T, 1)
    h = nn_linear(ctx, h_skip, slw, slb);
    ggml_tensor *xres_post_skip_t = h;
    ggml_set_name(xres_post_skip_t, "xres_post_skip");
    ggml_set_output(xres_post_skip_t);
    ggml_build_forward_expand(gf, xres_post_skip_t);

    // Save x_res for res_projection.
    ggml_tensor *x_res = h;

    // conv1: Linear(D → D)  (wavenet hidden_dim is also D in this config)
    ggml_tensor *wnh = nn_linear(ctx, x_res, c1w, c1b);          // (D, T, 1)
    ggml_set_name(wnh, "wnh_post_conv1");
    ggml_set_output(wnh);
    ggml_build_forward_expand(gf, wnh);

    // ---- WaveNet (8 layers) ----
    // g = cond_layer(t_emb2.unsqueeze(2))    # (2D*L, 1, 1) → broadcast over T
    //   cond_layer: Conv1d(D, 2D*L, k=1)
    // For ggml, cond_layer.weight has shape [K=1, Cin=D, Cout=2D*L] f16. We
    // apply it as a Linear since k=1: nn_linear(t_emb2_3d, weight_2d).
    ggml_tensor *cond_w = cfm_t(w, "estimator.wavenet.cond_layer.conv.conv.weight");
    ggml_tensor *cond_b = cfm_t(w, "estimator.wavenet.cond_layer.conv.conv.bias");
    if (!cond_w) {
        RS_LOG_ERR("[dit] wavenet.cond_layer.weight missing");
        ggml_free(ctx);
        return false;
    }
    // cond_w original PyTorch (Cout=2*D*N_layers, Cin=D, K=1) → ggml stores it
    // as (K=1, Cin=D, Cout=2*D*N_layers). Reshape to (D, 2*D*N_layers) for
    // mul_mat (drop the K=1 axis).
    const int N_wn = hp.s2mel_wavenet_layers; // 8
    ggml_tensor *cond_w_2d = ggml_reshape_2d(ctx, cond_w, D, 2 * D * N_wn);
    ggml_tensor *t_emb2_d  = t_emb2; // (D, 1)
    ggml_tensor *g_full    = nn_linear(ctx, t_emb2_d, cond_w_2d, cond_b);
    // g_full: (2*D*N_wn, 1, 1). We'll slice per-layer below as
    //   g_l = g_full[i*2*D:(i+1)*2*D]  (2*D,)
    // and add elementwise to wnh's 2*D channels per layer (broadcast over T).

    {
        // Fast path: keep the WaveNet head in the ggml graph so Metal/CUDA can
        // execute the heavy Conv1d layers. The old host fallback below assumed
        // T<60, but reference prompts can push T well past 1000.
        ggml_tensor *x_wn = ggml_cont(ctx, ggml_transpose(ctx, wnh)); // (T, D, 1)
        ggml_tensor *wn_acc = ggml_scale(ctx, x_wn, 0.0f);            // (T, D, 1)

        auto add_conv_bias = [&](ggml_tensor *y, ggml_tensor *b, int C) {
            if (!b) return y;
            ggml_tensor *br = ggml_reshape_3d(ctx, b, 1, C, 1);
            return ggml_add(ctx, y, br);
        };
        auto view_channels = [&](ggml_tensor *x, int c0, int C) {
            return ggml_view_3d(ctx, x, T, C, 1, x->nb[1], x->nb[2],
                                (size_t)c0 * x->nb[1]);
        };

        for (int li = 0; li < N_wn; ++li) {
            const std::string in_pfx =
                std::string("estimator.wavenet.in_layers.") +
                std::to_string(li) + ".conv.conv.";
            ggml_tensor *inw = cfm_t(w, in_pfx + "weight");
            ggml_tensor *inb = cfm_t(w, in_pfx + "bias");
            if (!inw) {
                RS_LOG_ERR("[dit] wavenet in_layers.%d.weight missing", li);
                ggml_free(ctx);
                return false;
            }

            ggml_tensor *x_pad = ggml_pad_reflect_1d(ctx, x_wn, 2, 2);
            x_pad = ggml_cont(ctx, x_pad);
            ggml_tensor *x_in_w = ggml_conv_1d(ctx, w_f16(ctx, inw), x_pad,
                                                /*s=*/1, /*p=*/0, /*d=*/1);
            x_in_w = add_conv_bias(x_in_w, inb, 2 * D);

            ggml_tensor *g_l = ggml_view_2d(ctx, g_full, 2 * D, 1,
                                            g_full->nb[1],
                                            (size_t)li * 2 * D * sizeof(float));
            g_l = ggml_reshape_3d(ctx, g_l, 1, 2 * D, 1);
            x_in_w = ggml_add(ctx, x_in_w, g_l);

            ggml_tensor *gate_t = view_channels(x_in_w, 0, D);
            ggml_tensor *gate_s = view_channels(x_in_w, D, D);
            ggml_tensor *acts = ggml_mul(ctx, ggml_tanh(ctx, gate_t),
                                         ggml_sigmoid(ctx, gate_s)); // (T, D, 1)

            const int res_skip_C = (li < N_wn - 1) ? 2 * D : D;
            const std::string rs_pfx =
                std::string("estimator.wavenet.res_skip_layers.") +
                std::to_string(li) + ".conv.conv.";
            ggml_tensor *rsw = cfm_t(w, rs_pfx + "weight");
            ggml_tensor *rsb = cfm_t(w, rs_pfx + "bias");
            if (!rsw) {
                RS_LOG_ERR("[dit] wavenet res_skip_layers.%d.weight missing", li);
                ggml_free(ctx);
                return false;
            }
            ggml_tensor *rs = ggml_conv_1d(ctx, w_f16(ctx, rsw), acts,
                                           /*s=*/1, /*p=*/0, /*d=*/1);
            rs = add_conv_bias(rs, rsb, res_skip_C);

            if (li < N_wn - 1) {
                ggml_tensor *r_res  = view_channels(rs, 0, D);
                ggml_tensor *r_skip = view_channels(rs, D, D);
                x_wn   = ggml_cont(ctx, ggml_add(ctx, x_wn, r_res));
                wn_acc = ggml_add(ctx, wn_acc, r_skip);
            } else {
                wn_acc = ggml_add(ctx, wn_acc, rs);
            }
        }

        ggml_tensor *wn_out_DT = ggml_cont(ctx, ggml_transpose(ctx, wn_acc));
        ggml_set_name(wn_out_DT, "wnh_post_wavenet");
        ggml_set_output(wn_out_DT);

        ggml_tensor *rp_out = nn_linear(ctx, x_res, rpw, rpb);
        ggml_tensor *y = ggml_add(ctx, wn_out_DT, rp_out);

        ggml_tensor *cw = ggml_mul_mat(ctx, flmw, ggml_silu(ctx, t_emb));
        cw = ggml_add(ctx, cw, flmb); // (2D, 1)
        ggml_tensor *shift = ggml_view_2d(ctx, cw, D, 1, cw->nb[1], 0);
        ggml_tensor *scale = ggml_view_2d(ctx, cw, D, 1, cw->nb[1],
                                          (size_t)D * sizeof(float));
        y = ggml_norm(ctx, y, 1e-6f);
        ggml_tensor *one_plus = ggml_scale_bias(ctx, scale, 1.0f, 1.0f);
        y = ggml_mul(ctx, y, one_plus);
        y = ggml_add(ctx, y, shift);
        y = nn_linear(ctx, y, flnw, flnb);
        ggml_set_name(y, "final_layer");
        ggml_set_output(y);

        ggml_tensor *c2w_2d = ggml_reshape_2d(ctx, c2w, D, Mel);
        ggml_tensor *dphi_t = nn_linear(ctx, y, c2w_2d, c2b);
        ggml_set_name(dphi_t, "dphi");
        ggml_set_output(dphi_t);
        ggml_build_forward_expand(gf, dphi_t);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            RS_LOG_ERR("[dit] graph alloc failed");
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(x_in, x_in_merge.data(), 0,
                                x_in_merge.size() * sizeof(float));
        ggml_backend_tensor_set(tfe_in, tfe.data(), 0, tfe.size() * sizeof(float));
        ggml_backend_tensor_set(tfe2_in, tfe2.data(), 0, tfe2.size() * sizeof(float));
        std::vector<int32_t> pos_h(T);
        for (int i = 0; i < T; ++i) pos_h[i] = i;
        ggml_backend_tensor_set(pos, pos_h.data(), 0,
                                pos_h.size() * sizeof(int32_t));
        ggml_backend_tensor_set(x_raw, in.x, 0,
                                (size_t)T * Mel * sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[dit] graph compute failed");
            ggml_free(ctx);
            return false;
        }

        out_dphi.assign((size_t)T * Mel, 0.0f);
        ggml_backend_tensor_get(dphi_t, out_dphi.data(), 0,
                                out_dphi.size() * sizeof(float));
        if (!dump_dir.empty()) {
            dump_tensor("h_post_merge",     h_post_merge_t);
            dump_tensor("layer0_out",       layer0_t);
            dump_tensor("layer6_out",       layer6_t);
            dump_tensor("layer12_out",      layer12_t);
            dump_tensor("xres_post_xfmr",   xres_post_xfmr_t);
            dump_tensor("xres_post_skip",   xres_post_skip_t);
            dump_tensor("wnh_post_conv1",   wnh);
            dump_tensor("wnh_post_wavenet", wn_out_DT);
            dump_tensor("res_proj",         rp_out);
            dump_tensor("final_layer",      y);
        }
        ggml_free(ctx);
        return true;
    }

    // WaveNet operates with T-fast (C, T) layout. Our wnh is already (D, T)
    // T-fast (ne[0]=D, ne[1]=T). For ggml_conv_1d the data layout is also
    // (T, Cin, B=1) with T-fast — same memory. So we keep it as-is.
    // But ggml_conv_1d treats ne[0] as T. We currently have ne[0]=D. We need to
    // transpose to (T, D, 1) for conv. SConv1d uses reflect padding which ggml
    // doesn't have natively, but for kernel=5 dilation=1 padding=2 we'll do
    // manual reflect pad on host before feed-in? No — the kernel=5, dilation=1
    // case: padding_total=4, left=2, right=2. We'd need reflect.
    //
    // For now, fall back to a host-side wavenet loop: read wnh from the graph
    // boundary, run wavenet on host using contiguous Conv1d with reflect pad
    // (T=51 makes the reflect pad easy), then return to graph. But this means
    // we have to materialize wnh, do the wavenet on CPU, materialize the
    // result, then continue with the FinalLayer + conv2.
    //
    // To keep the graph cohesive, instead I implement wavenet entirely on host
    // (8 small layers, T<60 typically). Implementing reflect padding +
    // conv1d via host loops is straightforward, and the perf cost is minor
    // compared to the transformer.
    //
    // We compute wavenet on host AFTER materializing wnh; then re-enter graph
    // for FinalLayer + conv2. Implementation continues outside the graph.

    ggml_set_name(x_res, "x_res");
    ggml_set_output(x_res);

    // We also need t_emb materialized for the final layer's adaLN.
    ggml_set_name(t_emb, "t_emb");
    ggml_set_output(t_emb);
    // And g_full for the wavenet on host.
    ggml_set_name(g_full, "g_full");
    ggml_set_output(g_full);

    ggml_build_forward_expand(gf, wnh);
    ggml_build_forward_expand(gf, x_res);
    ggml_build_forward_expand(gf, t_emb);
    ggml_build_forward_expand(gf, g_full);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        RS_LOG_ERR("[dit] main graph alloc failed");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, x_in_merge.data(), 0,
                            x_in_merge.size() * sizeof(float));
    ggml_backend_tensor_set(tfe_in, tfe.data(), 0, tfe.size() * sizeof(float));
    ggml_backend_tensor_set(tfe2_in, tfe2.data(), 0, tfe2.size() * sizeof(float));
    std::vector<int32_t> pos_h(T);
    for (int i = 0; i < T; ++i) pos_h[i] = i;
    ggml_backend_tensor_set(pos, pos_h.data(), 0, pos_h.size() * sizeof(int32_t));
    // x_raw is ggml (Mel, T, 1) with ne[0]=Mel-fast. in.x is host (T, Mel) C-fast,
    // i.e. memory[t*Mel + m] = val(t, m). ggml expects memory[m + t*Mel] = val(m, t)
    // — same indexing. Pass in.x directly without transposing.
    ggml_backend_tensor_set(x_raw, in.x, 0, (size_t)T * Mel * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("[dit] main graph compute failed");
        ggml_free(ctx);
        return false;
    }

    // Read out wnh, x_res, t_emb, g_full.
    std::vector<float> wnh_h((size_t)D * T);
    std::vector<float> xres_h((size_t)D * T);
    std::vector<float> temb_h(D);
    std::vector<float> g_full_h((size_t)2 * D * N_wn);
    ggml_backend_tensor_get(wnh,   wnh_h.data(),   0, wnh_h.size() * sizeof(float));
    ggml_backend_tensor_get(x_res, xres_h.data(),  0, xres_h.size() * sizeof(float));
    ggml_backend_tensor_get(t_emb, temb_h.data(),  0, temb_h.size() * sizeof(float));
    ggml_backend_tensor_get(g_full, g_full_h.data(), 0, g_full_h.size() * sizeof(float));
    // Intermediate dumps for bisection. Read into temp buffers then dump.
    if (!dump_dir.empty()) {
        dump_tensor("h_post_merge",     h_post_merge_t);
        dump_tensor("layer0_out",       layer0_t);
        dump_tensor("layer6_out",       layer6_t);
        dump_tensor("layer12_out",      layer12_t);
        dump_tensor("xres_post_xfmr",   xres_post_xfmr_t);
        dump_tensor("xres_post_skip",   xres_post_skip_t);
        // wnh and x_res are already retrieved above.
        dump_intern("wnh_post_conv1", wnh_h.data(),  wnh_h.size());
    }
    ggml_free(ctx);

    // ---- Host-side WaveNet ----
    // The ggml-side wnh tensor has ne[0]=D, ne[1]=T, i.e. D-fast / T-slow in
    // memory (memory[t*D + d] = val(d, t)). That is the same byte layout as
    // host (T, D) C-fast. The host WaveNet below operates in (D-slow, T-fast)
    // layout — xin[c*T + t] = val(c, t) — to match PyTorch (B, C, T). We
    // transpose at the boundary so the inner loop matches PT semantics.
    std::vector<float> wnh_DTfast;
    transpose_2d(wnh_h, T, D, wnh_DTfast); // (T, D) C-fast → (D, T) C-fast
    //
    // We iterate as in_layer[i] → tanh*sigmoid → res_skip_layer[i].
    // We accumulate output := output + res_skip_acts (with the last layer
    // contributing the entire skip channel). For non-last layers, the first
    // half of res_skip_acts is added to x (residual), and the second half is
    // added to output.
    //
    // Both layers are Conv1d(K=5, dilation=1, padding=2 reflect) for in_layer
    // and Conv1d(K=1) for res_skip_layer.
    //
    // wnh_h layout: row r of length T is channel r. wnh_h[r*T + t].
    auto conv1d_reflect_k5_d1 = [&](const float *xin, int Cin, int Cout,
                                     int Tlen, const float *kernel, // (Cout, Cin, K=5)
                                     const float *bias,             // (Cout,) or null
                                     std::vector<float> &out) {
        out.assign((size_t)Cout * Tlen, 0.0f);
        const int K = 5;
        const int pad = 2;
        // Each output sample at t = sum over k=0..4 of input at (t + k - pad)
        // with reflect padding: index in [0..Tlen-1].
        // PyTorch reflect padding mirrors: e.g. -1 → 1, -2 → 2; Tlen → Tlen-2,
        // Tlen+1 → Tlen-3. Implement via lambda.
        auto reflect = [&](int idx) {
            if (Tlen == 1) return 0;
            int period = 2 * (Tlen - 1);
            int m = ((idx % period) + period) % period;
            if (m >= Tlen) m = period - m;
            return m;
        };
        for (int co = 0; co < Cout; ++co) {
            const float bv = bias ? bias[co] : 0.0f;
            float *out_row = &out[(size_t)co * Tlen];
            for (int t = 0; t < Tlen; ++t) {
                float acc = bv;
                for (int ci = 0; ci < Cin; ++ci) {
                    const float *xin_row = xin + (size_t)ci * Tlen;
                    const float *krow    = kernel + ((size_t)co * Cin + ci) * K;
                    for (int k = 0; k < K; ++k) {
                        int idx = reflect(t + k - pad);
                        acc += xin_row[idx] * krow[k];
                    }
                }
                out_row[t] = acc;
            }
        }
    };
    auto conv1d_k1 = [&](const float *xin, int Cin, int Cout, int Tlen,
                          const float *kernel, // (Cout, Cin, 1)
                          const float *bias,
                          std::vector<float> &out) {
        out.assign((size_t)Cout * Tlen, 0.0f);
        for (int co = 0; co < Cout; ++co) {
            const float bv = bias ? bias[co] : 0.0f;
            float *out_row = &out[(size_t)co * Tlen];
            for (int t = 0; t < Tlen; ++t) {
                float acc = bv;
                for (int ci = 0; ci < Cin; ++ci) {
                    acc += xin[(size_t)ci * Tlen + t]
                            * kernel[(size_t)co * Cin + ci];
                }
                out_row[t] = acc;
            }
        }
    };

    // Pull all wavenet weights to host as f32 (f16 in GGUF → cast on the fly).
    auto pull_w = [&](const std::string &k, std::vector<float> &dst) -> bool {
        ggml_tensor *t = cfm_t(w, k);
        if (!t) { dst.clear(); return false; }
        size_t n = ggml_nelements(t);
        dst.assign(n, 0.0f);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) dst[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            return false;
        }
        return true;
    };

    std::vector<float> x_wn = wnh_DTfast;  // (D, T) C-fast (D-slow, T-fast)
    std::vector<float> output((size_t)D * T, 0.0f);
    for (int li = 0; li < N_wn; ++li) {
        auto pfx = std::string("estimator.wavenet.in_layers.") + std::to_string(li) + ".conv.conv.";
        std::vector<float> inw, inb, rsw, rsb;
        if (!pull_w(pfx + "weight", inw)) {
            RS_LOG_ERR("[dit] wavenet in_layers.%d.weight missing", li);
            return false;
        }
        pull_w(pfx + "bias", inb);
        // in_layer Conv1d(D, 2D, K=5)
        std::vector<float> x_in_w;
        conv1d_reflect_k5_d1(x_wn.data(), D, 2 * D, T, inw.data(),
                              inb.empty() ? nullptr : inb.data(), x_in_w);
        // Add g_l (broadcast over T)
        const float *g_l = &g_full_h[(size_t)li * 2 * D];
        // x_in_w layout (2D, T). Add g_l[c] elementwise over T.
        for (int c = 0; c < 2 * D; ++c) {
            float gv = g_l[c];
            float *row = &x_in_w[(size_t)c * T];
            for (int t = 0; t < T; ++t) row[t] += gv;
        }
        // tanh(first half) * sigmoid(second half)
        std::vector<float> acts((size_t)D * T);
        for (int c = 0; c < D; ++c) {
            const float *t_in = &x_in_w[(size_t)c * T];
            const float *s_in = &x_in_w[(size_t)(c + D) * T];
            float *out_row = &acts[(size_t)c * T];
            for (int t = 0; t < T; ++t) {
                float th = std::tanh(t_in[t]);
                float sg = 1.0f / (1.0f + std::exp(-s_in[t]));
                out_row[t] = th * sg;
            }
        }
        // res_skip Conv1d(D, res_skip_channels, K=1)
        const int res_skip_C = (li < N_wn - 1) ? 2 * D : D;
        auto pfx2 = std::string("estimator.wavenet.res_skip_layers.") + std::to_string(li) + ".conv.conv.";
        if (!pull_w(pfx2 + "weight", rsw)) {
            RS_LOG_ERR("[dit] wavenet res_skip_layers.%d.weight missing", li);
            return false;
        }
        pull_w(pfx2 + "bias", rsb);
        std::vector<float> rs;
        conv1d_k1(acts.data(), D, res_skip_C, T, rsw.data(),
                   rsb.empty() ? nullptr : rsb.data(), rs);
        if (li < N_wn - 1) {
            // x_wn += rs[:D]; output += rs[D:]
            for (int c = 0; c < D; ++c) {
                float *xw = &x_wn[(size_t)c * T];
                const float *rA = &rs[(size_t)c * T];
                const float *rB = &rs[(size_t)(c + D) * T];
                float *o = &output[(size_t)c * T];
                for (int t = 0; t < T; ++t) {
                    xw[t] += rA[t];
                    o[t]  += rB[t];
                }
            }
        } else {
            for (int c = 0; c < D; ++c) {
                const float *rA = &rs[(size_t)c * T];
                float *o = &output[(size_t)c * T];
                for (int t = 0; t < T; ++t) o[t] += rA[t];
            }
        }
    }
    // output: (D, T) T-fast — this is the wavenet output. Transpose to
    // (T, D) C-fast for the rest of the calculation (matches x_res layout
    // produced by the transformer, which is (D, T) T-fast).
    //
    // Upstream then does .transpose(1,2) on wavenet output to make it (B, T, D)
    // before adding res_projection(x_res). So in "C-fast" terms output_TC =
    // transpose_2d(output_DT, D, T).
    std::vector<float> wn_out_TC;
    transpose_2d(output, D, T, wn_out_TC);
    // x_res's ggml buffer (ne0=D, ne1=T) is already in the (T, D) C-fast byte
    // layout that the res_projection mini-graph's xi expects — no transpose
    // needed. (We previously transposed it here, which scrambled the tensor.)
    // PT saves wavenet output as (D, T) directly (no transpose). Match that.
    dump_intern("wnh_post_wavenet", output.data(), output.size());

    // ---- res_projection: Linear(D, D), applied row-wise on x_res ----
    // Use a small mini-graph to leverage the backend.
    std::vector<float> rp_out_TC((size_t)T * D);
    {
        std::vector<uint8_t> meta2(ggml_tensor_overhead() * 32 +
                                    ggml_graph_overhead_custom(32, false));
        ggml_init_params ip2 = { meta2.size(), meta2.data(), true };
        ggml_context *ctx2 = ggml_init(ip2);
        ggml_cgraph *gf2 = ggml_new_graph_custom(ctx2, 32, false);
        ggml_tensor *xi = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, D, T);
        ggml_set_input(xi);
        ggml_tensor *y = nn_linear(ctx2, xi, rpw, rpb);
        ggml_set_output(y);
        ggml_build_forward_expand(gf2, y);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf2);
        ggml_backend_tensor_set(xi, xres_h.data(), 0, xres_h.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf2) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[dit] res_projection compute failed");
            ggml_free(ctx2);
            return false;
        }
        ggml_backend_tensor_get(y, rp_out_TC.data(), 0, rp_out_TC.size() * sizeof(float));
        ggml_free(ctx2);
    }
    dump_intern("res_proj", rp_out_TC.data(), rp_out_TC.size());
    // wn_out_TC + rp_out_TC
    for (size_t i = 0; i < wn_out_TC.size(); ++i) wn_out_TC[i] += rp_out_TC[i];

    // ---- FinalLayer: AdaLN (no affine) + Linear ----
    // adaLN_modulation: silu(c) → Linear(D, 2D) → (shift, scale)
    // modulate: y = norm_no_affine(x) * (1 + scale) + shift
    // Then linear: (D → D).
    // Then conv2: Conv1d(D, 80, k=1) on (B, D, T) — applied per-time-step as a
    // Linear (D → 80).
    std::vector<float> final_out_TC((size_t)T * D);
    {
        std::vector<uint8_t> meta3(ggml_tensor_overhead() * 128 +
                                    ggml_graph_overhead_custom(128, false));
        ggml_init_params ip3 = { meta3.size(), meta3.data(), true };
        ggml_context *ctx3 = ggml_init(ip3);
        ggml_cgraph *gf3 = ggml_new_graph_custom(ctx3, 128, false);
        ggml_tensor *xi = ggml_new_tensor_2d(ctx3, GGML_TYPE_F32, D, T);
        ggml_set_input(xi);
        ggml_tensor *temb_in = ggml_new_tensor_2d(ctx3, GGML_TYPE_F32, D, 1);
        ggml_set_input(temb_in);
        // silu(temb) → Linear(D, 2D)
        ggml_tensor *cw = ggml_mul_mat(ctx3, flmw, ggml_silu(ctx3, temb_in));
        cw = ggml_add(ctx3, cw, flmb); // (2D, 1)
        // split into shift, scale (each (D, 1))
        ggml_tensor *shift = ggml_view_2d(ctx3, cw, D, 1, cw->nb[1], 0);
        ggml_tensor *scale = ggml_view_2d(ctx3, cw, D, 1, cw->nb[1],
                                            (size_t)D * sizeof(float));
        // LayerNorm without affine: mean/var across last dim, eps=1e-6
        ggml_tensor *y = ggml_norm(ctx3, xi, 1e-6f);
        // modulate: y * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        // scale/shift are (D, 1); y is (D, T). Broadcast over T.
        ggml_tensor *one_plus = ggml_scale_bias(ctx3, scale, 1.0f, 1.0f);
        y = ggml_mul(ctx3, y, one_plus);
        y = ggml_add(ctx3, y, shift);
        // linear: (D → D)
        y = nn_linear(ctx3, y, flnw, flnb);
        ggml_set_output(y);
        ggml_build_forward_expand(gf3, y);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf3);
        ggml_backend_tensor_set(xi, wn_out_TC.data(), 0, wn_out_TC.size() * sizeof(float));
        ggml_backend_tensor_set(temb_in, temb_h.data(), 0, temb_h.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf3) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[dit] final_layer compute failed");
            ggml_free(ctx3);
            return false;
        }
        ggml_backend_tensor_get(y, final_out_TC.data(), 0, final_out_TC.size() * sizeof(float));
        ggml_free(ctx3);
    }
    dump_intern("final_layer", final_out_TC.data(), final_out_TC.size());

    // ---- conv2: Conv1d(D, Mel=80, k=1) ----
    // Applied on (B, D, T). Same as Linear(D → Mel) row-wise. Output shape
    // (B, Mel, T) = (Mel, T) T-fast. We want (T, Mel) C-fast for matching the
    // upstream numpy save.
    out_dphi.assign((size_t)T * Mel, 0.0f);
    {
        std::vector<uint8_t> meta4(ggml_tensor_overhead() * 32 +
                                    ggml_graph_overhead_custom(32, false));
        ggml_init_params ip4 = { meta4.size(), meta4.data(), true };
        ggml_context *ctx4 = ggml_init(ip4);
        ggml_cgraph *gf4 = ggml_new_graph_custom(ctx4, 32, false);
        ggml_tensor *xi = ggml_new_tensor_2d(ctx4, GGML_TYPE_F32, D, T);
        ggml_set_input(xi);
        // conv2.weight in GGUF: PyTorch Conv1d (Mel, D, K=1) → ggml (1, D, Mel)
        // = reshape to (D, Mel) for nn_linear (in=D, out=Mel).
        ggml_tensor *c2w_2d = ggml_reshape_2d(ctx4, c2w, D, Mel);
        ggml_tensor *y = nn_linear(ctx4, xi, c2w_2d, c2b);
        ggml_set_output(y);
        ggml_build_forward_expand(gf4, y);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf4);
        ggml_backend_tensor_set(xi, final_out_TC.data(), 0, final_out_TC.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf4) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[dit] conv2 compute failed");
            ggml_free(ctx4);
            return false;
        }
        ggml_backend_tensor_get(y, out_dphi.data(), 0, out_dphi.size() * sizeof(float));
        ggml_free(ctx4);
    }
    return true;
}

// -----------------------------------------------------------------------------
// CFM Euler solver (matches BASECFM.solve_euler in upstream).
//
// Output: mel [T_out, n_mels] in time-major [T, C] layout, with the prompt
// region cropped off exactly like infer_v2.py.
// -----------------------------------------------------------------------------
static bool cfm_euler(ggml_backend_sched_t sched, const S2MelWeights &w,
                      const HParams &hp, std::mt19937 &rng,
                      const std::vector<float> &prompt_condition,
                      const std::vector<float> &cond_512,
                      const std::vector<float> &prompt_mel,
                      const std::vector<float> &style,
                      int T_prompt, int T_out, int n_steps,
                      float cfg_rate, float temperature,
                      std::vector<float> &mel_out) {
    const int C = hp.s2mel_n_mels; // 80
    const int D = hp.s2mel_lr_channels;
    const int Sty = hp.s2mel_style_dim;
    const int T_total = T_prompt + T_out;
    if (T_out <= 0 || T_total <= 0 ||
        (int)cond_512.size() != T_out * D ||
        (T_prompt > 0 && (int)prompt_condition.size() != T_prompt * D) ||
        (T_prompt > 0 && (int)prompt_mel.size() != T_prompt * C) ||
        (int)style.size() != Sty) {
        RS_LOG_ERR("[indextts2] cfm_euler: bad shapes "
                   "(T_prompt=%d T_out=%d cond=%zu prompt_cond=%zu "
                   "prompt_mel=%zu style=%zu)",
                   T_prompt, T_out, cond_512.size(), prompt_condition.size(),
                   prompt_mel.size(), style.size());
        return false;
    }

    std::vector<float> mu((size_t)T_total * D, 0.0f);
    if (T_prompt > 0) {
        std::memcpy(mu.data(), prompt_condition.data(),
                    (size_t)T_prompt * D * sizeof(float));
    }
    std::memcpy(mu.data() + (size_t)T_prompt * D, cond_512.data(),
                (size_t)T_out * D * sizeof(float));

    std::vector<float> prompt_x((size_t)T_total * C, 0.0f);
    if (T_prompt > 0) {
        std::memcpy(prompt_x.data(), prompt_mel.data(),
                    (size_t)T_prompt * C * sizeof(float));
    }

    // z = randn(0, 1) * temperature
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> z((size_t)T_total * C);
    for (float &v : z) v = nd(rng) * temperature;
    if (T_prompt > 0) {
        std::fill(z.begin(), z.begin() + (size_t)T_prompt * C, 0.0f);
    }

    std::vector<float> mu_zero((size_t)T_total * D, 0.0f);
    std::vector<float> prompt_zero((size_t)T_total * C, 0.0f);
    std::vector<float> style_zero((size_t)Sty, 0.0f);
    std::vector<float> dphi, cfg_dphi;
    const float dt = 1.0f / (float)std::max(1, n_steps);
    for (int step = 0; step < n_steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();
        const float t_now = (float)step / (float)std::max(1, n_steps);
        RS_LOG_INFO("[indextts2] S2Mel Euler step %d/%d: cond DiT "
                    "(T=%d, prompt=%d, t=%.4f)",
                    step + 1, n_steps, T_total, T_prompt, t_now);
        DiTInputs dc{};
        dc.x = z.data();
        dc.prompt_x = prompt_x.data();
        dc.style = style.data();
        dc.mu = mu.data();
        dc.t = t_now;
        dc.T = T_total;
        if (!dit_forward(sched, w, hp, dc, dphi)) {
            RS_LOG_ERR("[indextts2] cfm_euler: DiT cond step %d failed", step);
            return false;
        }

        RS_LOG_INFO("[indextts2] S2Mel Euler step %d/%d: uncond DiT",
                    step + 1, n_steps);
        DiTInputs du{};
        du.x = z.data();
        du.prompt_x = prompt_zero.data();
        du.style = style_zero.data();
        du.mu = mu_zero.data();
        du.t = t_now;
        du.T = T_total;
        if (!dit_forward(sched, w, hp, du, cfg_dphi)) {
            RS_LOG_ERR("[indextts2] cfm_euler: DiT cfg step %d failed", step);
            return false;
        }

        for (size_t i = 0; i < z.size(); ++i) {
            float dphi_dt = (1.0f + cfg_rate) * dphi[i] - cfg_rate * cfg_dphi[i];
            z[i] += dt * dphi_dt;
        }
        if (T_prompt > 0) {
            std::fill(z.begin(), z.begin() + (size_t)T_prompt * C, 0.0f);
        }
        const auto step_end = std::chrono::steady_clock::now();
        const double step_sec =
            std::chrono::duration<double>(step_end - step_start).count();
        RS_LOG_INFO("[indextts2] S2Mel Euler step %d/%d done in %.1f s",
                    step + 1, n_steps, step_sec);
    }
    mel_out.assign(z.begin() + (size_t)T_prompt * C, z.end());
    return true;
}

// -----------------------------------------------------------------------------
// Smoke-harness helpers: parse a tiny subset of the .npy v1.0 header so we can
// load the dump_s2mel_ref.py outputs without pulling in cnpy. We accept only
// C-order f32 arrays.
// -----------------------------------------------------------------------------
static bool load_npy_f32(const std::string &path,
                         std::vector<int> &shape,
                         std::vector<float> &data) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { RS_LOG_ERR("[s2mel-smoke] cannot open %s", path.c_str()); return false; }
    char magic[10];
    if (std::fread(magic, 1, 10, f) != 10 ||
        std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        RS_LOG_ERR("[s2mel-smoke] %s: not a .npy", path.c_str());
        std::fclose(f);
        return false;
    }
    uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
    std::vector<char> hdr(hlen);
    if (std::fread(hdr.data(), 1, hlen, f) != hlen) {
        std::fclose(f);
        return false;
    }
    std::string hs(hdr.begin(), hdr.end());
    if (hs.find("'<f4'") == std::string::npos ||
        hs.find("'fortran_order': False") == std::string::npos) {
        RS_LOG_ERR("[s2mel-smoke] %s: must be C-order float32", path.c_str());
        std::fclose(f);
        return false;
    }
    size_t shp = hs.find("'shape':");
    size_t lp  = hs.find('(', shp);
    size_t rp  = hs.find(')', lp);
    std::string shape_str = hs.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        std::string tok = shape_str.substr(pos, comma == std::string::npos
                                                  ? std::string::npos
                                                  : comma - pos);
        // strip spaces
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
        if (!tok.empty()) shape.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    size_t n = 1; for (int s : shape) n *= (size_t)s;
    data.assign(n, 0.0f);
    if (std::fread(data.data(), sizeof(float), n, f) != n) {
        RS_LOG_ERR("[s2mel-smoke] %s: short read", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

static bool dump_f32(const std::string &path, const std::vector<float> &v) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { RS_LOG_WARN("[s2mel-smoke] cannot open %s for write", path.c_str()); return false; }
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return true;
}

static std::string prod_dump_dir() {
    const char *d = std::getenv("RS_INDEXTTS2_DUMP_PROD_DIR");
    if (!d || !*d) return {};
    std::string dir = d;
    if (!dir.empty() && dir.back() != '/') dir.push_back('/');
    return dir;
}

static void dump_prod_f32(const std::string &dir, const char *name,
                          const std::vector<float> &v) {
    if (dir.empty()) return;
    dump_f32(dir + name + ".f32", v);
}

// Public entry: returns true if the smoke path was taken; sets `taken=true`
// in that case so Decode knows not to fall through to AR + vocoder.
bool Model::RunS2MelSmoke(State &s, ggml_backend_sched_t sched, bool &taken) {
    taken = false;
    const char *dir_c = std::getenv("RS_INDEXTTS2_S2MEL_TEST_DIR");
    if (!dir_c) return true;
    std::string dir = dir_c;
    if (!dir.empty() && dir.back() != '/') dir.push_back('/');
    taken = true;

    // -------- Load reference inputs --------
    std::vector<int> sh_l, sh_s, sh_p, sh_rm, sh_st, sh_si;
    std::vector<float> lm_latent, S_infer_pre, prompt_cond, ref_mel, style;
    if (!load_npy_f32(dir + "lm_latent.npy", sh_l, lm_latent))   return false;
    if (!load_npy_f32(dir + "S_infer_pre.npy", sh_si, S_infer_pre)) return false;
    if (!load_npy_f32(dir + "prompt_condition.npy", sh_p, prompt_cond)) return false;
    if (!load_npy_f32(dir + "ref_mel.npy", sh_rm, ref_mel))       return false;
    if (!load_npy_f32(dir + "style.npy", sh_st, style))           return false;

    if (sh_l.size() != 2 || sh_l[1] != hp_.n_embd) {
        RS_LOG_ERR("[s2mel-smoke] lm_latent shape mismatch: got [%d,%d], "
                   "want [*, %d]",
                   sh_l.size() >= 1 ? sh_l[0] : -1,
                   sh_l.size() >= 2 ? sh_l[1] : -1, hp_.n_embd);
        return false;
    }
    const int T_mel = sh_l[0];

    // -------- Stage 1: gpt_layer --------
    std::vector<float> gpt_emb;
    if (!run_gpt_layer(sched, backend_, s2mel_, hp_, lm_latent, T_mel, gpt_emb))
        return false;
    dump_f32(dir + "gpt_layer_out.f32", gpt_emb);

    // S_infer = S_infer_pre + gpt_layer_out (same as infer_v2.py line 650).
    // S_infer_pre is the residual that would normally come from the semantic
    // codec's vq2emb output.
    std::vector<float> S_infer = S_infer_pre;
    if (S_infer.size() != gpt_emb.size()) {
        RS_LOG_ERR("[s2mel-smoke] S_infer_pre size %zu != gpt_emb size %zu",
                   S_infer.size(), gpt_emb.size());
        return false;
    }
    for (size_t i = 0; i < S_infer.size(); ++i) S_infer[i] += gpt_emb[i];

    // -------- Stage 2: length_regulator --------
    const int T_out = (int)((double)T_mel * 1.72); // matches dump script
    std::vector<float> lr_out;
    if (!run_length_regulator(sched, s2mel_, hp_, S_infer, T_mel, T_out, lr_out))
        return false;
    dump_f32(dir + "lr_out.f32", lr_out);

    // -------- Stage 3: cat([prompt_cond, lr_out]) → DiT estimator step 0 --
    // Verify the DiT estimator against the upstream `dit_step0_out.npy` dump.
    // The CFM Euler loop itself is not yet wired through — we just check the
    // single-step v_pred against PyTorch for now.
    std::vector<int> sh_zp, sh_px, sh_cc;
    std::vector<float> z_after, prompt_x, cat_cond;
    bool have_dit_inputs = true;
    if (!load_npy_f32(dir + "z_after_prompt_paste.npy", sh_zp, z_after))
        have_dit_inputs = false;
    if (have_dit_inputs &&
        !load_npy_f32(dir + "prompt_x.npy", sh_px, prompt_x))
        have_dit_inputs = false;
    if (have_dit_inputs &&
        !load_npy_f32(dir + "cat_condition.npy", sh_cc, cat_cond))
        have_dit_inputs = false;

    if (have_dit_inputs && sh_zp.size() == 2 && sh_px.size() == 2 &&
        sh_cc.size() == 2) {
        const int Mel = hp_.s2mel_n_mels;       // 80
        const int Cnd = hp_.s2mel_lr_channels;  // 512
        const int Sty = hp_.s2mel_style_dim;    // 192
        if (sh_zp[0] != Mel || sh_px[0] != Mel ||
            sh_zp[1] != sh_px[1] || sh_zp[1] != sh_cc[0] ||
            sh_cc[1] != Cnd) {
            RS_LOG_ERR("[s2mel-smoke] DiT input shape mismatch "
                       "(z=[%d,%d] px=[%d,%d] cc=[%d,%d])",
                       sh_zp[0], sh_zp[1], sh_px[0], sh_px[1],
                       sh_cc[0], sh_cc[1]);
        } else {
            const int T_dit = sh_zp[1];
            // PyTorch saves (Mel, T) = (T, Mel) T-fast in numpy memory order,
            // which is (Mel, T) C-fast in our notation. We need (T, Mel) C-fast
            // — transpose Mel↔T.
            std::vector<float> x_TC, px_TC;
            transpose_2d(z_after,  Mel, T_dit, x_TC);
            transpose_2d(prompt_x, Mel, T_dit, px_TC);
            // cat_cond is already (T, 512) C-fast.
            if ((int)style.size() != Sty) {
                RS_LOG_ERR("[s2mel-smoke] style size %zu != %d",
                           style.size(), Sty);
            } else {
                // --- Step-0 single-shot verification (cond branch only) ---
                DiTInputs di{};
                di.x        = x_TC.data();
                di.prompt_x = px_TC.data();
                di.style    = style.data();
                di.mu       = cat_cond.data();
                di.t        = 0.0f;
                di.T        = T_dit;
                std::vector<float> dphi_TC;
                if (!dit_forward(sched, s2mel_, hp_, di, dphi_TC)) {
                    RS_LOG_ERR("[s2mel-smoke] dit_forward failed");
                } else {
                    // dit_forward returns (T, Mel) C-fast. Transpose back to
                    // (Mel, T) C-fast to match the upstream numpy dump.
                    std::vector<float> dphi_MT;
                    transpose_2d(dphi_TC, T_dit, Mel, dphi_MT);
                    dump_f32(dir + "dit_step0_out.f32", dphi_MT);
                    RS_LOG_INFO("[s2mel-smoke] dit_step0 dumped (T=%d, Mel=%d)",
                                T_dit, Mel);
                }

                // --- Multi-step CFM Euler with CFG (mirrors solve_euler) ---
                // n_steps / cfg are env-tunable so we can match the dump
                // script's defaults (n_steps=4, cfg=0.7).
                int   n_steps  = 4;
                float cfg_rate = 0.7f;
                if (const char *e = std::getenv("RS_INDEXTTS2_S2MEL_NSTEPS"))
                    n_steps = std::max(1, std::atoi(e));
                if (const char *e = std::getenv("RS_INDEXTTS2_S2MEL_CFG"))
                    cfg_rate = std::atof(e);

                // prompt_len = number of leading time-frames that hold the
                // reference prompt. lr_out has the non-prompt portion, so
                // prompt_len = T_dit - lr_T.
                const int prompt_len = T_dit - T_out;
                if (prompt_len < 0 || prompt_len > T_dit) {
                    RS_LOG_ERR("[s2mel-smoke] bad prompt_len=%d (T_dit=%d, "
                               "T_out=%d)", prompt_len, T_dit, T_out);
                } else {
                    // z is held in (T, Mel) C-fast; start from the dumped
                    // post-prompt-paste z (already prompt-zeroed).
                    std::vector<float> z = x_TC;
                    std::vector<float> mu_zero((size_t)T_dit * Cnd, 0.0f);
                    std::vector<float> px_zero((size_t)T_dit * Mel, 0.0f);
                    std::vector<float> style_zero((size_t)Sty, 0.0f);

                    const float dt = 1.0f / (float)n_steps;
                    std::vector<float> dphi_cond_TC, dphi_uncond_TC;
                    for (int step = 0; step < n_steps; ++step) {
                        const float t_now = (float)step / (float)n_steps;

                        DiTInputs dc{};
                        dc.x = z.data();    dc.prompt_x = px_TC.data();
                        dc.style = style.data(); dc.mu = cat_cond.data();
                        dc.t = t_now; dc.T = T_dit;
                        if (!dit_forward(sched, s2mel_, hp_, dc, dphi_cond_TC)) {
                            RS_LOG_ERR("[s2mel-smoke] dit_forward(cond) step %d failed", step);
                            break;
                        }

                        DiTInputs du{};
                        du.x = z.data();    du.prompt_x = px_zero.data();
                        du.style = style_zero.data(); du.mu = mu_zero.data();
                        du.t = t_now; du.T = T_dit;
                        if (!dit_forward(sched, s2mel_, hp_, du, dphi_uncond_TC)) {
                            RS_LOG_ERR("[s2mel-smoke] dit_forward(uncond) step %d failed", step);
                            break;
                        }

                        // dphi_dt = (1+cfg)*dphi - cfg*cfg_dphi ; z += dt*dphi_dt
                        for (size_t i = 0; i < z.size(); ++i) {
                            float d = (1.0f + cfg_rate) * dphi_cond_TC[i]
                                    - cfg_rate * dphi_uncond_TC[i];
                            z[i] += dt * d;
                        }
                        // z[..., :prompt_len] = 0 (zero first prompt_len rows).
                        std::fill(z.begin(),
                                  z.begin() + (size_t)prompt_len * Mel,
                                  0.0f);

                        // Dump dphi (cond), cfg_dphi (uncond), and x_after as
                        // (Mel, T) C-fast to match PT numpy convention.
                        std::vector<float> tmp_MT;
                        transpose_2d(dphi_cond_TC,   T_dit, Mel, tmp_MT);
                        dump_f32(dir + "dit_step" + std::to_string(step) + "_out.f32", tmp_MT);
                        transpose_2d(dphi_uncond_TC, T_dit, Mel, tmp_MT);
                        dump_f32(dir + "dit_step" + std::to_string(step) + "_cfg.f32", tmp_MT);
                        transpose_2d(z, T_dit, Mel, tmp_MT);
                        dump_f32(dir + "x_after_step" + std::to_string(step) + ".f32", tmp_MT);
                    }

                    // mel_out = z[:, :, prompt_len:] — drop prompt rows.
                    const int T_mel_out = T_dit - prompt_len;
                    std::vector<float> mel_TC(z.begin() + (size_t)prompt_len * Mel,
                                              z.end());
                    std::vector<float> mel_MT;
                    transpose_2d(mel_TC, T_mel_out, Mel, mel_MT);
                    dump_f32(dir + "mel_out.f32", mel_MT);
                    RS_LOG_INFO("[s2mel-smoke] cfm_euler n_steps=%d cfg=%.2f "
                                "prompt_len=%d T_out_mel=%d dumped",
                                n_steps, cfg_rate, prompt_len, T_mel_out);
                    // Expose the mel to the downstream Decode path so it can
                    // fall through to BigVGAN and produce real audio.
                    s.mel_pred = std::move(mel_TC);
                }
            }
        }
    } else {
        // DiT intermediate inputs not present — try to load mel_out.npy directly
        // so BigVGAN can still run and produce audio.
        (void)prompt_cond; (void)ref_mel; (void)style;
        std::vector<int> sh_mo;
        std::vector<float> mel_out_npy;
        if (load_npy_f32(dir + "mel_out.npy", sh_mo, mel_out_npy) &&
            sh_mo.size() == 2 && sh_mo[0] == hp_.s2mel_n_mels) {
            // mel_out.npy is (Mel, T) C-fast — transpose to (T, Mel) for s.mel_pred.
            const int Mel = sh_mo[0];
            const int T_mo = sh_mo[1];
            std::vector<float> mel_TC;
            transpose_2d(mel_out_npy, Mel, T_mo, mel_TC);
            s.mel_pred = std::move(mel_TC);
            RS_LOG_INFO("[s2mel-smoke] loaded mel_out.npy directly (Mel=%d T=%d) "
                        "→ forwarding to BigVGAN", Mel, T_mo);
        } else {
            RS_LOG_WARN("[s2mel-smoke] no DiT inputs and no mel_out.npy — "
                        "BigVGAN will not run");
        }
    }

    RS_LOG_INFO("[s2mel-smoke] outputs dumped to %s", dir.c_str());
    (void)s;
    return true;
}

// -----------------------------------------------------------------------------
// Public entry: orchestrate gpt_layer → length_regulator → CFM Euler.
// -----------------------------------------------------------------------------
bool Model::RunS2Mel(State &s, ggml_backend_sched_t sched) {
    if (s.mel_codes.empty()) {
        RS_LOG_ERR("[indextts2] RunS2Mel: no mel codes (RunAR must run first)");
        return false;
    }
    const std::string dump_dir = prod_dump_dir();

    const int T_mel = (int)s.mel_codes.size();
    if ((int)s.lm_latent.size() != T_mel * hp_.n_embd) {
        RS_LOG_WARN("[indextts2] lm_latent missing — using zero init "
                    "(T=%d, D=%d). Second AR pass not yet wired.",
                    T_mel, hp_.n_embd);
        s.lm_latent.assign((size_t)T_mel * hp_.n_embd, 0.0f);
    }

    // 1. gpt_layer: 1280 → 1024 (no activations between linears).
    std::vector<float> gpt_emb;
    if (!run_gpt_layer(sched, backend_, s2mel_, hp_, s.lm_latent, T_mel,
                       gpt_emb)) {
        return false;
    }
    dump_prod_f32(dump_dir, "lm_latent", s.lm_latent);
    dump_prod_f32(dump_dir, "gpt_layer_out", gpt_emb);

    // 2. S_infer = semantic_codec.vq2emb(codes) + gpt_layer(lm_latent).
    std::vector<float> s_infer = gpt_emb;
    std::vector<float> code_emb;
    if (semantic_codec_model_) {
        code_emb.assign((size_t)T_mel * hp_.s2mel_lr_in_channels, 0.0f);
        if (semantic_codec_model_->DecodeCodes(s.mel_codes.data(), T_mel,
                                                code_emb.data())) {
            for (size_t i = 0; i < s_infer.size(); ++i) s_infer[i] += code_emb[i];
        } else {
            RS_LOG_WARN("[indextts2] semantic_codec DecodeCodes failed — "
                        "using latent-only S_infer");
        }
    } else {
        RS_LOG_WARN("[indextts2] semantic_codec not loaded — using latent-only S_infer");
    }
    dump_prod_f32(dump_dir, "code_emb", code_emb);
    dump_prod_f32(dump_dir, "S_infer", s_infer);
    log_vec_stats("[indextts2] S2Mel S_infer", s_infer.data(), s_infer.size());

    // 2. length_regulator: target_lengths = (code_lens * 1.72).long() upstream.
    const int T_out = std::max(1, (int)((double)T_mel * 1.72));
    std::vector<float> cond;
    if (!run_length_regulator(sched, s2mel_, hp_, s_infer, T_mel, T_out, cond)) {
        return false;
    }
    dump_prod_f32(dump_dir, "cond", cond);
    log_vec_stats("[indextts2] S2Mel cond", cond.data(), cond.size());

    // 3. Reference prompt for S2Mel: mel_fn(ref audio), style, and
    // length_regulator(S_ref, target=ref_mel_T).
    std::vector<float> ref_mel;
    int T_prompt = 0;
    if (!s.ref_audio_22k.empty()) {
        T_prompt = compute_indextts2_ref_mel(hp_, s.ref_audio_22k, ref_mel);
        if (T_prompt <= 0) {
            RS_LOG_WARN("[indextts2] S2Mel: reference mel extraction failed — "
                        "running without prompt mel");
            ref_mel.clear();
            T_prompt = 0;
        }
    }
    if (!ref_mel.empty()) {
        dump_prod_f32(dump_dir, "ref_mel", ref_mel);
        log_vec_stats("[indextts2] S2Mel ref_mel", ref_mel.data(),
                      ref_mel.size());
    }

    std::vector<float> prompt_condition;
    if (T_prompt > 0 && !s.sc_hidden.empty() && s.T_sc > 0 &&
        (int)s.sc_hidden.size() == s.T_sc * hp_.s2mel_lr_in_channels) {
        if (!run_length_regulator(sched, s2mel_, hp_, s.sc_hidden, s.T_sc,
                                  T_prompt, prompt_condition)) {
            RS_LOG_ERR("[indextts2] S2Mel: prompt length_regulator failed");
            return false;
        }
        dump_prod_f32(dump_dir, "prompt_condition", prompt_condition);
    } else {
        if (T_prompt > 0) {
            RS_LOG_WARN("[indextts2] S2Mel: S_ref missing — running without "
                        "prompt conditioning");
        }
        T_prompt = 0;
        ref_mel.clear();
        prompt_condition.clear();
    }

    std::vector<float> style((size_t)hp_.s2mel_style_dim, 0.0f);
    if ((int)s.spk_embedding.size() == hp_.s2mel_style_dim) {
        style = s.spk_embedding;
    } else {
        RS_LOG_WARN("[indextts2] S2Mel: CAMPPlus style missing — using zeros");
    }
    dump_prod_f32(dump_dir, "style", style);

    // 4. CFM Euler solver with CFG through the real DiT estimator.
    int         n_steps      = 6;      // upstream 25; 6 near-transparent, 3-4 acceptable
    const float cfg_rate     = 0.7f;   // upstream default 0.7
    const float temperature  = 1.0f;
    if (const char *e = std::getenv("RS_INDEXTTS2_S2MEL_NSTEPS")) {
        n_steps = std::max(1, std::atoi(e));
    }
    RS_LOG_INFO("[indextts2] S2Mel CFM start: output=%d prompt=%d total=%d "
                "n_steps=%d cfg=%.2f",
                T_out, T_prompt, T_prompt + T_out, n_steps, cfg_rate);
    if (!cfm_euler(sched, s2mel_, hp_, s.rng, prompt_condition, cond,
                   ref_mel, style, T_prompt, T_out, n_steps, cfg_rate,
                   temperature, s.mel_pred)) {
        return false;
    }
    log_vec_stats("[indextts2] S2Mel mel_pred", s.mel_pred.data(),
                  s.mel_pred.size());
    dump_prod_f32(dump_dir, "mel_pred", s.mel_pred);
    RS_LOG_INFO("[indextts2] S2Mel produced mel of shape [%d, %d] "
                "(prompt=%d, n_steps=%d)",
                T_out, hp_.s2mel_n_mels, T_prompt, n_steps);
    return true;
}

} // namespace indextts2
