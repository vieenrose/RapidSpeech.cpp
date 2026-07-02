#include "indextts2_bigvgan_metal.h"
#include "indextts2.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace indextts2 {

// =====================================================================
// Metal shader library. T-major layout: element(t,c) at index t + c*T.
// =====================================================================
static const char *kBVShaderSrc = R"MTL(
#include <metal_stdlib>
using namespace metal;

// Conv1d with simdgroup reduction over (K*IC): one 32-lane threadgroup per
// output element, lanes split the K*IC MAC work, simd_sum reduces. Much faster
// than the naive kernel when IC is large (resblock convs, IC up to 1536).
kernel void bv_conv1d_simd(device float *out       [[buffer(0)]],
                           device const float *inp [[buffer(1)]],
                           device const float *w   [[buffer(2)]],
                           device const float *b   [[buffer(3)]],
                           constant uint &T_in     [[buffer(4)]],
                           constant uint &C_in     [[buffer(5)]],
                           constant uint &T_out    [[buffer(6)]],
                           constant uint &C_out    [[buffer(7)]],
                           constant uint &K        [[buffer(8)]],
                           constant uint &pad      [[buffer(9)]],
                           constant uint &dilation [[buffer(10)]],
                           uint tg   [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    uint t = tg % T_out, oc = tg / T_out;
    if (oc >= C_out) return;
    uint work = K * C_in;
    float partial = 0.0f;
    for (uint idx = lane; idx < work; idx += 32) {
        uint k = idx / C_in, ic = idx % C_in;
        int t_in = (int)t - (int)pad + (int)k * (int)dilation;
        if (t_in < 0 || t_in >= (int)T_in) continue;
        partial += inp[t_in + ic * T_in] * w[oc + ic * C_out + k * (C_out * C_in)];
    }
    float sum = simd_sum(partial);
    if (lane == 0) out[t + oc * T_out] = sum + b[oc];
}

// Conv1d — weight [OC, IC, K] (OC fastest). in/out T-major [T, C].
kernel void bv_conv1d(device float *out       [[buffer(0)]],
                      device const float *inp [[buffer(1)]],
                      device const float *w   [[buffer(2)]],
                      device const float *b   [[buffer(3)]],
                      constant uint &T_in     [[buffer(4)]],
                      constant uint &C_in     [[buffer(5)]],
                      constant uint &T_out    [[buffer(6)]],
                      constant uint &C_out    [[buffer(7)]],
                      constant uint &K        [[buffer(8)]],
                      constant uint &pad      [[buffer(9)]],
                      constant uint &dilation [[buffer(10)]],
                      uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T_out || oc >= C_out) return;
    float sum = b[oc];
    for (uint k = 0; k < K; k++) {
        int t_in = (int)t - (int)pad + (int)k * (int)dilation;
        if (t_in < 0 || t_in >= (int)T_in) continue;
        uint w_base = oc + k * (C_out * C_in);
        for (uint ic = 0; ic < C_in; ic++)
            sum += inp[t_in + ic * T_in] * w[w_base + ic * C_out];
    }
    out[t + oc * T_out] = sum;
}

// ConvTranspose1d — weight [IC, OC, K] (IC fastest). in/out T-major.
kernel void bv_conv_transpose1d(device float *out       [[buffer(0)]],
                                device const float *inp [[buffer(1)]],
                                device const float *w   [[buffer(2)]],
                                device const float *b   [[buffer(3)]],
                                constant uint &T_in     [[buffer(4)]],
                                constant uint &C_in     [[buffer(5)]],
                                constant uint &T_out    [[buffer(6)]],
                                constant uint &C_out    [[buffer(7)]],
                                constant uint &K        [[buffer(8)]],
                                constant uint &stride   [[buffer(9)]],
                                constant uint &pad      [[buffer(10)]],
                                uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T_out || oc >= C_out) return;
    float sum = b[oc];
    for (uint k = 0; k < K; k++) {
        int t_in_raw = (int)t - (int)k + (int)pad;
        if (t_in_raw < 0 || (t_in_raw % (int)stride) != 0) continue;
        int t_in = t_in_raw / (int)stride;
        if (t_in < 0 || t_in >= (int)T_in) continue;
        uint w_base = k * (C_in * C_out);
        for (uint ic = 0; ic < C_in; ic++)
            sum += inp[t_in + ic * T_in] * w[ic + oc * C_in + w_base];
    }
    out[t + oc * T_out] = sum;
}

// Anti-aliased Snake, part 1: FIR upsample 2x (Kaiser, ratio scale 2) fused with
// snakebeta. Input x [T, C] → output y [2T, C].
// Geometry (ratio=2, K=12): replicate-pad(5,5) → zero-stuff to 2L-1 (L=T+10) →
// pad(K-1) each side → FIR → crop 15 → length 2T. snakebeta: y=a+sin(ea*a)^2*ib.
kernel void bv_aa_up_snake(device float *out        [[buffer(0)]],
                           device const float *x    [[buffer(1)]],
                           device const float *h    [[buffer(2)]],  // fir_up [K]
                           constant float *a_exp    [[buffer(3)]],  // exp(alpha)[C]
                           constant float *b_inv    [[buffer(4)]],  // exp(-beta)[C]
                           constant uint &T         [[buffer(5)]],
                           constant uint &C         [[buffer(6)]],
                           constant uint &K         [[buffer(7)]],
                           uint2 gid               [[thread_position_in_grid]]) {
    uint t2 = gid.x, c = gid.y;
    uint T2 = 2 * T;
    if (t2 >= T2 || c >= C) return;
    int stuffed_len = (int)(2 * T + 19);   // 2L-1, L=T+10
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        int p = (int)t2 + 15 + (int)k;      // into stuffed_pad
        int j = p - 11;                     // into stuffed
        if (j < 0 || j >= stuffed_len) continue;
        if (j & 1) continue;                // odd stuffed slot = 0
        int i = j >> 1;                     // into x_padded (len L=T+10)
        int xi = i - 5;                     // undo left pad
        xi = clamp(xi, 0, (int)T - 1);      // replicate pad
        acc += h[k] * x[xi + (int)c * (int)T];
    }
    acc *= 2.0f;
    float tv = acc * a_exp[c];
    float sn = sin(tv);
    out[t2 + c * T2] = acc + sn * sn * b_inv[c];
}

// Anti-aliased Snake, part 2: FIR downsample 2x. Input u [2T, C] → out [T, C].
// replicate-pad(5,6) → strided FIR (stride 2).
kernel void bv_aa_down(device float *out        [[buffer(0)]],
                       device const float *u    [[buffer(1)]],
                       device const float *h    [[buffer(2)]],  // fir_down [K]
                       constant uint &T         [[buffer(3)]],  // output T
                       constant uint &C         [[buffer(4)]],
                       constant uint &K         [[buffer(5)]],
                       uint2 gid               [[thread_position_in_grid]]) {
    uint t = gid.x, c = gid.y;
    if (t >= T || c >= C) return;
    int U = (int)(2 * T);
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        int m = 2 * (int)t + (int)k;
        int ui = m - 5;                     // undo left pad(5)
        ui = clamp(ui, 0, U - 1);           // replicate pad
        acc += h[k] * u[ui + (int)c * U];
    }
    out[t + c * (int)T] = acc;
}

// dst[i] += src[i]
kernel void bv_add_into(device float *dst       [[buffer(0)]],
                        device const float *src [[buffer(1)]],
                        constant uint &N        [[buffer(2)]],
                        uint idx               [[thread_position_in_grid]]) {
    if (idx >= N) return;
    dst[idx] += src[idx];
}

// x[i] *= s
kernel void bv_scale(device float *x       [[buffer(0)]],
                     constant float &s     [[buffer(1)]],
                     constant uint &N      [[buffer(2)]],
                     uint idx             [[thread_position_in_grid]]) {
    if (idx >= N) return;
    x[idx] *= s;
}

// im2col for Conv1d: input [T, C_in] T-major → col [IC*K, T] row-major
// (row r = ic*K + k, contiguous in t). Then MPS GEMM W[OC,IC*K]×col = out[OC,T].
kernel void bv_im2col(device float *col        [[buffer(0)]],
                      device const float *inp  [[buffer(1)]],
                      constant uint &T         [[buffer(2)]],
                      constant uint &C_in      [[buffer(3)]],
                      constant uint &K         [[buffer(4)]],
                      constant uint &pad       [[buffer(5)]],
                      constant uint &dilation  [[buffer(6)]],
                      constant uint &row_stride [[buffer(7)]],  // >= T, 16B-aligned
                      uint2 gid               [[thread_position_in_grid]]) {
    uint t = gid.x, r = gid.y;          // r = ic*K + k
    uint KIC = C_in * K;
    if (t >= T || r >= KIC) return;
    uint ic = r / K, k = r % K;
    int t_in = (int)t - (int)pad + (int)k * (int)dilation;
    float v = (t_in >= 0 && t_in < (int)T) ? inp[t_in + ic * T] : 0.0f;
    col[r * row_stride + t] = v;
}

// Add per-channel bias to a T-major [T, OC] buffer (MPS GEMM output).
kernel void bv_add_bias(device float *out       [[buffer(0)]],
                        device const float *b   [[buffer(1)]],
                        constant uint &T         [[buffer(2)]],
                        constant uint &OC        [[buffer(3)]],
                        uint2 gid               [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T || oc >= OC) return;
    out[t + oc * T] += b[oc];
}

// Tiled GEMM: C[M,N] = A[M,K] * B[K,N] (all row-major). 16x16 threadgroup tiles
// with shared memory — coalesced, robust, and stays in our compute encoder (no
// MPS, no cross-encoder sync). Used for Conv1d as W[OC,KIC] * col[KIC,T].
kernel void bv_gemm(device float *C        [[buffer(0)]],
                    device const float *A  [[buffer(1)]],
                    device const float *B  [[buffer(2)]],
                    constant uint &M       [[buffer(3)]],
                    constant uint &N       [[buffer(4)]],
                    constant uint &K       [[buffer(5)]],
                    constant uint &lda     [[buffer(6)]],
                    constant uint &ldb     [[buffer(7)]],
                    constant uint &ldc     [[buffer(8)]],
                    uint2 tg  [[threadgroup_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];
    uint row = tg.y * 16 + lid.y;   // M index
    uint col = tg.x * 16 + lid.x;   // N index
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += 16) {
        As[lid.y][lid.x] = (row < M && (k0 + lid.x) < K) ? A[row * lda + k0 + lid.x] : 0.0f;
        Bs[lid.y][lid.x] = ((k0 + lid.y) < K && col < N) ? B[(k0 + lid.y) * ldb + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 16; kk++) acc += As[lid.y][kk] * Bs[kk][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * ldc + col] = acc;
}
)MTL";

// =====================================================================
// ObjC bridging helpers
// =====================================================================
static inline id<MTLDevice>               dev(void *p) { return (__bridge id<MTLDevice>)p; }
static inline id<MTLLibrary>              lib(void *p) { return (__bridge id<MTLLibrary>)p; }
static inline id<MTLCommandQueue>         q(void *p)   { return (__bridge id<MTLCommandQueue>)p; }
static inline id<MTLComputePipelineState> ps(void *p)  { return (__bridge id<MTLComputePipelineState>)p; }
static inline id<MTLBuffer>               buf(void *p) { return (__bridge id<MTLBuffer>)p; }

static inline void *retain_id(id x) {
#if __has_feature(objc_arc)
    return (__bridge_retained void *)x;
#else
    return (__bridge void *)[x retain];
#endif
}
static inline void release_id(void *p) {
    if (!p) return;
#if __has_feature(objc_arc)
    (void)(__bridge_transfer id)p;
#else
    [(__bridge id)p release];
#endif
}

// =====================================================================
// Ctor / dtor
// =====================================================================
BigVGANMetalDecoder::BigVGANMetalDecoder() {}
BigVGANMetalDecoder::~BigVGANMetalDecoder() {
    auto rel = [](void *&p){ release_id(p); p = nullptr; };
    rel(pipe_conv1d_); rel(pipe_conv1d_simd_); rel(pipe_conv_t_); rel(pipe_up_snake_);
    rel(pipe_down_); rel(pipe_add_into_); rel(pipe_axpy3_);
    rel(pipe_im2col_); rel(pipe_add_bias_); rel(pipe_gemm_);
    rel(buf_fir_up_); rel(buf_fir_down_);
    auto relc = [&](Conv &c){ rel(c.w); rel(c.b); rel(c.w_mps); };
    auto rels = [&](Snake &s){ rel(s.a_exp); rel(s.b_inv); };
    relc(conv_pre_); relc(conv_post_); rels(act_post_);
    for (auto &st : stages_) {
        relc(st.ct);
        for (auto &rb : st.rb)
            for (int i = 0; i < 3; ++i) { rels(rb.a1[i]); rels(rb.a2[i]); relc(rb.c1[i]); relc(rb.c2[i]); }
    }
    rel(queue_); rel(library_); rel(device_);
}

// =====================================================================
// Weight helpers
// =====================================================================
void *BigVGANMetalDecoder::get_fn(const char *name) {
    id<MTLFunction> fn = [lib(library_) newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { RS_LOG_ERR("BigVGANMetal: kernel '%s' not found", name); return nullptr; }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev(device_) newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { RS_LOG_ERR("BigVGANMetal: pipeline '%s': %s", name,
                         [[err localizedDescription] UTF8String]); return nullptr; }
    return retain_id(p);
}

void *BigVGANMetalDecoder::alloc_buffer(size_t bytes, const void *data) {
    id<MTLBuffer> b = [dev(device_) newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (data) memcpy([b contents], data, bytes);
    return retain_id(b);
}

// Read a ggml tensor (possibly on a GPU backend, possibly f16) into host f32.
std::vector<float> BigVGANMetalDecoder::read_host(const ggml_tensor *t) {
    if (!t) return {};
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), raw.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
    } else {
        std::fill(out.begin(), out.end(), 0.0f);
    }
    return out;
}

void *BigVGANMetalDecoder::upload_vec(const std::vector<float> &v) {
    return v.empty() ? nullptr : alloc_buffer(v.size() * sizeof(float), v.data());
}

// ggml Conv1d weight [K, IC, OC] (K fastest) → Metal [OC, IC, K] (OC fastest).
void *BigVGANMetalDecoder::upload_conv_w(const ggml_tensor *t, int K, int IC, int OC) {
    std::vector<float> src = read_host(t);
    if ((int)src.size() != K * IC * OC) {
        RS_LOG_ERR("BigVGANMetal: conv_w size %zu != %d*%d*%d", src.size(), K, IC, OC);
        return nullptr;
    }
    std::vector<float> b((size_t)K * IC * OC);
    for (int k = 0; k < K; k++)
        for (int ic = 0; ic < IC; ic++)
            for (int oc = 0; oc < OC; oc++)
                b[(size_t)oc + ic * OC + (size_t)k * OC * IC] =
                    src[(size_t)k + ic * K + (size_t)oc * K * IC];
    return alloc_buffer(b.size() * sizeof(float), b.data());
}

// ggml Conv1d weight [K, IC, OC] → MPS GEMM weight [OC, wmps_row] row-major,
// wmps_row = KIC padded to 4 floats (16 B) — MPS mis-reads tightly-packed rows
// for some sizes (produces NaN). Padding columns are zero (unread by GEMM but
// keep the buffer well-defined). Returns the padded row stride via *row_out.
void *BigVGANMetalDecoder::upload_conv_w_mps(const ggml_tensor *t, int K, int IC, int OC) {
    std::vector<float> src = read_host(t);
    if ((int)src.size() != K * IC * OC) return nullptr;
    const int KIC = IC * K;
    const int row = (KIC + 3) & ~3;      // 16-byte aligned row stride
    std::vector<float> b((size_t)OC * row, 0.0f);
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                b[(size_t)oc * row + (size_t)ic * K + k] =
                    src[(size_t)k + ic * K + (size_t)oc * K * IC];
    return alloc_buffer(b.size() * sizeof(float), b.data());
}

// ggml ConvTranspose weight [K, OC, IC] (K fastest) → Metal [IC, OC, K] (IC fastest).
void *BigVGANMetalDecoder::upload_conv_t_w(const ggml_tensor *t, int K, int IC, int OC) {
    std::vector<float> src = read_host(t);
    if ((int)src.size() != K * IC * OC) {
        RS_LOG_ERR("BigVGANMetal: conv_t_w size %zu != %d*%d*%d", src.size(), K, IC, OC);
        return nullptr;
    }
    std::vector<float> b((size_t)K * IC * OC);
    for (int k = 0; k < K; k++)
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                b[(size_t)ic + oc * IC + (size_t)k * IC * OC] =
                    src[(size_t)k + oc * K + (size_t)ic * K * OC];
    return alloc_buffer(b.size() * sizeof(float), b.data());
}

bool BigVGANMetalDecoder::load_snake(const BigVGANWeights &bv,
                                     const std::string &alpha_key,
                                     const std::string &beta_key, Snake &out) {
    auto ita = bv.tensors.find(alpha_key), itb = bv.tensors.find(beta_key);
    if (ita == bv.tensors.end() || itb == bv.tensors.end()) {
        RS_LOG_ERR("BigVGANMetal: missing snake %s / %s", alpha_key.c_str(), beta_key.c_str());
        return false;
    }
    std::vector<float> a = read_host(ita->second);
    std::vector<float> be = read_host(itb->second);
    out.C = (int)a.size();
    std::vector<float> a_exp(a.size()), b_inv(be.size());
    for (size_t i = 0; i < a.size(); ++i) a_exp[i] = std::exp(a[i]);
    for (size_t i = 0; i < be.size(); ++i) b_inv[i] = std::exp(-be[i]); // 1/exp(beta)
    out.a_exp = upload_vec(a_exp);
    out.b_inv = upload_vec(b_inv);
    return out.a_exp && out.b_inv;
}

// wkey/bkey Conv1d (transpose=false) or ConvTranspose1d (transpose=true).
// Reads dims from the weight ne[]: Conv1d [K,IC,OC]; ConvT [K,OC,IC].
bool BigVGANMetalDecoder::load_conv(const BigVGANWeights &bv,
                                    const std::string &wkey, const std::string &bkey,
                                    int K, Conv &out, bool transpose) {
    auto itw = bv.tensors.find(wkey);
    if (itw == bv.tensors.end()) { RS_LOG_ERR("BigVGANMetal: missing %s", wkey.c_str()); return false; }
    const ggml_tensor *w = itw->second;
    out.K = (int)w->ne[0];
    if (out.K != K && K > 0) out.K = K;
    if (transpose) { out.OC = (int)w->ne[1]; out.IC = (int)w->ne[2]; out.w = upload_conv_t_w(w, out.K, out.IC, out.OC); }
    else           { out.IC = (int)w->ne[1]; out.OC = (int)w->ne[2]; out.w = upload_conv_w(w, out.K, out.IC, out.OC);
                     out.w_mps = upload_conv_w_mps(w, out.K, out.IC, out.OC);
                     out.wmps_row = (out.IC * out.K + 3) & ~3; }
    if (!out.w) return false;
    auto itb = bv.tensors.find(bkey);
    if (itb != bv.tensors.end()) out.b = upload_vec(read_host(itb->second));
    if (!out.b) out.b = upload_vec(std::vector<float>((size_t)out.OC, 0.0f)); // zero bias
    return out.b != nullptr;
}

// =====================================================================
// init
// =====================================================================
bool BigVGANMetalDecoder::init(const BigVGANWeights &bv, const HParams &hp) {
    if (valid_) return true;
    @autoreleasepool {
        device_ = retain_id(MTLCreateSystemDefaultDevice());
        if (!device_) { RS_LOG_ERR("BigVGANMetal: no Metal device"); return false; }
        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> l = [dev(device_) newLibraryWithSource:[NSString stringWithUTF8String:kBVShaderSrc]
                            options:opts error:&err];
        if (!l) { RS_LOG_ERR("BigVGANMetal: compile failed: %s", [[err localizedDescription] UTF8String]); return false; }
        library_ = retain_id(l);
        queue_ = retain_id([dev(device_) newCommandQueue]);

        pipe_conv1d_   = get_fn("bv_conv1d");
        pipe_conv1d_simd_ = get_fn("bv_conv1d_simd");
        pipe_conv_t_   = get_fn("bv_conv_transpose1d");
        pipe_up_snake_ = get_fn("bv_aa_up_snake");
        pipe_down_     = get_fn("bv_aa_down");
        pipe_add_into_ = get_fn("bv_add_into");
        pipe_axpy3_    = get_fn("bv_scale");
        pipe_im2col_   = get_fn("bv_im2col");
        pipe_add_bias_ = get_fn("bv_add_bias");
        pipe_gemm_     = get_fn("bv_gemm");
        if (!pipe_conv1d_ || !pipe_conv1d_simd_ || !pipe_conv_t_ || !pipe_up_snake_ ||
            !pipe_down_ || !pipe_add_into_ || !pipe_axpy3_ || !pipe_im2col_ ||
            !pipe_add_bias_ || !pipe_gemm_) return false;

        // FIR filters (shared) — resblocks.0 copy, as in RunBigVGAN.
        {
            auto u = bv.tensors.find("resblocks.0.activations.0.upsample.filter");
            auto d = bv.tensors.find("resblocks.0.activations.0.downsample.lowpass.filter");
            if (u == bv.tensors.end() || d == bv.tensors.end()) {
                RS_LOG_ERR("BigVGANMetal: missing Kaiser FIR filters"); return false;
            }
            std::vector<float> fu = read_host(u->second), fd = read_host(d->second);
            fir_K_ = (int)fu.size();
            buf_fir_up_ = upload_vec(fu);
            buf_fir_down_ = upload_vec(fd);
            if (!buf_fir_up_ || !buf_fir_down_ || (int)fd.size() != fir_K_) {
                RS_LOG_ERR("BigVGANMetal: bad FIR (K=%d)", fir_K_); return false;
            }
        }

        // conv_pre: Conv1d(n_mels → initial, K=7).
        if (!load_conv(bv, "conv_pre.weight", "conv_pre.bias", 7, conv_pre_, false)) return false;

        const std::vector<int> &ur = hp.bigvgan_upsample_rates;
        const std::vector<int> &rk = hp.bigvgan_resblock_kernel_sizes;
        const int num_up = (int)ur.size();
        stages_.resize(num_up);
        for (int i = 0; i < num_up; ++i) {
            UpStage &st = stages_[i];
            const std::string up = "ups." + std::to_string(i) + ".0";
            if (!load_conv(bv, up + ".weight", up + ".bias", 2 * ur[i], st.ct, true)) return false;
            st.ct.stride = ur[i];
            st.ct.pad    = (st.ct.K - ur[i]) / 2;
            for (int j = 0; j < 3; ++j) {
                ResBlock &rb = st.rb[j];
                rb.K = rk[j];
                const std::string pfx = "resblocks." + std::to_string(i * 3 + j);
                static const int dil[3] = {1, 3, 5};
                for (int idx = 0; idx < 3; ++idx) {
                    if (!load_snake(bv, pfx + ".activations." + std::to_string(2 * idx) + ".act.alpha",
                                        pfx + ".activations." + std::to_string(2 * idx) + ".act.beta", rb.a1[idx])) return false;
                    if (!load_snake(bv, pfx + ".activations." + std::to_string(2 * idx + 1) + ".act.alpha",
                                        pfx + ".activations." + std::to_string(2 * idx + 1) + ".act.beta", rb.a2[idx])) return false;
                    if (!load_conv(bv, pfx + ".convs1." + std::to_string(idx) + ".weight",
                                       pfx + ".convs1." + std::to_string(idx) + ".bias", rb.K, rb.c1[idx], false)) return false;
                    rb.c1[idx].dilation = dil[idx];
                    rb.c1[idx].pad = dil[idx] * (rb.K - 1) / 2;
                    if (!load_conv(bv, pfx + ".convs2." + std::to_string(idx) + ".weight",
                                       pfx + ".convs2." + std::to_string(idx) + ".bias", rb.K, rb.c2[idx], false)) return false;
                    rb.c2[idx].dilation = 1;
                    rb.c2[idx].pad = (rb.K - 1) / 2;
                }
            }
        }

        // activation_post + conv_post.
        if (!load_snake(bv, "activation_post.act.alpha", "activation_post.act.beta", act_post_)) return false;
        if (!load_conv(bv, "conv_post.weight", "conv_post.bias", 7, conv_post_, false)) return false;
        conv_pre_.pad = 3; conv_pre_.dilation = 1;
        conv_post_.pad = 3; conv_post_.dilation = 1;
        use_tanh_final_ = hp.bigvgan_use_tanh_at_final;

        valid_ = true;
        RS_LOG_INFO("BigVGANMetal: ready — %d up-stages, K_fir=%d, tanh=%d",
                    num_up, fir_K_, (int)use_tanh_final_);
        return true;
    }
}

// =====================================================================
// decode — single command buffer, memory barriers between dependent dispatches.
// =====================================================================
bool BigVGANMetalDecoder::decode(const float *mel, int n_mels, int T_mel,
                                 std::vector<float> &audio_out, bool use_mps) {
    if (!valid_) return false;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        NSMutableArray *keep = [NSMutableArray array]; // retain scratch buffers until commit

        auto barrier = [&]{ [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
        auto newbuf = [&](size_t n) -> id<MTLBuffer> {
            id<MTLBuffer> b = [dev(device_) newBufferWithLength:n * sizeof(float)
                               options:MTLResourceStorageModeShared];
            [keep addObject:b];
            return b;
        };

        // Reused im2col column buffer (convs run sequentially; command-buffer
        // ordering + Metal hazard tracking make reuse safe). Grows on demand.
        id<MTLBuffer> colbuf = nil;
        size_t colcap = 0;

        // Conv1d via im2col + MPS GEMM: W[OC, IC*K] × col[IC*K, T] → out[OC, T]
        // (row-major result == our T-major [T, OC]). This replaces the naive
        // strided-memory kernel (memory-bound) with Apple's optimized GEMM.
        auto conv1d = [&](const Conv &c, id<MTLBuffer> x, int T) -> id<MTLBuffer> {
            if (!use_mps) {
                // Naive strided-IC conv kernel (robust for all sizes).
                id<MTLBuffer> y = newbuf((size_t)T * c.OC);
                [enc setComputePipelineState:ps(pipe_conv1d_)];
                [enc setBuffer:y offset:0 atIndex:0];
                [enc setBuffer:x offset:0 atIndex:1];
                [enc setBuffer:buf(c.w) offset:0 atIndex:2];
                [enc setBuffer:buf(c.b) offset:0 atIndex:3];
                uint Ti=(uint)T, Ci=(uint)c.IC, To=(uint)T, Co=(uint)c.OC, K=(uint)c.K, pad=(uint)c.pad, dil=(uint)c.dilation;
                [enc setBytes:&Ti length:4 atIndex:4]; [enc setBytes:&Ci length:4 atIndex:5];
                [enc setBytes:&To length:4 atIndex:6]; [enc setBytes:&Co length:4 atIndex:7];
                [enc setBytes:&K length:4 atIndex:8]; [enc setBytes:&pad length:4 atIndex:9];
                [enc setBytes:&dil length:4 atIndex:10];
                NSUInteger mt = ps(pipe_conv1d_).maxTotalThreadsPerThreadgroup;
                NSUInteger tx = std::min<NSUInteger>(T, mt), ty = std::max<NSUInteger>(1, std::min<NSUInteger>(c.OC, mt / tx));
                [enc dispatchThreads:MTLSizeMake(T, c.OC, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
                barrier();
                return y;
            }
            const uint KIC = (uint)(c.IC * c.K);
            // Pad the col row stride to a multiple of 4 floats (16 bytes).
            const uint rowB = ((uint)T + 3u) & ~3u;   // >= T, 16B-aligned
            size_t need = (size_t)KIC * (size_t)rowB;
            // Reuse ONE col buffer (grown on demand): the tiled-GEMM path keeps
            // everything in one compute encoder with barriers, so conv i's GEMM
            // read is ordered before conv i+1's im2col write — reuse is safe and
            // avoids the multi-GB blow-up of one fresh col per conv at high T.
            if (!colbuf || colcap < need) {
                colbuf = [dev(device_) newBufferWithLength:need * sizeof(float)
                          options:MTLResourceStorageModeShared];
                [keep addObject:colbuf];
                colcap = need;
            }
            id<MTLBuffer> colbuf2 = colbuf;
            // im2col on the current compute encoder.
            [enc setComputePipelineState:ps(pipe_im2col_)];
            [enc setBuffer:colbuf2 offset:0 atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1];
            uint Tt=(uint)T, Ci=(uint)c.IC, K=(uint)c.K, pad=(uint)c.pad, dil=(uint)c.dilation, rstride=rowB;
            [enc setBytes:&Tt length:4 atIndex:2]; [enc setBytes:&Ci length:4 atIndex:3];
            [enc setBytes:&K length:4 atIndex:4]; [enc setBytes:&pad length:4 atIndex:5];
            [enc setBytes:&dil length:4 atIndex:6]; [enc setBytes:&rstride length:4 atIndex:7];
            {
                NSUInteger mt = ps(pipe_im2col_).maxTotalThreadsPerThreadgroup;
                NSUInteger tx = std::min<NSUInteger>(T, mt), ty = std::max<NSUInteger>(1, std::min<NSUInteger>(KIC, mt / tx));
                [enc dispatchThreads:MTLSizeMake(T, KIC, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
            }
            barrier();
            // GEMM: y[OC,T] = w_mps[OC,KIC] * col[KIC,T], via our own tiled kernel
            // (stays in this compute encoder — no MPS, no cross-encoder sync, no
            // NaN). ldc = T (tight, so y is contiguous T-major for downstream).
            id<MTLBuffer> y = newbuf((size_t)T * c.OC);
            [enc setComputePipelineState:ps(pipe_gemm_)];
            [enc setBuffer:y offset:0 atIndex:0];
            [enc setBuffer:buf(c.w_mps) offset:0 atIndex:1];
            [enc setBuffer:colbuf2 offset:0 atIndex:2];
            uint gM=(uint)c.OC, gN=(uint)T, gK=KIC, glda=(uint)c.wmps_row, gldb=rowB, gldc=(uint)T;
            [enc setBytes:&gM length:4 atIndex:3]; [enc setBytes:&gN length:4 atIndex:4];
            [enc setBytes:&gK length:4 atIndex:5]; [enc setBytes:&glda length:4 atIndex:6];
            [enc setBytes:&gldb length:4 atIndex:7]; [enc setBytes:&gldc length:4 atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)T + 15) / 16, ((NSUInteger)c.OC + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            barrier();
            // bias-add.
            [enc setComputePipelineState:ps(pipe_add_bias_)];
            [enc setBuffer:y offset:0 atIndex:0];
            [enc setBuffer:buf(c.b) offset:0 atIndex:1];
            uint To=(uint)T, Co=(uint)c.OC;
            [enc setBytes:&To length:4 atIndex:2]; [enc setBytes:&Co length:4 atIndex:3];
            {
                NSUInteger mt = ps(pipe_add_bias_).maxTotalThreadsPerThreadgroup;
                NSUInteger tx = std::min<NSUInteger>(T, mt), ty = std::max<NSUInteger>(1, std::min<NSUInteger>(c.OC, mt / tx));
                [enc dispatchThreads:MTLSizeMake(T, c.OC, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
            }
            barrier();
            return y;
        };
        auto conv_t = [&](const Conv &c, id<MTLBuffer> x, int T, int &T_out) -> id<MTLBuffer> {
            T_out = (T - 1) * c.stride + c.K - 2 * c.pad;
            id<MTLBuffer> y = newbuf((size_t)T_out * c.OC);
            [enc setComputePipelineState:ps(pipe_conv_t_)];
            [enc setBuffer:y offset:0 atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1];
            [enc setBuffer:buf(c.w) offset:0 atIndex:2];
            [enc setBuffer:buf(c.b) offset:0 atIndex:3];
            uint Ti=(uint)T, Ci=(uint)c.IC, To=(uint)T_out, Co=(uint)c.OC, K=(uint)c.K, st=(uint)c.stride, pad=(uint)c.pad;
            [enc setBytes:&Ti length:4 atIndex:4]; [enc setBytes:&Ci length:4 atIndex:5];
            [enc setBytes:&To length:4 atIndex:6]; [enc setBytes:&Co length:4 atIndex:7];
            [enc setBytes:&K length:4 atIndex:8]; [enc setBytes:&st length:4 atIndex:9];
            [enc setBytes:&pad length:4 atIndex:10];
            NSUInteger mt = ps(pipe_conv_t_).maxTotalThreadsPerThreadgroup;
            NSUInteger tx = std::min<NSUInteger>(T_out, mt), ty = std::max<NSUInteger>(1, std::min<NSUInteger>(c.OC, mt / tx));
            [enc dispatchThreads:MTLSizeMake(T_out, c.OC, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
            barrier();
            return y;
        };
        auto aa_act = [&](const Snake &s, id<MTLBuffer> x, int T) -> id<MTLBuffer> {
            id<MTLBuffer> u = newbuf((size_t)(2 * T) * s.C);
            [enc setComputePipelineState:ps(pipe_up_snake_)];
            [enc setBuffer:u offset:0 atIndex:0];
            [enc setBuffer:x offset:0 atIndex:1];
            [enc setBuffer:buf(buf_fir_up_) offset:0 atIndex:2];
            [enc setBuffer:buf(s.a_exp) offset:0 atIndex:3];
            [enc setBuffer:buf(s.b_inv) offset:0 atIndex:4];
            uint Tt=(uint)T, Cc=(uint)s.C, K=(uint)fir_K_;
            [enc setBytes:&Tt length:4 atIndex:5]; [enc setBytes:&Cc length:4 atIndex:6];
            [enc setBytes:&K length:4 atIndex:7];
            NSUInteger mt = ps(pipe_up_snake_).maxTotalThreadsPerThreadgroup;
            NSUInteger nx = 2 * T, tx = std::min<NSUInteger>(nx, mt), ty = std::max<NSUInteger>(1, std::min<NSUInteger>(s.C, mt / tx));
            [enc dispatchThreads:MTLSizeMake(nx, s.C, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
            barrier();
            id<MTLBuffer> y = newbuf((size_t)T * s.C);
            [enc setComputePipelineState:ps(pipe_down_)];
            [enc setBuffer:y offset:0 atIndex:0];
            [enc setBuffer:u offset:0 atIndex:1];
            [enc setBuffer:buf(buf_fir_down_) offset:0 atIndex:2];
            [enc setBytes:&Tt length:4 atIndex:3]; [enc setBytes:&Cc length:4 atIndex:4];
            [enc setBytes:&K length:4 atIndex:5];
            NSUInteger tx2 = std::min<NSUInteger>(T, mt), ty2 = std::max<NSUInteger>(1, std::min<NSUInteger>(s.C, mt / tx2));
            [enc dispatchThreads:MTLSizeMake(T, s.C, 1) threadsPerThreadgroup:MTLSizeMake(tx2, ty2, 1)];
            barrier();
            return y;
        };
        auto add_into = [&](id<MTLBuffer> dst, id<MTLBuffer> src, size_t N) {
            [enc setComputePipelineState:ps(pipe_add_into_)];
            [enc setBuffer:dst offset:0 atIndex:0];
            [enc setBuffer:src offset:0 atIndex:1];
            uint n=(uint)N; [enc setBytes:&n length:4 atIndex:2];
            NSUInteger mt = ps(pipe_add_into_).maxTotalThreadsPerThreadgroup;
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(N, mt), 1, 1)];
            barrier();
        };
        auto scale = [&](id<MTLBuffer> x, float s, size_t N) {
            [enc setComputePipelineState:ps(pipe_axpy3_)];
            [enc setBuffer:x offset:0 atIndex:0];
            [enc setBytes:&s length:4 atIndex:1];
            uint n=(uint)N; [enc setBytes:&n length:4 atIndex:2];
            NSUInteger mt = ps(pipe_axpy3_).maxTotalThreadsPerThreadgroup;
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(N, mt), 1, 1)];
            barrier();
        };
        // AMPBlock1: cur = cur + conv2(act2(conv1(act1(cur)))) over 3 dilations.
        auto amp_block = [&](const ResBlock &rb, id<MTLBuffer> x_in, int T) -> id<MTLBuffer> {
            int C = rb.c1[0].IC;
            id<MTLBuffer> cur = newbuf((size_t)T * C);
            // copy x_in → cur via add_into on a zeroed buffer.
            memset([cur contents], 0, (size_t)T * C * sizeof(float));
            add_into(cur, x_in, (size_t)T * C);
            for (int idx = 0; idx < 3; ++idx) {
                id<MTLBuffer> xt = aa_act(rb.a1[idx], cur, T);
                xt = conv1d(rb.c1[idx], xt, T);
                xt = aa_act(rb.a2[idx], xt, T);
                xt = conv1d(rb.c2[idx], xt, T);
                add_into(cur, xt, (size_t)T * C);
            }
            return cur;
        };

        // ---- Input mel → GPU [T_mel, n_mels] ----
        id<MTLBuffer> x = newbuf((size_t)T_mel * n_mels);
        memcpy([x contents], mel, (size_t)T_mel * n_mels * sizeof(float));
        int T = T_mel;

        // conv_pre
        x = conv1d(conv_pre_, x, T);
        // upsample stages
        for (auto &st : stages_) {
            int T_up = 0;
            x = conv_t(st.ct, x, T, T_up);
            T = T_up;
            const int C = st.ct.OC;
            id<MTLBuffer> xs = amp_block(st.rb[0], x, T);
            id<MTLBuffer> r1 = amp_block(st.rb[1], x, T);
            add_into(xs, r1, (size_t)T * C);
            id<MTLBuffer> r2 = amp_block(st.rb[2], x, T);
            add_into(xs, r2, (size_t)T * C);
            scale(xs, 1.0f / 3.0f, (size_t)T * C);
            x = xs;
        }
        // activation_post + conv_post
        x = aa_act(act_post_, x, T);
        x = conv1d(conv_post_, x, T);   // OC = 1

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { RS_LOG_ERR("BigVGANMetal: GPU error: %s",
                        [[cb.error localizedDescription] UTF8String]); return false; }

        audio_out.assign((size_t)T, 0.0f);
        const float *out = (const float *)[x contents];
        for (int i = 0; i < T; ++i) {
            float v = out[i];
            if (use_tanh_final_) v = std::tanh(v);
            audio_out[i] = v;
        }
        return true;
    }
}

} // namespace indextts2
