// Compatibility shims for building against the iPhO sensevoice-era ggml
// (the only ggml tree proven to compile CUDA under the Jetson Nano gen1's
// nvcc 10.2). Only the FunASR-nano LLM components use these newer ops.
#pragma once
#include "ggml.h"

#ifndef RS_GGML_HAS_SET_ROWS
// ggml_swiglu_split(a, b) == silu(a) * b — exact re-expression.
static inline struct ggml_tensor *ggml_swiglu_split(
        struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
    return ggml_mul(ctx, ggml_silu(ctx, a), b);
}
// ggml_set_rows has no old-ggml equivalent; the FunASR-nano LLM KV cache is
// the sole user and that arch is not deployed on the gen1. Compile-time
// stub; aborts if ever reached at runtime.
static inline struct ggml_tensor *ggml_set_rows(
        struct ggml_context *ctx, struct ggml_tensor *a,
        struct ggml_tensor *b, struct ggml_tensor *c) {
    (void) ctx; (void) b; (void) c;
    GGML_ABORT("ggml_set_rows: not available in this ggml build "
               "(FunASR-nano LLM unsupported on the gen1 toolchain)");
    return a;
}
#endif

// ---- CosyVoice3 (arch not deployed on the gen1) ------------------------
static inline struct ggml_tensor *ggml_softplus(
        struct ggml_context *ctx, struct ggml_tensor *a) {
    (void) ctx;
    GGML_ABORT("ggml_softplus: not available in this ggml build");
    return a;
}
static inline struct ggml_tensor *ggml_fill_inplace(
        struct ggml_context *ctx, struct ggml_tensor *a, float v) {
    (void) ctx; (void) v;
    GGML_ABORT("ggml_fill_inplace: not available in this ggml build");
    return a;
}
static inline struct ggml_tensor *ggml_scale_bias(
        struct ggml_context *ctx, struct ggml_tensor *a, float s, float b) {
    (void) s; (void) b;
    GGML_ABORT("ggml_scale_bias: not available in this ggml build");
    return a;
}
#ifndef GGML_ROPE_TYPE_NORMAL
#define GGML_ROPE_TYPE_NORMAL 0
#endif
static inline struct ggml_tensor *ggml_pad_ext(
        struct ggml_context *ctx, struct ggml_tensor *a,
        int lp0, int rp0, int lp1, int rp1,
        int lp2, int rp2, int lp3, int rp3) {
    (void) lp0; (void) rp0; (void) lp1; (void) rp1;
    (void) lp2; (void) rp2; (void) lp3; (void) rp3;
    GGML_ABORT("ggml_pad_ext: not available in this ggml build");
    return a;
}
static inline struct ggml_tensor *ggml_interpolate(
        struct ggml_context *ctx, struct ggml_tensor *a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode) {
    (void) ctx; (void) ne0; (void) ne1; (void) ne2; (void) ne3; (void) mode;
    GGML_ABORT("ggml_interpolate: not available in this ggml build");
    return a;
}
