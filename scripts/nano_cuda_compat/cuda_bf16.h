/* Fake <cuda_bf16.h> for CUDA 10.2 (bf16 arrived in CUDA 11). Maps nv_bfloat16
   to __half so modern ggml-cuda compiles on the Jetson Nano gen1 (sm_53), where
   all real bf16 paths are behind __CUDA_ARCH__>=800 guards the preprocessor
   strips. kreier-style stub. */
#ifndef GGML_NANO_CUDA_BF16_STUB_H
#define GGML_NANO_CUDA_BF16_STUB_H
#include <cuda_fp16.h>

typedef __half  __nv_bfloat16;
typedef __half2 __nv_bfloat162;
typedef __half  nv_bfloat16;
typedef __half2 nv_bfloat162;

#if defined(__CUDACC__)
__device__ __forceinline__ float        __bfloat162float(__nv_bfloat16 x) { return __half2float(x); }
__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float x)        { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat162 __float2bfloat162_rn(float x)   { return __float2half2_rn(x); }
__device__ __forceinline__ __nv_bfloat162 make_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return __halves2half2(a, b); }
#endif

#endif
