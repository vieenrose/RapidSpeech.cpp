/* Force-included into nvcc compiles for CUDA 10.2 on Jetson Nano gen1 (kreier-style).
   - __builtin_assume: not in gcc-8 aarch64 (clang/gcc-9+); make it a no-op.
   - CUBLAS_COMPUTE_* + math-mode enums: added in CUDA 11; back-define for <11000. */
#ifndef GGML_NANO_NVCC_COMPAT_H
#define GGML_NANO_NVCC_COMPAT_H

#define __builtin_assume(x) ((void)0)

/* bf16 cudaDataType enums (CUDA 11+); map to fp16 equivalents for CUDA 10.2. */
#include <library_types.h>
#ifndef CUDA_R_16BF
#define CUDA_R_16BF CUDA_R_16F
#define CUDA_C_16BF CUDA_C_16F
#endif

#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
#include <cublas_v2.h>
#ifndef CUBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F CUDA_R_32F
#define CUBLAS_COMPUTE_64F CUDA_R_64F
#endif
#ifndef CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#endif
#endif

#endif
