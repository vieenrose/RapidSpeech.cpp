#!/bin/bash
# Native CUDA build of RapidSpeech.cpp for the NVIDIA Jetson Nano gen1
# (JetPack 4.6.1 / L4T r32.7, Ubuntu 18.04, CUDA 10.2, sm_53 Maxwell).
#
# Run this ON the device, or inside an aarch64 L4T r32.7 container.
# It fetches the validated ggml tree (b2a092a7), applies the CUDA-10.2 / sm_53
# source patch, sets up the toolchain quirks CUDA 10.2 needs, and builds.
#
# Validation status: built & functionally verified on a GB10 (sm_121) via the
# dustynv/l4t-pytorch:r32.7.1 container with the device compute-capability spoofed
# to 530 (RS_FORCE_CC=530) so the whole ggml dispatch takes the exact sm_53 path.
# SenseVoice produced the correct transcription and melo8k produced audio bit-equal
# to the CPU backend (corr 0.999999). Absolute GPU speed must still be measured on
# real Nano hardware (cublas-10.2 JIT + Blackwell != Maxwell make GB10 timings
# meaningless). See docs/CUDA_10.2_JETSON_NANO.md.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUDA=${CUDA:-/usr/local/cuda-10.2}
export PATH="$CUDA/bin:$PATH"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ---------------------------------------------------------------------------
# 1) gcc-8: JetPack 4.6.1 ships gcc-7.5, which lacks <charconv> (used by
#    ggml-cuda.cu). gcc-8 is the minimum that compiles modern-ggml CUDA.
# ---------------------------------------------------------------------------
command -v g++-8 >/dev/null || { $SUDO apt-get update; $SUDO apt-get install -y --no-install-recommends g++-8 gcc-8; }

# ---------------------------------------------------------------------------
# 2) Fetch the pinned ggml (b2a092a7) and apply the CUDA-10.2 / sm_53 patch.
#    The patch (idempotent via reverse-check) carries:
#      - pre-Volta NVIDIA batched-cuBLAS GEMM -> FP32 compute + non-tensor algo
#        (sm_53 Maxwell has NO tensor cores; the tensor-op path returns
#         CUBLAS_STATUS_NOT_SUPPORTED). THIS is the functional bug fix.
#      - gcc<9 aarch64 NEON _x2/_x4 load fallbacks
#      - a no-op __builtin_assume for compilers that lack it
#      - RS_FORCE_CC / RS_GEMM_NO_TENSOR env hooks (test-only; harmless on Nano)
# ---------------------------------------------------------------------------
# `protocol.file.allow=always` lets the submodule fetch use file:// transport,
# which git >= 2.38.1 blocks by default (CVE-2022-39253) — this bites inside
# containers / against a local object cache. If the pinned commit is already
# present (offline), fall back to checking it out directly.
git -C "$ROOT" -c protocol.file.allow=always submodule update --init ggml \
  || git -C "$ROOT/ggml" checkout -q "$(git -C "$ROOT" ls-tree HEAD ggml | awk '{print $3}')"
if ! git -C "$ROOT/ggml" apply --reverse --check "$ROOT/patches/ggml-cuda-10.2-sm53.patch" 2>/dev/null; then
  git -C "$ROOT/ggml" apply "$ROOT/patches/ggml-cuda-10.2-sm53.patch"
fi

# ---------------------------------------------------------------------------
# 3) CUDA 10.2 has no cuda_bf16.h. Provide a minimal stub (bf16 -> fp16; the
#    bf16 paths are arch-gated out on sm_53, so the mapping is never executed).
# ---------------------------------------------------------------------------
INC="$CUDA/targets/aarch64-linux/include"
if [ ! -f "$INC/cuda_bf16.h" ]; then
  $SUDO tee "$INC/cuda_bf16.h" >/dev/null <<'BF'
#pragma once
#include <cuda_fp16.h>
typedef __half  nv_bfloat16;  typedef __half  __nv_bfloat16;
typedef __half2 nv_bfloat162; typedef __half2 __nv_bfloat162;
BF
  $SUDO cp "$INC/cuda_bf16.h" "$INC/cuda_bf16.hpp"
fi

# ---------------------------------------------------------------------------
# 4) Configure + build. The -D back-defines map CUDA-11+ cuBLAS enums onto the
#    CUDA-10.2 equivalents (CUDA_R_16BF -> CUDA_R_16F, CUBLAS_COMPUTE_* ->
#    cudaDataType). C++14 device standard (nvcc 10.2 caps device code at C++14).
# ---------------------------------------------------------------------------
cd "$ROOT" && rm -rf build-nano && mkdir build-nano && cd build-nano
cmake .. -DRS_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=ON -DCMAKE_CUDA_ARCHITECTURES=53 \
  -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF \
  -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_CUDA_FLAGS="--forward-unknown-to-host-compiler -arch=sm_53 \
     -DCUDA_R_16BF=CUDA_R_16F -DCUBLAS_COMPUTE_16F=CUDA_R_16F -DCUBLAS_COMPUTE_32F=CUDA_R_32F \
     -DCUBLAS_COMPUTE_32F_FAST_16F=CUDA_R_32F -DCUBLAS_COMPUTE_32F_FAST_TF32=CUDA_R_32F \
     -DCUBLAS_COMPUTE_16BF=CUDA_R_16F -DCUBLAS_TF32_TENSOR_OP_MATH=CUBLAS_TENSOR_OP_MATH"
make -j"$(nproc)" rs-asr-offline rs-tts-offline rs-asr-vad-online
echo "NANO-CUDA-BUILD-OK"
