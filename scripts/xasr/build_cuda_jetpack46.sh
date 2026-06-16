#!/bin/bash
# Configure + build RapidSpeech.cpp with the CUDA-10.2 / sm_53 backend, intended
# to run inside the JetPack-4.6 image (scripts/xasr/Dockerfile.jetpack46-cuda).
# Mirrors scripts/build_jetson_nano_gen1_native.sh but standalone/idempotent.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA=${CUDA:-/usr/local/cuda-10.2}
export PATH="$CUDA/bin:$PATH"

# Pinned ggml + CUDA-10.2 / sm_53 patch (idempotent).
git config --global --add safe.directory "$ROOT/ggml" 2>/dev/null || true
git -C "$ROOT" -c protocol.file.allow=always submodule update --init ggml 2>/dev/null || true
git -C "$ROOT/ggml" apply --reverse --check "$ROOT/patches/ggml-cuda-10.2-sm53.patch" 2>/dev/null \
  || git -C "$ROOT/ggml" apply "$ROOT/patches/ggml-cuda-10.2-sm53.patch"

cd "$ROOT" && rm -rf build-nano && mkdir build-nano
cmake -S . -B build-nano -DRS_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=ON -DCMAKE_CUDA_ARCHITECTURES=53 \
  -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF \
  -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DCMAKE_CUDA_HOST_COMPILER=g++-8 \
  -DCUDAToolkit_ROOT="$CUDA" \
  -DCMAKE_CUDA_FLAGS="--forward-unknown-to-host-compiler -arch=sm_53 \
     -DCUDA_R_16BF=CUDA_R_16F -DCUBLAS_COMPUTE_16F=CUDA_R_16F -DCUBLAS_COMPUTE_32F=CUDA_R_32F \
     -DCUBLAS_COMPUTE_32F_FAST_16F=CUDA_R_32F -DCUBLAS_COMPUTE_32F_FAST_TF32=CUDA_R_32F \
     -DCUBLAS_COMPUTE_16BF=CUDA_R_16F -DCUBLAS_TF32_TENSOR_OP_MATH=CUBLAS_TENSOR_OP_MATH"
cmake --build build-nano --target rs-asr-offline -j"$(nproc)"
echo "XASR-CUDA-BUILD-OK -> build-nano/rs-asr-offline"
