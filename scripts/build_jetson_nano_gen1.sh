#!/bin/bash
set -e
ROOTDIR=/workspace
TC=$ROOTDIR/toolchain/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu
CROSS=$TC/bin/aarch64-linux-gnu-
SYSROOT=$TC/aarch64-linux-gnu/libc
CUDA=/usr/local/cuda-10.2
CT=$CUDA/targets/aarch64-linux
cd /rs && rm -rf build-nano && mkdir build-nano && cd build-nano
cmake .. \
  -DRS_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=${CROSS}gcc -DCMAKE_CXX_COMPILER=${CROSS}g++ \
  -DCMAKE_AR=${CROSS}ar -DCMAKE_RANLIB=${CROSS}ranlib \
  -DCMAKE_SYSROOT=$SYSROOT \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
  -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
  -DCMAKE_CUDA_HOST_COMPILER=${CROSS}g++ \
  -DCMAKE_CUDA_STANDARD=14 \
  -DCMAKE_CUDA_ARCHITECTURES=53 \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_GENERIC=OFF -DGGML_CPU_ARM=OFF \
  -DCMAKE_CUDA_FLAGS="--forward-unknown-to-host-compiler -arch=sm_53 -Xcompiler -fPIC" \
  -DCMAKE_C_FLAGS="-I$CUDA/include -I$CT/include" \
  -DCMAKE_CXX_FLAGS="-I$CUDA/include -I$CT/include" \
  -DCMAKE_PREFIX_PATH="$CT" \
  -DCMAKE_FIND_ROOT_PATH="$SYSROOT;$CT" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DOpenMP_C_FLAGS="-fopenmp" -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_C_LIB_NAMES="gomp" -DOpenMP_CXX_LIB_NAMES="gomp" \
  -DOpenMP_gomp_LIBRARY="$SYSROOT/usr/lib64/libgomp.so" \
  -DCUDAToolkit_INCLUDE_DIRECTORIES=$CT/include \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--rpath-link=$CT/lib/stubs -Wl,--rpath-link=$CT/lib -L$CT/lib/stubs -L$CT/lib -lcudart -lcublas -lcufft -lcublasLt -lcuda -lpthread -ldl -lrt" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--rpath-link=$CT/lib/stubs -Wl,--rpath-link=$CT/lib -L$CT/lib -L$CT/lib/stubs -lcudart -lcublas -lcufft -lcublasLt -lcuda -lpthread -ldl -lrt"
make -j6
echo NANO-CUDA-BUILD-OK
