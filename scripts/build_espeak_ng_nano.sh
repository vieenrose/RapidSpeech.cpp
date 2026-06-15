#!/bin/bash
# Cross-build espeak-ng 1.52 for the Jetson Nano gen1 (aarch64, glibc 2.27) and assemble an
# ESPEAK_NG_ROOT for the Matcha English text frontend (-DRS_MATCHA_ESPEAK=ON).
#
# WHY 1.52 specifically: the matcha-icefall-zh-en model was built with this exact espeak-ng
# (csukuangfj fork, commit f6fed6c — the same FetchContent sherpa-onnx uses). 1.51 produces
# different English STRESS marks and breaks byte-identity with sherpa's frontend.
#
# Run inside the ipho-sdk-ubuntu18.04 container (has aarch64-linux-gnu-gcc 7.5 = exact Nano
# glibc 2.27 match, + cmake). Output: $OUT/ with lib/ include/ espeak-ng-data/.
#   docker run --rm -v "$PWD":/rs ipho-sdk-ubuntu18.04 bash /rs/scripts/build_espeak_ng_nano.sh
set -e
SRC_ZIP_URL="https://github.com/csukuangfj/espeak-ng/archive/f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip"
WORK=${WORK:-/tmp/espeak_ng_build}
OUT=${OUT:-/rs/prebuilt/espeak-ng-nano}
CMARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF \
  -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_KLATT=OFF -DUSE_SPEECHPLAYER=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"
curl -fsSL -o e.zip "$SRC_ZIP_URL" && unzip -q e.zip && cd espeak-ng-*

# 1) host build — only to generate the (architecture-independent) espeak-ng-data dictionaries.
cmake -B build-host $CMARGS >/dev/null
cmake --build build-host -j"$(nproc)" --target espeak-ng-data >/dev/null 2>&1 || cmake --build build-host -j"$(nproc)" >/dev/null

# 2) aarch64 cross build — just the shared library (running the data tool on x86 would fail).
cat > aarch64.cmake <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
cmake -B build-nano -DCMAKE_TOOLCHAIN_FILE="$PWD/aarch64.cmake" $CMARGS >/dev/null
cmake --build build-nano -j"$(nproc)" --target espeak-ng >/dev/null

# 3) assemble ESPEAK_NG_ROOT (aarch64 lib + headers + arch-independent data).
rm -rf "$OUT" && mkdir -p "$OUT/lib" "$OUT/include"
cp -P build-nano/src/libespeak-ng/libespeak-ng.so* "$OUT/lib/"
cp -P build-nano/src/ucd-tools/libucd.so*        "$OUT/lib/"   # espeak-ng's Unicode DB (DT_NEEDED)
cp -r src/include/espeak-ng "$OUT/include/"
cp -r build-host/espeak-ng-data "$OUT/espeak-ng-data"

echo "espeak-ng 1.52 (aarch64) -> $OUT"
aarch64-linux-gnu-objdump -T "$OUT/lib/libespeak-ng.so" | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1
# Build RapidSpeech for the Nano with:
#   -DRS_MATCHA_ESPEAK=ON -DESPEAK_NG_ROOT=$OUT
# and at runtime set ESPEAK_DATA_PATH=<deployed>/espeak-ng-data
