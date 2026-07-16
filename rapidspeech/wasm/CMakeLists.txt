cmake_minimum_required(VERSION 3.14)
project(rapidspeech-wasm VERSION 0.1.0 LANGUAGES C CXX)

# ── Emscripten-specific settings ─────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable all GPU backends for WASM
set(RS_CUDA OFF CACHE BOOL "" FORCE)
set(RS_METAL OFF CACHE BOOL "" FORCE)
set(RS_VULKAN OFF CACHE BOOL "" FORCE)
set(RS_CANN OFF CACHE BOOL "" FORCE)
set(RS_OPENCL OFF CACHE BOOL "" FORCE)
set(RS_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(RS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)

# Opt-in WebGPU backend for the browser. Off by default because building the
# emdawnwebgpu port adds ~minutes to the WASM build and only Chromium-based
# browsers ship WebGPU enabled at the moment.
option(RS_WASM_WEBGPU "Enable WebGPU backend in the WASM build" OFF)
if (RS_WASM_WEBGPU)
    set(GGML_WEBGPU ON  CACHE BOOL "" FORCE)
    # Our WASM link line uses -sASYNCIFY=1. emdawnwebgpu's JSPI mode is
    # mutually exclusive with ASYNCIFY, so force the asyncify path.
    set(GGML_WEBGPU_JSPI OFF CACHE BOOL "" FORCE)
endif()

# Enable pthreads in the WASM build. Required because rapidspeech (and ggml)
# spawn std::thread for parallel fbank / mat-mul; without -pthread the
# std::thread constructor throws system_error → abort.
#
# Caveats when enabling:
#   1. Pages must be served cross-origin-isolated (COOP: same-origin,
#      COEP: require-corp) so SharedArrayBuffer is available.
#   2. Combining ASYNCIFY with pthreads is allowed but async ops must stay on
#      the main thread (which is the case here — model loading is on main).
#   3. ALLOW_MEMORY_GROWTH + pthreads requires MAXIMUM_MEMORY to be set.
option(RS_WASM_PTHREADS "Enable pthreads in the WASM build" OFF)
if (RS_WASM_PTHREADS)
    # Propagate -pthread to ggml (and everything else built in this tree).
    # Must be set before add_subdirectory(ggml) so ggml's targets pick it up.
    add_compile_options(-pthread)
    add_link_options(-pthread)
else()
    # Single-threaded build: ggml's `find_package(Threads REQUIRED)` +
    # `Threads::Threads` would otherwise add -pthread on Emscripten, pulling the
    # multi-threaded libc (libc-mt) and clashing with the single-threaded stub
    # (duplicate emscripten_futex_* / __lock / __wait symbols at link time).
    # Pre-create an EMPTY Threads::Threads so ggml's find_package sees the
    # requirement already satisfied and emits no -pthread flag anywhere.
    if (NOT TARGET Threads::Threads)
        add_library(Threads::Threads INTERFACE IMPORTED GLOBAL)
    endif()
    set(Threads_FOUND TRUE CACHE INTERNAL "")
    set(CMAKE_THREAD_LIBS_INIT "" CACHE INTERNAL "")
    set(CMAKE_HAVE_THREADS_LIBRARY 1 CACHE INTERNAL "")
    set(CMAKE_USE_PTHREADS_INIT 0 CACHE INTERNAL "")
    set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "" FORCE)
endif()

set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml")

# Match the root project: some models (e.g. OmniVoice TTS) have tensor names
# longer than ggml's default GGML_MAX_NAME=64. The standalone WASM project
# does not inherit add_compile_definitions() from the root CMakeLists, so
# we have to repeat it here, BEFORE add_subdirectory(ggml).
add_compile_definitions(GGML_MAX_NAME=128)

if(NOT EXISTS "${GGML_PATH}/CMakeLists.txt")
    message(FATAL_ERROR "ggml not found at ${GGML_PATH}")
endif()

# Performance flags for the WHOLE tree (ggml + rapidspeech), set BEFORE
# add_subdirectory(ggml) so ggml's targets inherit them.
#   -msimd128       : 128-bit WASM SIMD (ggml's vectorized quant/mul_mat kernels).
#   -O3 at LINK     : REQUIRED for emscripten to run the Binaryen wasm-opt pass;
#                     CMAKE_BUILD_TYPE=Release only sets -O3 for COMPILE, so
#                     without this the final .wasm is left largely unoptimized.
# NOTE: do NOT add -mrelaxed-simd here. It measured ZERO speedup (clang only
# auto-emits a few relaxed FMAs; ggml's hot loops are hand-written intrinsics)
# but makes the module UNPARSEABLE on engines without relaxed-SIMD — older iOS
# Safari aborted with "CompileError: relaxed simd instructions not supported".
# If relaxed-dot kernels are ever hand-written, gate them to a separate
# desktop-only build variant.
add_compile_options(-msimd128)
add_link_options(-O3 -msimd128)
# Relaxed SIMD (desktop-Chrome-only variant): maps relaxed_dot/madd to real
# FMA/dot hardware — big win in the quant vec_dot kernels. NOT for the default
# builds: auto-emitted relaxed opcodes CompileError on older iOS Safari, so
# the page feature-detects and only loads this variant where supported.
option(RS_WASM_RELAXED_SIMD "Enable WASM relaxed-simd (desktop-only variant)" OFF)
if (RS_WASM_RELAXED_SIMD)
    add_compile_options(-mrelaxed-simd)
    add_link_options(-mrelaxed-simd)
endif()

# Pass Emscripten flags to ggml
set(GGML_WASM_SINGLE_FILE ON CACHE BOOL "ggml: single-file WASM" FORCE)

add_subdirectory(${GGML_PATH} ggml)

# ── RapidSpeech core sources (ASR + TTS) ─────────────────────
set(RS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/rs_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/rs_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/rs_kws.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/audio_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/fftsg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/text_frontend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/whisper_mel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/rs_log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/rs_wav.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/sensevoice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/sensevoice_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/context_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/keyword_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/kws_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/funasr-nano.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/silero_vad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/fireredvad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/llm_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/llm_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/llm_kv_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/qwen3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/qwen3_asr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/moss_td.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/xasr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/xasr_zipformer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/omnivoice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/openvoice2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/bert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/campplus.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api/rapidspeech_c.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api/rs_vad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api/rs_speaker.cpp
)

# ── Static library ───────────────────────────────────────────
add_library(rapidspeech-core-wasm STATIC ${RS_SOURCES})

target_include_directories(rapidspeech-core-wasm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api
    ${GGML_PATH}/include
    ${GGML_PATH}/src
)

target_compile_definitions(rapidspeech-core-wasm PRIVATE RAPIDSPEECH_BUILD)
target_link_libraries(rapidspeech-core-wasm PRIVATE ggml)

# ── WASM executable ──────────────────────────────────────────
add_executable(rapidspeech-wasm
    rapidspeech_wasm.cpp
)

target_include_directories(rapidspeech-wasm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${GGML_PATH}/include
)

target_link_libraries(rapidspeech-wasm PRIVATE rapidspeech-core-wasm ggml)

# Pull WebGPU backend into the link when enabled, and surface its INTERFACE
# link options (--use-port=emdawnwebgpu, -sASYNCIFY/-sJSPI, etc.) onto the
# final executable.
if (RS_WASM_WEBGPU AND TARGET ggml-webgpu)
    target_link_libraries(rapidspeech-core-wasm PUBLIC ggml-webgpu)
    target_compile_definitions(rapidspeech-core-wasm PUBLIC RS_USE_WEBGPU)
endif()

# ── Emscripten link flags ────────────────────────────────────
set_target_properties(rapidspeech-wasm PROPERTIES
    SUFFIX ".js"
)

target_link_options(rapidspeech-wasm PRIVATE
    # Memory
    -sINITIAL_MEMORY=256MB
    -sALLOW_MEMORY_GROWTH=1
    -sSTACK_SIZE=10MB
    # Enable WASM SIMD for better performance
    -msimd128

    # Filesystem support (for reading .gguf model). -lnodefs.js exposes NODEFS
    # so Node harnesses can stream the model from disk. -lworkerfs.js exposes
    # WORKERFS so the browser worker can mount the downloaded gguf as a Blob —
    # ggml then reads it lazily WITHOUT a ~700 MB in-heap MEMFS copy (critical
    # for memory-capped iOS).
    -sFORCE_FILESYSTEM=1
    -lnodefs.js
    -lworkerfs.js

    # Export the WASM entry-point functions to JS
    -sEXPORTED_FUNCTIONS=[_rs_wasm_init,_rs_wasm_init_ex,_rs_wasm_free,_rs_wasm_push_audio,_rs_wasm_push_text,_rs_wasm_push_reference_audio,_rs_wasm_push_reference_text,_rs_wasm_process,_rs_wasm_redecode,_rs_wasm_get_text,_rs_wasm_get_audio_ptr,_rs_wasm_get_audio_len,_rs_wasm_reset,_rs_wasm_get_sample_rate,_rs_wasm_get_arch_name,_rs_wasm_is_ready,_rs_wasm_get_version,_rs_wasm_set_user_input_prompt,_rs_wasm_moss_transcribe_mel,_rs_wasm_moss_transcribe_pcm,_rs_wasm_set_use_llm,_rs_wasm_set_ctc_precheck,_rs_wasm_asr_stream_supported,_rs_wasm_asr_stream_set_chunk_len,_rs_wasm_asr_stream_push,_rs_wasm_asr_stream_finish,_rs_wasm_asr_stream_reset,_rs_wasm_set_tts_params,_rs_wasm_set_tts_diffusion_steps,_rs_wasm_vad_init,_rs_wasm_vad_free,_rs_wasm_vad_reset,_rs_wasm_vad_set_threshold,_rs_wasm_vad_push_audio,_rs_wasm_vad_is_speech,_rs_wasm_vad_get_probability,_rs_wasm_vad_get_arch,_rs_wasm_vad_drain_segments,_rs_wasm_vad_drain_frames,_rs_wasm_kws_init,_rs_wasm_kws_free,_rs_wasm_kws_reset,_rs_wasm_kws_push_audio,_rs_wasm_kws_poll,_rs_wasm_kws_get_hits_ptr,_rs_wasm_kws_get_hits_count,_rs_wasm_kws_get_phrase_pool_ptr,_rs_wasm_kws_get_phrase_pool_len,_rs_wasm_kws_get_sample_rate,_rs_wasm_kws_get_arch_name,_rs_wasm_setenv,_rs_wasm_speaker_init,_rs_wasm_speaker_embed,_rs_wasm_speaker_dim,_rs_wasm_speaker_free,_malloc,_free]
    -sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,stringToUTF8,lengthBytesUTF8,getValue,setValue,FS,HEAPF32,HEAP32,HEAPU8,HEAPU32]

    # No closure compiler (easier debugging)
    --closure 0

    # Modularize for clean global scope
    -sMODULARIZE=1
    -sEXPORT_NAME=RapidSpeechModule

    # Async model loading
    -sASYNCIFY=1
)

# Maximum WASM memory. With pthreads the memory is SHARED, and shared
# WebAssembly.Memory cannot be re-reserved, so browsers reserve the MAXIMUM up
# front at instantiation. iOS WebKit refuses large reservations (~>1.5 GB), so
# an iOS-targeted build must use a small maximum (e.g. 1536MB) — see the
# rapidspeech-wasm-ios variant built by the demo.
set(RS_WASM_MAX_MEMORY "4GB" CACHE STRING "WASM -sMAXIMUM_MEMORY (shared memory is reserved up front)")

if (RS_WASM_PTHREADS)
    # ALLOW_MEMORY_GROWTH + pthreads requires MAXIMUM_MEMORY to be set so the
    # SharedArrayBuffer can be reallocated safely across workers.
    target_link_options(rapidspeech-wasm PRIVATE
        -pthread
        -sUSE_PTHREADS=1
        # Size the worker pool to the machine's core count (was fixed at 4) so
        # the encoder FFN / prefill / decode matmuls can use every core.
        "SHELL:-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency"
        -sMAXIMUM_MEMORY=${RS_WASM_MAX_MEMORY}
    )
endif()

# If building with pthreads support, add:
# target_link_options(rapidspeech-wasm PRIVATE -pthread -sUSE_PTHREADS=1)
