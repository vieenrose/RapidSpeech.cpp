# CUDA 10.2 / Jetson Nano gen1 (sm_53) support — status & how it was validated

This branch targets the **NVIDIA Jetson Nano gen1** (JetPack 4.6.1 / L4T r32.7,
Ubuntu 18.04, **CUDA 10.2**, Tegra X1, **sm_53 Maxwell**). It carries the changes
needed to compile *and run* RapidSpeech.cpp's CUDA backend on that toolchain.

## TL;DR

| | Status |
|---|---|
| Compiles under nvcc 10.2 / gcc-8 / C++14 / sm_53 | ✅ validated |
| SenseVoice / silero-VAD / melo8k run on GPU without crashing | ✅ validated (sm_53 dispatch) |
| GPU output correctness | ✅ SenseVoice transcription correct; melo8k GPU==CPU (corr 0.999999) |
| GPU **speed** vs CPU on the actual Nano | ⏳ **pending real hardware** (see "Why speed isn't here") |

Build it with [`scripts/build_jetson_nano_gen1_native.sh`](../scripts/build_jetson_nano_gen1_native.sh)
(native, run on the device or an aarch64 L4T r32.7 container).

## The fix that matters: pre-Volta cuBLAS GEMM

ggml's **batched** cuBLAS matmul (`ggml_cuda_mul_mat_batched_cublas`)
unconditionally requested `CUBLAS_COMPUTE_16F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
for every non-AMD GPU. **Tensor-op GEMM requires tensor cores (sm_70+).** The
Jetson Nano gen1 is **sm_53 Maxwell — no tensor cores**, so that call returns
`CUBLAS_STATUS_NOT_SUPPORTED` and the process aborts on the first ASR/TTS matmul.

(The *non*-batched path was already guarded `cc >= VOLTA` and fell back to
`cublasSgemm`; only the batched path was missing the guard.)

The patch adds a pre-Volta NVIDIA branch → **FP32 compute + `CUBLAS_GEMM_DEFAULT`**
(non-tensor), mirroring the existing AMD/CDNA fallback. On a real sm_53 device this
path is taken automatically (`cc < 700`). MMQ (quantized matmul) is also auto-skipped
on sm_53 because it needs DP4A (sm_61+); ggml dequantizes and uses cuBLAS instead.

## The full CUDA-10.2 build patch set

Carried by `patches/ggml-cuda-10.2-sm53.patch` + the build script:

| Patch | Why |
|---|---|
| pre-Volta batched-cuBLAS → FP32/non-tensor | sm_53 has no tensor cores (the bug above) |
| NEON `_x2`/`_x4` load inlines for gcc<9 aarch64 | gcc-8's `arm_neon.h` lacks the `vld1q_*_x4` intrinsics |
| no-op `__builtin_assume` | gcc lacks the builtin |
| **gcc-8** host compiler | gcc-7.5 (JetPack default) lacks `<charconv>` |
| `cuda_bf16.h` stub | CUDA 10.2 has no bf16 header (bf16 unused on sm_53) |
| `CUDA_R_16BF` / `CUBLAS_COMPUTE_*` `-D` back-defines | those enums are CUDA-11+ |
| `CMAKE_CUDA_STANDARD=14` | nvcc 10.2 caps device code at C++14 |

ggml is pinned to **b2a092a7** (the tree the patch was generated against).

## How it was validated without a Nano

The crash reproduces on a **GB10 (sm_121)** too, because cuBLAS 10.2 doesn't
recognise that arch either. To exercise the *exact* sm_53 code paths, the patch
adds two **test-only, env-gated** hooks (no effect unless set; never set on a real
Nano):

- `RS_FORCE_CC=530` — clamps the detected compute capability so ggml's whole
  dispatch (MMQ vs cuBLAS, tensor vs non-tensor) behaves as on real sm_53.
- `RS_GEMM_NO_TENSOR=1` — forces the non-tensor GEMM path directly.

Under `RS_FORCE_CC=530` in the `dustynv/l4t-pytorch:r32.7.1` container (genuine
nvcc 10.2 + real CUDA-10.2 cuBLAS), both models ran to completion with correct
output. Reproduce with `scripts/build_jetson_nano_gen1_native.sh` then run
`rs-asr-offline` / `rs-tts-offline` with `--gpu true RS_FORCE_CC=530`.

## Memory footprint (peak RSS)

Re-built and re-run on a GB10 in an aarch64 **CUDA-10.2** container (genuine nvcc 10.2
toolchain, `RS_FORCE_CC=530`), measuring peak `VmHWM`:

| Model (CUDA, gen1) | peak RSS |
|---|---|
| SenseVoice + silero-VAD | **756 MB** |
| melo8k TTS | **572 MB** |

RSS still carries the GB10's larger CUDA-13 driver context, so on a real 4 GB Nano
(small CUDA-10.2 context) the absolute numbers are lower — but the figure is useful
relatively: the ggml backend keeps the conv-heavy ASR model well under 1 GB, where a
cuDNN-based stack alone faults in ~782 MB before weights. (The cuDNN-free onnxruntime
1.11 build of the same models needs ~1.2 GB / ~720 MB respectively — see the
edge-speech-gpu-bench comparison.)

## Why speed isn't here

The GB10 has no native sm_53 cuBLAS SASS, so every kernel **JIT-compiles to sm_121
on first use** — the one SenseVoice encode measured ~59 s, essentially all JIT. And
even warm, it is Blackwell silicon, not Maxwell, so no GB10 number predicts Nano RTF,
nor even whether GPU beats the A57 CPU on these small models. Absolute speed — and
the "is CUDA actually worth it vs CPU on the Nano" question — will be measured on the
real device.
