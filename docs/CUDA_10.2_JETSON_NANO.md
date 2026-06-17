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
| GPU **speed** vs CPU on the actual Nano | ✅ **measured** — X-ASR encoder, `RS_GEMM_FP16` 1.75× & token-exact (see "On-device speed") |

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
- `RS_GEMM_FP16=1` — **opt-in FP16-compute GEMM** for default-precision matmuls
  on pre-Volta NVIDIA GPUs. sm_53 (GM20B) has native 2x FP16 throughput, but the
  stock path routes pre-Volta to FP32; this keeps the bulk encoder GEMMs in FP16
  (`CUBLAS_COMPUTE_16F`, non-tensor algo) for ~2x compute and half the memory
  traffic — the dominant cost on the Nano's ~25.6 GB/s memory. Numerically
  sensitive matmuls (attention scores, joiner logits) stay FP32 via
  `GGML_PREC_F32`, so output is token-exact with the FP32 path (verified on the
  X-ASR encoder, all chunk variants). On a real Nano set `RS_GEMM_FP16=1` alone;
  on newer hardware pair it with `RS_GEMM_NO_TENSOR=1` to exercise the same path.
  This is a *latency* lever specific to the Nano (it gave nothing on the
  bandwidth-rich GB10 dev host). **Measured on a real Nano: 1.75× faster encoder
  and token-exact** — see "On-device speed" below.

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

## On-device speed (real Jetson Nano gen1, sm_53)

Measured on a **real Nano** (L4T R32.5.1, CUDA 10.2, MAXN, native sm_53 build, no
JIT) with the X-ASR zh-en **960 ms** streaming encoder, full reference clip (1105
frames / 11 chunks), steady-state, warm-up excluded. **Encoder ms/chunk** is the
figure of merit:

| Config | Encoder ms/chunk | Tokens |
|--------|-----------------:|:------:|
| CPU f16 | 637.1 | exact |
| CPU q3_k-im | 535.0 | exact |
| GPU f16, **default (FP32 GEMM)** | 703.6 | exact |
| GPU f16, **`RS_GEMM_FP16=1`** | **401.5** | **exact** |
| GPU q3_k-im, FP32 GEMM | 415.2 | exact |
| GPU q3_k-im, `RS_GEMM_FP16=1` | 444.7 | exact |

All six outputs are **byte-exact** with the CPU f16 reference and decode the correct
transcript; all run faster than real time (960 ms audio/chunk).

- **`RS_GEMM_FP16` gives a 1.75× encoder speedup (703.6 → 401.5 ms/chunk),
  token-exact** — the FP16-throughput roofline prediction, confirmed on Maxwell.
- **Without the lever, GPU is *not* worth it for f16:** stock GPU FP32 (703.6) is
  *slower* than the A57 CPU (637.1). The FP16 lever — or quantization — is what
  puts GPU ahead.
- **Quantization is the other bandwidth lever:** q3_k-im FP32 (415.2) ≈ f16 +
  FP16-lever (401.5). They are **substitutes, not additive** (q3_k-im + lever is
  444.7, slightly worse — 3-bit weights already minimise traffic). ~400 ms/chunk
  is the Nano bandwidth floor for this encoder.

### vs sherpa-onnx, and CPU-thread / CUDA-core scaling

A four-engine sweep (sherpa-onnx CPU, sherpa-onnx CUDA, RapidSpeech CPU,
RapidSpeech CUDA) over every non-broken quantization, plus a CPU-thread (1→4) and
CUDA-core (1→4) sweep. Encoder ms/chunk, 4 threads/cores unless noted:

| Engine / weights | CPU (4 thr) | CUDA (best) | Correct? |
|------------------|------------:|------------:|:--------:|
| sherpa-onnx fp32 | 396.9 | — *not runnable* | ✓ |
| sherpa-onnx int8 | **329.8** | — *not runnable* | ✓ |
| RapidSpeech f16 | 639.1 | **386.5** (`RS_GEMM_FP16`) | ✓ |
| RapidSpeech q8_0 | 459.2 | **355.3** (FP32) | ✓ |
| RapidSpeech q3_k-im | 498.5 | 445.4 | ✓ |
| RapidSpeech iq4_xs | 452.2 | 451.8 | ✓ |

- **sherpa-onnx CUDA cannot run on this Nano:** the onnxruntime build is CPU-only
  (no aarch64 onnxruntime-gpu wheel for this JetPack; the cuDNN/cuBLAS context OOMs
  4 GB anyway). RapidSpeech.cpp's ggml-CUDA backend is the only way to put this
  model on the Nano GPU — the motivation for this whole branch.
- **sherpa-onnx CPU is the fastest engine** (MLAS tuned-ARM GEMM + op fusion +
  int8) and **saturates at 3 threads**; RapidSpeech CPU scales ~linearly to 4
  threads (~3.4× from 1→4) but starts behind (unfused graph, per-op f16→f32
  dequant). RapidSpeech CUDA + FP16 lever reaches sherpa-CPU parity *on the GPU*.
- **CUDA latency is essentially independent of CPU core count** (flat 4→2 cores,
  ~5% slower at 1) — the encoder is GPU-bound and the greedy is light host work, so
  the CUDA path frees the CPU. `XASR_NTHREADS=N` sets the ggml CPU thread count
  (CPU backend only); the harness defaults to 4.
- Set thread count for CPU runs with `XASR_NTHREADS`; limit CUDA host cores with
  `taskset -c`.

### Build note (this device)

The build script writes a `cuda_bf16.h` stub into the CUDA include dir via `sudo`.
If passwordless sudo isn't available, create the stub in a writable dir and add
`-I <dir>` to `CMAKE_CUDA_FLAGS` instead, and configure with `-DRS_XASR_DEV_TEST=ON`
to build the `xasr-dev-test` timing harness used for the table above.
