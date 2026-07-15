<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" />
</div>

English | [简体中文](./README-CN.md)

<a href="https://huggingface.co/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/🤗-Hugging Face-blue"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech/files?version=main" target="_blank"><img src="https://img.shields.io/badge/ModelScope-blue"></a>
<a href="https://colab.research.google.com/drive/16U6k9zhdtfrEwVLP9a6ks99J0bEHNQyS?usp=sharing" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="https://rapidai-rapidspeech-wasm.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face wasm Demo-blue"></a>
<a href="https://rapidai-rapidspeech-wasm.ms.show" target="_blank"><img src="https://img.shields.io/badge/魔搭-wasm Demo-blue"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf"></a>


# RapidSpeech.cpp 🎙️

**RapidSpeech.cpp** is a high-performance, **edge-native speech intelligence framework** built on top of **ggml**.
It aims to provide **pure C++**, **zero-dependency**, and **on-device inference** for large-scale ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

------

## 🌟 Key Differentiators

While the open-source ecosystem already offers powerful cloud-side frameworks such as **vLLM-omni**, as well as mature on-device solutions like **sherpa-onnx**, **RapidSpeech.cpp** introduces a new generation of design choices focused on edge deployment.

### 1. vs. vLLM: Edge-first, not cloud-throughput-first

- **vLLM**
    - Designed for data centers and cloud environments
    - Strongly coupled with Python and CUDA
    - Maximizes GPU throughput via techniques such as PageAttention

- **RapidSpeech.cpp**
    - Designed specifically for **edge and on-device inference**
    - Optimized for **low latency, low memory footprint, and lightweight deployment**
    - Runs on embedded devices, mobile platforms, laptops, and even NPU-only systems
    - **No Python runtime required**

### 2. vs. sherpa-onnx: Deeper control over the inference stack

| Aspect | sherpa-onnx (ONNX Runtime) | **RapidSpeech.cpp (ggml)** |
| --- | --- | --- |
| **Memory Management** | Managed internally by ORT, relatively opaque | **Zero runtime allocation** — memory is fully planned during graph construction to avoid edge-side OOM |
| **Quantization** | Primarily INT8, limited support for ultra-low bit-width | **Full K-Quants family** (Q4_K / Q5_K / Q6_K), significantly reducing bandwidth and memory usage while preserving accuracy |
| **GPU Performance** | Relies on execution providers with operator mapping overhead | **Native backends** (`ggml-cuda`, `ggml-metal`) with speech-specific optimizations, outperforming generic `onnxruntime-gpu` |
| **Deployment** | Requires shared libraries and external config files | **Single binary deployment** — model weights and configs are fully encapsulated in **GGUF** |

------

## 📦 Model Support

**Automatic Speech Recognition (ASR)**
- [x] SenseVoice-small
- [x] FunASR-nano
- [x] X-ASR zh-en streaming **zipformer2 transducer** (k2-fsa / sherpa-onnx; 80-dim kaldi fbank, streaming chunked encoder + stateless decoder + joiner, greedy search). Numerically matches sherpa-onnx (encoder corr 1.0, token-exact transcription). GGUF: [Luigi/x-asr-zh-en-960ms-zipformer2-gguf](https://huggingface.co/Luigi/x-asr-zh-en-960ms-zipformer2-gguf) (f16 / Q4_K); convert with `scripts/xasr/convert_xasr_to_gguf.py`.
- [x] Qwen3-ASR **audio-encoder + Qwen3-0.6B LLM decoder** (Whisper-style 128-mel HTK frontend, 8× conv2d audio tower + 18-layer transformer, 2-layer projector into a 28-layer Qwen3 LLM; greedy autoregressive decode; native Traditional Chinese + punctuation, code-switched zh/en). Jetson-Nano-gen1 (Tegra X1, sm_53, CUDA 10.2): cuFFT batched-mel + flash-attn disabled; all weights on GPU. **Measured RTF ≈ 1.3** for a 13.5s clip with the q8_0 GGUF (transcript verified against the reference engine). GGUF: [Luigi/qwen3-asr-0.6b-rapidspeech-gguf](https://huggingface.co/Luigi/qwen3-asr-0.6b-rapidspeech-gguf); convert with `scripts/convert_qwen3_asr_to_gguf.py` (from a local `Qwen/Qwen3-ASR-0.6B` checkpoint).
- [x] MOSS-Transcribe-Diarize **transcription + speaker diarization + word timestamps** in one pass (Whisper-Medium 80-mel encoder + Qwen3-0.6B LLM decoder; audio tokens spliced into the LLM, emits `[start][Sxx]text[end]` segments). zh-TW / English meetings. Jetson Nano gen1 (Tegra X1, sm_53, CUDA 10.2): manual multi-head attention (no sm_53 flash-attn kernel), direct-DFT Whisper mel; validated **correct on-device**. Smart weight placement (measured, 12 s clip): with a get_rows-compatible embed (**embq8** gguf: `rs-quantize --token-embedding-type q8_0`) everything runs on the GPU — encoder 4.7 s, prefill 3.0 s, decode **554 ms/tok**, ~44 s total (2.5× over CPU-only); a legacy q6_K-embed gguf auto-falls-back to the per-component split (encoder GPU + LLM/decode on the CPU scheduler, decode 895 ms/tok) because sm_53 CUDA has no q6_K get_rows. Q4_K GGUF (~700 MB): [Luigi/moss-transcribe-diarize-zhtw-gguf](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-gguf); also runs **entirely in the browser via WebAssembly** (streaming decoder + optional CAM++ cross-window speaker linking). convert with `scripts/convert_moss_td_to_gguf.py`.
- [ ] FireRedASR2

**Text-to-Speech (TTS)**
- [x] OpenVoice2 (MeloTTS + voice cloning)
- [x] MeloTTS (VITS acoustic model; 8 kHz zh/en build runs on Jetson Nano gen1)
- [x] Matcha-TTS (CFM/flow-matching; 8 kHz zh-tw/en, built-in **zh+en** text frontend (en via espeak, `-DRS_MATCHA_ESPEAK=ON`), CUDA warm-persistent demo on Jetson Nano gen1 sm_53 — see [matcha-server](examples/matcha_server/))
- [x] OmniVoice (single-stage non-autoregressive diffusion TTS, multilingual + voice cloning)
- [x] MOSS-TTS-Nano-100M **autoregressive LLM-TTS** (OpenMOSS) — the first ggml port of the `MossTTSNanoForCausalLM` model (stock llama.cpp can't load it). Two-level AR: 12-layer GPT-2 global transformer (interleaved RoPE, gelu_new) + 1-layer local decoder over **16 RVQ codebooks** (tied heads), then the MOSS-Audio-Tokenizer-Nano codec (12.5 fps → 48 kHz). Torch-free / ORT-free — converters + parity harnesses validate every stage against the reference ONNX: global prefill + KV-cache decode **MSE 1.4e-6**, and the assembled greedy generation loop matches ONNX **bit-exact** (frames 0–5, 16/16 codebooks, F32). Voice cloning (incl. zh-TW/台湾腔 accent) from a reference clip, temperature/top-k sampling. Jetson Nano gen1 (Tegra X1, sm_53): Q8_0 weights, **RTF 0.86** stock ggml-CUDA → **~0.35** with a custom warp-reduce `mul_mat_vec` kernel (2.6× over stock on Maxwell; see [`tools/moss_cuda/`](tools/moss_cuda/)). GGUF: [Luigi/MOSS-TTS-Nano-100M-GGUF](https://huggingface.co/Luigi/MOSS-TTS-Nano-100M-GGUF) (codec from [hans00/MOSS-TTS-Nano-GGUF](https://huggingface.co/hans00/MOSS-TTS-Nano-GGUF) via [codec.cpp](https://github.com/mybigday/codec.cpp)); convert with `scripts/convert_moss_nano_to_gguf.py`. See [`docs/moss_streaming_integration.md`](docs/moss_streaming_integration.md).
- [x] PrimeTTS (v2 / v2.1) — single-speaker **MB-iSTFT-VITS**: TextEncoder (6× windowed rel-pos transformer) → deterministic duration predictor → residual-coupling flow → **multiband-iSTFT + PQMF** generator → 16 kHz waveform. Compact and fast (no BERT, no speaker embedding, no SDP); backend-agnostic ggml graphs. See [Luigi/PrimeTTS](https://huggingface.co/Luigi/PrimeTTS).
- [ ] CosyVoice3
- [ ] Qwen3-TTS

------

## 🏗️ Architecture Overview

RapidSpeech.cpp is not just an inference wrapper — it is a full-featured speech application framework:

- **Core Engine**
  A `ggml`-based computation backend supporting mixed-precision inference from INT4 to FP32.

- **Architecture Layer**
  A plugin-style model construction and loading system, with support for FunASR-nano, SenseVoice, X-ASR, Qwen3-ASR, MOSS-Transcribe-Diarize, MeloTTS, Matcha-TTS, OmniVoice, MOSS-TTS-Nano, PrimeTTS (MB-iSTFT-VITS), and planned support for CosyVoice, Qwen3-TTS, and more.

- **Business Logic Layer**
  Built-in ring buffers, VAD (voice activity detection), text frontend processing (e.g., phonemization), and multi-session management.

------

## 🚀 Core Features

- [x] **Extreme Quantization**: Native support for 4-bit, 5-bit, and 6-bit quantization schemes to match diverse hardware constraints.
- [x] **Zero Dependencies**: Implemented entirely in C/C++, producing a single lightweight binary.
- [x] **GPU / NPU Acceleration**: Customized CUDA and Metal backends optimized for speech models.
- [x] **Unified Model Format**: Both ASR and TTS models use an extended **GGUF** format.
- [x] **Python Bindings**: Python API via pybind11, installable with `pip install`.

------

## 🛠️ Quick Start

### Download Models

Models are available on:

- 🤗 Hugging Face: https://huggingface.co/RapidAI/RapidSpeech
- ModelScope: https://www.modelscope.cn/models/RapidAI/RapidSpeech

### Build from Source

```bash
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
git submodule sync && git submodule update --init --recursive
cmake -B build
cmake --build build --config Release
```

Build artifacts are located in the `build/` directory:
- `rs-asr-offline` — Offline ASR command-line tool
- `rs-asr-vad-online` — VAD-segmented quasi-streaming ASR command-line tool
- `rs-tts-offline` — Offline TTS command-line tool
- `rs-quantize` — Model quantization tool
- `matcha-server` — warm-persistent Matcha-TTS demo (Jetson Nano gen1 / sm_53; see below)

### C++ CLI Usage

#### Offline Recognition (rs-asr-offline)

**Basic — single file without VAD:**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

**With VAD segmentation (recommended for long audio):**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

When a VAD model is provided, the tool automatically segments the audio by speech activity and produces timestamped results per segment.

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to GGUF model file (required) | — |
| `-w, --wav` | Path to WAV audio file (16 kHz, required) | — |
| `-v, --vad` | Path to VAD GGUF model — Silero or FireRed, auto-detected from `general.architecture` (optional, enables VAD segmentation) | — |
| `-t, --threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |
| `--vad-threshold` | VAD speech probability threshold (0–1, lower = more sensitive) | 0.5 |
| `--silence-ms` | Silence duration to split segments (ms) | 600 |
| `--max-segment-s` | Max segment length for ASR input (seconds) | 30.0 |

#### VAD-Segmented Quasi-Streaming Recognition (rs-asr-vad-online)

**WAV file (simulate streaming):**

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

**Microphone (live mode):**

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  --mic \
  -t 4
```

**Two-pass mode (CTC fast pass + LLM rescoring, FunASR-Nano only):**

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  --two-pass
```

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to ASR GGUF model file (required) | — |
| `-v, --vad` | Path to Silero VAD model file (required) | — |
| `-w, --wav` | Path to WAV audio file (16 kHz) | — |
| `--mic` | Use microphone input (live mode) | off |
| `--mic-device` | Audio device index for mic input | auto |
| `--mic-chunk-ms` | Mic read chunk size (ms) | 32 |
| `-t, --threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |
| `--vad-threshold` | VAD speech detection threshold (0–1, lower = more sensitive) | 0.5 |
| `--silence-ms` | Silence timeout for segment splitting (ms) | 600 |
| `--two-pass` | Enable 2-pass mode: CTC decode + LLM rescore | off |
| `--ctc-precheck` | CTC pre-check before LLM to skip silence (reduces hallucination, slightly increases RTF) | off |

#### Text-to-Speech (rs-tts-offline)

##### MeloTTS / OpenVoice2

OpenVoice2 builds on [MeloTTS](https://github.com/myshell-ai/MeloTTS) as the base acoustic model (VITS-style: text encoder + duration predictor + stochastic flow decoder + HiFi-GAN vocoder). MeloTTS ships one checkpoint per language; the `--lang` flag must match the language of the GGUF you converted.

**English (MeloTTS-English):**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --lang English \
  -o output.wav \
  --threads 4
```

**Chinese (MeloTTS-Chinese):**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-zh.gguf \
  -t "你好，欢迎使用 RapidSpeech 语音合成。" \
  --lang Chinese \
  -o output.wav
```

**Japanese (MeloTTS-Japanese):**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-jp.gguf \
  -t "こんにちは、RapidSpeech へようこそ。" \
  --lang Japanese \
  -o output.wav
```

Accepted `--lang` values: `English`/`EN`/`en`, `Chinese`/`ZH`/`zh`, `Japanese`/`JA`/`ja`. The language string is case-insensitive but must match the model's language — feeding Chinese text to an English model will produce garbled audio.

> ⚠️ **`--lang` also selects the speaker id, so it is mandatory for non-English models.** It is
> not just a front-end hint: `Chinese` → speaker 1, English/Japanese/other → speaker 0. The
> CLI **defaults to English (speaker 0)** when `--lang` is omitted. MeloTTS-Chinese has only one
> valid voice — speaker **1** (`spk2id = {'ZH': 1}`) — so running a Chinese GGUF without
> `--lang Chinese` picks speaker 0, feeds the wrong speaker embedding into the flow/vocoder, and
> yields **silent or near-silent audio** (the decoder log-magnitudes collapse, not just a wrong
> timbre). Always pass `--lang Chinese` for ZH models. The same applies to the distilled 8 kHz
> melo build ([Luigi/vits-melo-tts-zh_en-8k](https://huggingface.co/Luigi/vits-melo-tts-zh_en-8k)).

**Voice cloning (OpenVoice2 = MeloTTS base + Tone Color Converter):**

OpenVoice2 separates speaker timbre from prosody. Pass a reference WAV with `--ref` to apply the speaker's voice to the synthesized speech. Requires the converter GGUF in the same directory as the base GGUF (the loader auto-discovers it).

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, this is cloned voice." \
  --lang English \
  --ref /path/to/reference.wav \
  -o output.wav
```

##### OmniVoice (diffusion TTS, multilingual + voice cloning)

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

**Voice cloning (OmniVoice):**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, this is cloned voice." \
  --ref /path/to/reference.wav \
  --ref-text "transcript of the reference audio" \
  -o output.wav
```

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to TTS GGUF model file (required) | — |
| `-t, --text` | Text to synthesize (required) | — |
| `-o, --output` | Output WAV file path | output.wav |
| `--lang` | Target language. MeloTTS: `English`/`Chinese`/`Japanese` (must match GGUF). OmniVoice: `English`/`zh`/... | English |
| `--ref` | Reference audio WAV for voice cloning (OpenVoice2 / OmniVoice) | — |
| `--ref-text` | Transcript of the reference audio (OmniVoice only) | — |
| `--bert` | ZH BERT GGUF (1024-dim, OpenVoice2 Chinese only, optional) | — |
| `--mbert` | Multilingual BERT GGUF (768-dim, optional) | — |
| `--instruct` | Voice description, e.g. `male`, `female`, `young adult` (OmniVoice) | male |
| `--seed` | Random seed (OmniVoice) | 42 |
| `--n-steps` | Diffusion steps 1-128, fewer = faster but lower quality (OmniVoice) | 32 |
| `--threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |

#### Model Quantization (rs-quantize)

```bash
./build/rs-quantize /path/to/funasr-nano-fp16.gguf /path/to/output-q4_k.gguf q4_k
```

Supported quantization types: `q4_0`, `q4_k`, `q5_0`, `q5_k`, `q8_0`, `f16`, `f32`

> ⚠️ **Note**: Q2_K quantization causes unacceptable accuracy loss for FunASR Nano, producing garbled output. Not recommended.

### Jetson Nano gen1 (Maxwell sm_53, CUDA 10.2) — cross-build + warm Matcha demo

The original Jetson Nano (Tegra X1, Maxwell **sm_53**, CUDA **10.2**, gcc 8.3, glibc 2.27)
is supported through an **x86 → aarch64 cross-build**. CUDA 10.2 / gcc 8.3 cannot compile
modern ggml-CUDA out of the box, so `scripts/build_jetson_nano_gen1.sh` force-includes a
small set of compat shims in `scripts/nano_cuda_compat/`:

| Shim | Purpose |
| --- | --- |
| `nvcc_compat.h` | `__builtin_assume` → noop, `CUDA_R_16BF` → `CUDA_R_16F`, `CUBLAS_COMPUTE_*` / `CUBLAS_TF32_TENSOR_OP_MATH` back-defines (force-included into every nvcc TU) |
| `cuda_bf16.h` | minimal `bf16` → `fp16` stub (CUDA 10.2 ships none) |
| `neon_x4_shim.h` | `vld1q_{s8,u8}_x4` (gcc-9 NEON intrinsics absent in gcc-8.3); host C/C++ TUs only — must **not** enter `.cu` (nvcc can't parse `arm_neon.h`) |

```bash
# in the aarch64 cross toolchain container (gcc-arm-8.3 + cuda-cross-aarch64-10-2):
scripts/build_jetson_nano_gen1.sh          # -> build-nano/  (CUDA, sm_53)
# CPU-only native build on the device:
scripts/build_jetson_nano_gen1_native.sh
```

#### Matcha-TTS from text (built-in zh + en frontend)

`rs-tts-offline` synthesizes Matcha directly from **mixed Chinese/English** text using the
built-in frontend (`frontend/matcha_frontend.{h,cpp}`): point it at the model's `tokens.txt`
+ `lexicon.txt` (zh-TW and zh both covered, no conversion) via env vars.

```bash
MATCHA_TOKENS=tokens.txt MATCHA_LEXICON=lexicon.txt \
  ./build/rs-tts-offline -m matcha8k.gguf -t "你好，我是 Amy，很高興為您服務。" -o out.wav
```

The frontend is a **faithful, byte-identical port** of sherpa-onnx's matcha frontend
(`matcha-tts-lexicon.cc` + phrase-matcher + text-utils + piper `ProcessPhonemes`) — proven
by a differential test (222/222 cases, see `tools/matcha_frontend_parity/`):
- **Normalization** → full-width punctuation to half-width (`，→,`, `：→,`, `。→.`, …).
- **Chinese**: `SplitUtf8` + greedy **phrase matching** against the lexicon (correct polyphones
  like 行長/長城/市長), recursing into per-char ids; OOV CJK dropped.
- **English**: espeak-ng IPA → the model's token set via the diphthong replacement table
  (`eɪ→A, oʊ→O, aɪ→I`, …; sherpa PR #2853), with a blank (space) token inserted before a word
  whenever the previous word was alphabetic. Build with `-DRS_MATCHA_ESPEAK=ON`
  (`-DESPEAK_NG_ROOT=<dir>`). For byte-identity use **espeak-ng 1.52** (the version the model was
  built with; 1.51 differs on stress marks). Without espeak, English is skipped.
- **Synthesis**: clause-split at punctuation + sherpa's `ScaleSilence` (silence_scale=0.2) so an
  English word embedded between Chinese renders cleanly and inter-clause pauses stay natural.

#### matcha-server — warm-persistent Matcha-TTS demo (`examples/matcha_server/`)

A demo of the **warm-persistent** pattern for Maxwell: Matcha is launch-bound on sm_53
(PTX-JIT + first-graph build dominate a single call), so `matcha-server` loads the gguf +
ggml-CUDA backend **once**, warms the kernels once, then serves synthesis requests in a
loop. Steady-state RTF drops from **~1.0 cold to ~0.18–0.22 warm** on real Nano gen1.
It speaks a line protocol on stdin (`<phoneme_ids_file> <out.wav> [length_scale]` →
`OK <wav> <nsamples> <synth_ms>`) — it takes pre-computed phoneme-ID files (from the
built-in frontend or sherpa-onnx's), staying independent of any one frontend. This demo
is the engine loop a production LiveKit-Agents TTS plugin is built from. See
`examples/matcha_server/README.md`.

```bash
MATCHA_USE_CUDA=1 ./build-nano/matcha-server matcha8k.gguf
```

### Python Usage

#### Installation

```bash
# CPU (Linux / macOS / Windows)
pip install rapidspeech

# CUDA (Linux, NVIDIA GPU)
pip install rapidspeech-cuda

# Metal (macOS, Apple Silicon)
pip install rapidspeech-metal
```

Supported backends:

| Backend | Distribution        | Platform                   | Notes                                                |
|---------|---------------------|----------------------------|------------------------------------------------------|
| CPU     | `rapidspeech`       | Linux / macOS / Windows    | Default; no GPU dependency                           |
| CUDA    | `rapidspeech-cuda`  | Linux + NVIDIA GPU         | Built against CUDA 11.8 (manylinux2014)              |
| Metal   | `rapidspeech-metal` | macOS (Apple Silicon)      | Uses fused Metal kernels for DAC vocoder             |
| Vulkan  | _source build only_ | Linux / Windows + Vulkan SDK | Cross-vendor GPU acceleration                      |
| CANN    | _source build only_ | Linux + Huawei Ascend NPU  | Requires CANN toolkit                                |
| OpenCL  | _source build only_ | Linux / Android + OpenCL ICD | Mobile / embedded GPUs                             |
| WebGPU  | _source build only_ | Native (Dawn) / WASM       | Browser deployment via emdawnwebgpu                  |

> Note: The pip distribution name varies by backend, but the Python import name is always `rapidspeech`. Backends marked _source build only_ are not published to PyPI — build them from source (see below).

#### Build Python Package from Source

```bash
# CPU build (default)
pip install .

# Pre-wired backends (recognized by setup.py, sets the wheel name automatically)
RS_BACKEND=cuda  pip install .              # requires nvcc in PATH
RS_BACKEND=metal pip install .              # macOS, auto-enabled on Apple Silicon

# Other backends — pass the CMake flag directly through RAPIDSPEECH_CMAKE_ARGS
RAPIDSPEECH_CMAKE_ARGS="-DRS_VULKAN=ON" pip install .   # Vulkan
RAPIDSPEECH_CMAKE_ARGS="-DRS_CANN=ON"   pip install .   # Huawei Ascend
RAPIDSPEECH_CMAKE_ARGS="-DRS_OPENCL=ON" pip install .   # OpenCL
RAPIDSPEECH_CMAKE_ARGS="-DRS_WEBGPU=ON" pip install .   # WebGPU (Dawn)
```

#### Python API

```python
import rapidspeech
import numpy as np

# Initialize ASR context
ctx = rapidspeech.asr_offline(
    model_path="funasr-nano-fp16.gguf",
    n_threads=4,
    use_gpu=True
)

# Read WAV audio (16 kHz, float32, mono)
pcm = ...  # np.ndarray, shape=[N], dtype=float32

# Push audio and recognize
ctx.push_audio(pcm)
ctx.process()

# Get recognition result
text = ctx.get_text()
print(f"Result: {text}")
```

See [python-api-examples/asr/asr-offline.py](python-api-examples/asr/asr-offline.py) for a complete example.

**TTS Python API:**

```python
import rapidspeech
import numpy as np

# Initialize TTS synthesizer
tts = rapidspeech.tts_synthesizer(
    model_path="openvoice2-base.gguf",
    n_threads=4,
    use_gpu=True
)

# Synthesize text to audio (returns full PCM as numpy array)
pcm = tts.synthesize("Hello, welcome to RapidSpeech!")

# Streaming synthesis (returns list of numpy array chunks)
chunks = tts.synthesize_streaming("Hello, welcome to RapidSpeech!")
for chunk in chunks:
    print(f"Chunk: {len(chunk)} samples")

# Optional: set reference audio for voice cloning
# reference_pcm = ...  # load reference audio
# tts.set_reference(reference_pcm, sample_rate=16000)
```

------

## 🧪 Examples & Bindings

End-to-end examples for every language binding live in their own folders, each
with a dedicated README that walks through installation, CLI flags, and the
underlying API surface.

| Folder | What it covers | README |
|--------|----------------|--------|
| 🐍 **Python** | `pip install rapidspeech` → offline / online ASR (with neural VAD, 2-pass LLM rescoring), offline / streaming TTS, voice cloning | [`python-api-examples/README.md`](python-api-examples/README.md) |
| 🌐 **Browser (WebAssembly)** | Three-tab demo: offline ASR, mic-driven online ASR, offline TTS. Runs locally with WebGPU + pthreads | [`wasm-examples/README.md`](wasm-examples/README.md) |
| 🟩 **Node.js** | CLI built on the same WASM module: file → ASR (with optional VAD + 2-pass), text → TTS (with voice cloning) | [`node-api-example/README.md`](node-api-example/README.md) |
| 💻 **C++ CLI** | `rs-asr-offline` / `rs-asr-vad-online` / `rs-tts-offline` / `rs-quantize` | this README (sections above) |
| ☁️ **Colab notebook** | Build the CLI on a free T4, run ASR/TTS, use the Python API end-to-end | [`colab/README.md`](colab/README.md) |
| 🤗 **HuggingFace Space** | Deploy the browser demo as a Docker-SDK Space (COOP/COEP-ready) | [`huggingface-space/HOWTO.md`](huggingface-space/HOWTO.md) |

Quick taste of each:

```bash
# Python — VAD-segmented 2-pass transcription
python python-api-examples/asr/asr-offline.py \
    --model funasr-nano.gguf --audio long.wav \
    --vad silero-vad.gguf --two-pass

# Browser — three tabs in one page
cd wasm-examples && python3 serve.py 8000  # then open http://localhost:8000

# Node.js — same WASM module, file-based ASR/TTS
node node-api-example/index.js asr -m funasr-nano.gguf -w audio.wav --two-pass
node node-api-example/index.js tts -m omnivoice.gguf -t "Hello world" -o out.wav
```

------

## 📊 Performance Benchmarks

Test environment: Apple M1 Pro, funasr-nano-fp16.gguf, 15s audio

| Configuration | RTF | Wall Time | Notes |
| --- | --- | --- | --- |
| CPU -t 4 | 0.465 | 12.4s | CPU-only inference |
| GPU -t 4 | 0.170 | 5.2s | Metal acceleration |
| GPU -t 4 Q4_K | 0.756 | — | Quantized model: GPU dequant overhead |
| CPU -t 4 Q4_K | 0.530 | — | Quantized model CPU inference, 596 MB (3.3× compression) |

> RTF (Real-Time Factor) = Processing time / Audio duration. Lower is faster. RTF < 1 means faster than real-time.

------

## 🔧 Model Format Conversion

### ASR Model (HF → GGUF)

A conversion tool from HuggingFace models to GGUF format is provided:

```bash
python scripts/convert_hf_to_gguf.py \
  --model /path/to/hf-model-dir \
  --outfile /path/to/output.gguf \
  --outtype f16
```

### Silero VAD Model (safetensors → GGUF)

To convert the Silero VAD model for use with `rs-asr-vad-online` or offline VAD segmentation:

```bash
python scripts/convert_silero_to_gguf.py \
  --model /path/to/silero_vad_16k.safetensors \
  --output /path/to/silero_vad_v6.gguf
```

The converted VAD model is also available for direct download from [HuggingFace](https://huggingface.co/RapidAI/RapidSpeech) and [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidSpeech).

### TTS Model (OpenVoice2 / MeloTTS → GGUF)

Convert MeloTTS (OpenVoice2 base) and the optional Tone Color Converter to GGUF. MeloTTS releases one HuggingFace repo per language; choose the matching `--base-model` and `--language` tag.

```bash
# English
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --output-dir ./models \
  --language EN

# Chinese
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-Chinese \
  --output-dir ./models \
  --language ZH

# Japanese
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-Japanese \
  --output-dir ./models \
  --language JA

# With Tone Color Converter (enables voice cloning via --ref)
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --converter-model myshell-ai/OpenVoiceV2 \
  --output-dir ./models \
  --language EN
```

Outputs:
- `openvoice2-base-<lang>.gguf` — Text encoder + duration predictor + flow decoder + HiFi-GAN vocoder
- `openvoice2-converter.gguf` — Tone color converter (only when `--converter-model` is supplied; needed for `--ref` voice cloning)

### TTS Model (OmniVoice → GGUF)

Merge OmniVoice PyTorch model (LLM + audio tokenizer) into a single GGUF:

```bash
python scripts/convert_omnivoice_to_gguf.py \
  --model /path/to/omnivoice-model \
  --tokenizer /path/to/omnivoice-audio-tokenizer \
  --output /path/to/omnivoice-merged.gguf \
  --outtype f16
```

------

## 🤝 Contributing

If you are interested in the following areas, we welcome your PRs or participation in discussions:

- Adapting more models to the framework.
- Refining and optimizing the project architecture.
- Improving inference performance.

## Acknowledgements

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
