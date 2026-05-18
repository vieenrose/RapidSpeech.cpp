# OmniVoice TTS — Usage Guide & Quality Evaluation

OmniVoice is a single-stage non-autoregressive diffusion TTS model (Qwen3-0.6B backbone,
8-codebook MaskGIT + DAC vocoder). This document covers quantization selection, diffusion
step tuning, and audio quality benchmarks.

## Quick Start

```bash
# Basic synthesis
./build/rs-tts-offline -m models/omnivoice-f16.gguf \
  -t "Hello, this is a test." -o output.wav

# With voice cloning
./build/rs-tts-offline -m models/omnivoice-q4_k_m.gguf \
  -t "Your text here." --ref reference.wav --ref-text "transcript" \
  --n-steps 32 -o output.wav

# Chinese synthesis
./build/rs-tts-offline -m models/omnivoice-q4_k_m.gguf \
  -t "你好，这是一段测试文本。" --lang zh --n-steps 32 -o output.wav
```

## Model Selection

| Model | Size | bpw | Quality | Speed | Recommendation |
|-------|------|-----|---------|-------|----------------|
| **F16** | 1552 MB | 16.0 | ★★★★★ | ★★★ | Baseline, development |
| **Q8_0** | 1073 MB | 8.5 | ★★★★★ | ★★★★ | Near-lossless |
| **Q5_K_M** | 870 MB | ~5.5 | ★★★★★ | ★★★★ | High quality daily use |
| **Q4_K_M** | 823 MB | ~4.5 | ★★★★ | ★★★★ | **Recommended default** |
| **Q3_K** | 696 MB | ~3.5 | ★★★½ | ★★★★ | Good quality/size balance |
| **IQ3_XXS** | 687 MB | ~3.06 | ★★★ | ★★★★ | Minimum viable for TTS |
| **Q2_K** | 635 MB | ~2.6 | ★★ | ★★★★ | Degraded, use with caution |
| **IQ2_XXS** | 606 MB | ~2.06 | ★ | ★★★★ | **Not recommended** |

> **Key insight**: IQ2_XXS (2.06 bpw) fails on OmniVoice TTS regardless of imatrix
> calibration. The 28-layer Qwen3 backbone has insufficient parameter redundancy to
> absorb 2-bit weight noise, causing logit collapse in MaskGIT codebook sampling.
> IQ3_XXS (~3.06 bpw) is the practical minimum for intelligible output. See
> [Why IQ2_XXS fails](#why-iq2_xxs-fails) below.

## Diffusion Steps

MaskGIT iteratively unmasks acoustic tokens. More steps = more refinement opportunities.

### Step Count vs Quality

| Steps | F16 Quality | IQ3_XXS Quality | Speed (RTF) | Use Case |
|-------|------------|-----------------|-------------|----------|
| 8 | ★★★ | ★★ | ~0.3× | Fast preview |
| 16 | ★★★★ | ★★★ | ~0.5× | Development testing |
| **32** | **★★★★★** | **★★★½** | **~1.0×** | **Default (FP16/Q4_K+)** |
| 50 | ★★★★★ | ★★★★ | ~1.5× | IQ3_XXS / Q3_K |
| 64 | ★★★★★ | ★★★★½ | ~2.0× | IQ3_XXS最佳质量 |
| 128 | ★★★★★ | ★★★★★ | ~4.0× | Maximum quality (slow) |

### Per-Model Recommended Steps

```
F16 / Q8_0 / Q5_K_M / Q4_K_M:   --n-steps 32
Q3_K / IQ3_XXS:                  --n-steps 50
Q2_K:                            --n-steps 64 (still degraded)
```

## CFG & Temperature

OmniVoice uses Classifier-Free Guidance. Quantized models benefit from adjusted CFG scale
and temperature to compensate for reduced logit contrast.

The engine auto-applies quantization-aware CFG adjustments based on detected weight precision:

| Quant Level | CFG Scale | Temperature |
|-------------|-----------|-------------|
| F16 / Q8_0 | 2.0 | 1.0 |
| Q4_0 / Q4_1 / Q4_K | 2.5 | 0.9 |
| Q2_K / IQ2 / IQ3 / IQ4 | 3.0 | 0.85 |

These are applied automatically — no user configuration needed.

## Performance (RTF)

Real-Time Factor = synthesis time / audio duration. RTF < 1.0 means faster than real-time.

Test machine: Apple M1 Pro, GPU (Metal). Sentence: "The quick brown fox jumps over the lazy dog."

### English (en) — "The quick brown fox jumps over the lazy dog." (2.60 s)

| Model | Steps=16 | Steps=32 | Steps=50 |
|-------|----------|----------|----------|
| F16 | 0.73 | 0.90 | 1.31 |
| Q5_K_M | 0.74 | 1.09 | 1.62 |
| Q4_K_M | 0.61 | 1.11 | 1.54 |
| IQ3_XXS | 0.61 | 1.04 | 1.53 |
| Q3_K | 0.69 | 1.09 | 1.59 |
| Q2_K | 0.64 | 0.98 | 1.44 |

### Chinese (zh) — "人工智能正在改变我们与计算机交互的方式。" (4.24 s)

| Model | Steps=16 | Steps=32 | Steps=50 |
|-------|----------|----------|----------|
| F16 | 0.52 | 0.81 | 1.11 |
| Q5_K_M | 0.53 | 0.92 | 1.43 |
| Q4_K_M | 0.52 | 0.87 | 1.30 |
| IQ3_XXS | 0.50 | 0.87 | 1.33 |
| Q3_K | 0.51 | 0.89 | 1.44 |
| Q2_K | 0.48 | 0.82 | 1.46 |

> All RTF values measured on Apple M1 Pro (Metal GPU), 8 threads. Lower is faster; <1.0 = faster than real-time.
> IQ types (IQ3_XXS) are slightly faster than K-quants at equivalent step counts due to optimized Metal matrix-multiply kernels.

> RTF data will be filled after benchmark completion.

## Audio Samples

Click to listen and compare quality across models and step counts.

### English — "The quick brown fox jumps over the lazy dog."

#### 16 Steps

| Model | Audio                                                                              |
|-------|------------------------------------------------------------------------------------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_en_s16.wav"></audio>     |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_en_s16.wav"></audio>  |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_en_s16.wav"></audio> |
| Q2_K | <audio controls src="../../assets/omnivoice/omnivoice-q2_k_en_s16.wav"></audio>    |

#### 32 Steps

| Model | Audio |
|-------|-------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_en_s32.wav"></audio> |
| Q5_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q5_k_m_en_s32.wav"></audio> |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_en_s32.wav"></audio> |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_en_s32.wav"></audio> |
| Q3_K | <audio controls src="../../assets/omnivoice/omnivoice-q3_k_en_s32.wav"></audio> |
| Q2_K | <audio controls src="../../assets/omnivoice/omnivoice-q2_k_en_s32.wav"></audio> |

#### 50 Steps

| Model | Audio |
|-------|-------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_en_s50.wav"></audio> |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_en_s50.wav"></audio> |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_en_s50.wav"></audio> |
| Q3_K | <audio controls src="../../assets/omnivoice/omnivoice-q3_k_en_s50.wav"></audio> |
| Q2_K | <audio controls src="../../assets/omnivoice/omnivoice-q2_k_en_s50.wav"></audio> |

### Chinese — "人工智能正在改变我们与计算机交互的方式。"

#### 16 Steps

| Model | Audio |
|-------|-------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_zh_s16.wav"></audio> |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_zh_s16.wav"></audio> |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_zh_s16.wav"></audio> |

#### 32 Steps

| Model | Audio |
|-------|-------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_zh_s32.wav"></audio> |
| Q5_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q5_k_m_zh_s32.wav"></audio> |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_zh_s32.wav"></audio> |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_zh_s32.wav"></audio> |
| Q3_K | <audio controls src="../../assets/omnivoice/omnivoice-q3_k_zh_s32.wav"></audio> |
| Q2_K | <audio controls src="../../assets/omnivoice/omnivoice-q2_k_zh_s32.wav"></audio> |

#### 50 Steps

| Model | Audio |
|-------|-------|
| F16 | <audio controls src="../../assets/omnivoice/omnivoice-f16_zh_s50.wav"></audio> |
| Q4_K_M | <audio controls src="../../assets/omnivoice/omnivoice-q4_k_m_zh_s50.wav"></audio> |
| IQ3_XXS | <audio controls src="../../assets/omnivoice/omnivoice-iq3_xxs_zh_s50.wav"></audio> |
| Q3_K | <audio controls src="../../assets/omnivoice/omnivoice-q3_k_zh_s50.wav"></audio> |

## Voice Cloning

Provide a reference audio clip to clone the speaker's voice:

```bash
./build/rs-tts-offline \
  -m models/omnivoice-q4_k_m.gguf \
  -t "Your text to synthesize." \
  --ref speaker_reference.wav \
  --ref-text "transcript of the reference audio" \
  --n-steps 32 \
  -o output.wav
```

Requirements:
- Reference audio: WAV format, any sample rate (will be resampled to 24kHz)
- Reference text: exact transcript of what the speaker says in the reference clip
- For best results, use ≥3 seconds of reference audio with clean speech

## Voice Control (without reference)

Control voice characteristics via `--instruct`:

```bash
./build/rs-tts-offline -m models/omnivoice-q4_k_m.gguf \
  -t "Your text." --instruct "female, young adult, high pitch, fast speed" -o output.wav
```

Effective instruct descriptors: `male`/`female`, `young adult`/`middle-aged`/`senior`,
`low pitch`/`moderate pitch`/`high pitch`, `slow speed`/`fast speed`.

## imatrix Calibration

For activation-aware quantization with IQ1/IQ2 types:

```bash
# Step 1: Collect imatrix on the F16 model
./build/rs-imatrix -m models/omnivoice-f16.gguf \
  -o imatrix-f16.dat --n-steps 16 --gpu

# Step 2: Quantize with imatrix
./build/rs-quantize models/omnivoice-f16.gguf \
  models/omnivoice-iq3_xxs.gguf iq3_xxs --imatrix imatrix-f16.dat
```

Note: Even with imatrix, IQ2_XXS fails on OmniVoice. IQ3_XXS is the practical minimum.

## Why IQ2_XXS Fails

At 2.06 bits-per-weight, each 1024-dimensional hidden vector carries only ~2109 bits of
information. The MaskGIT codebook classification (8 codebooks × 1025 classes each) requires
precise logit separation that IQ2_XXS cannot provide:

1. **Grid too coarse**: IQ2_XXS maps weights to a small precomputed lattice; each weight
   can only take ~4 distinct values. Across 28 transformer layers, the accumulation of
   quantization noise destroys the fine-grained differences needed for codebook selection.

2. **CFG amplifies noise**: Classifier-Free Guidance subtracts uncond from cond logits and
   multiplies by CFG scale (2.0–3.0). This 2-3× amplification applies to quantization
   noise as well as signal.

3. **No parameter redundancy**: With only ~300M backbone parameters, there is no excess
   capacity to absorb the information loss from 2-bit quantization. Large LLMs (7B+)
   tolerate IQ2_XXS because they have orders of magnitude more redundancy.

4. **imatrix cannot fix this**: Importance weighting helps by preserving sensitive columns,
   but at 2.06 bpw the fundamental information capacity is below the task requirement.

## Stability Tips

- **Long text (>100 chars)**: Set `--n-steps` higher. RoPE positions beyond the training
  context (4096) produce unfamiliar embeddings. More steps give the model more chances to
  self-correct.
- **Chinese + English mixed**: Use `--lang zh` for Chinese-dominant text, `--lang en`
  otherwise. The language token affects the style embedding.
- **Reproducibility**: Use `--seed <N>` for deterministic output.
- **GPU vs CPU**: GPU (Metal/CUDA) is ~2-3× faster. For IQ types, GPU benefit is larger
  because the IQ matrix multiply kernels are heavily optimized.
