# Matcha-TTS → RapidSpeech.cpp (ggml) port — design & status

Goal: run [Luigi/matcha-zh-tw-en-8k](https://huggingface.co/Luigi/matcha-zh-tw-en-8k)
(a zh-TW/en, 8 kHz, code-mixed Matcha-TTS) on RapidSpeech.cpp's ggml backend, so it runs
cuDNN-free on the Jetson Nano gen1 (CUDA 10.2 / sm_53) alongside SenseVoice and the existing
melo8k/openvoice2 TTS. Counterpart to the sherpa-onnx cuDNN-free path (already validated).

## The model (from the two ONNX graphs)
- **Acoustic** `model-steps-3.onnx` — 4802 nodes, 305 weights, opset 14. A conditional
  flow-matching (CFM) TTS:
  - **Text encoder**: embedding (`model.encoder.emb`, n_vocab≈2190 × hidden 192) → conv prenet
    → transformer blocks (MatMul/Softmax attention, InstanceNorm).
  - **Duration predictor** → length regulator (repeat by predicted durations).
  - **CFM decoder**: a UNet (`Conv`/`ConvTranspose`/`InstanceNormalization`, Snake-style
    `Softplus`+`Tanh`, `Sin` time-embedding, `RandomNormalLike` initial noise) solved with
    **3 unrolled Euler ODE steps** (`num_ode_steps=3`).
- **Vocoder** `vocos-8khz-univ.onnx` — 168 nodes, 80 weights. Vocos: `conv_pre` → ConvNeXt
  blocks (depthwise `dw` conv + `LayerNorm` + GELU-`Erf` + per-channel `gamma`) → iSTFT head
  (`Cos`/`Sin` + the inverse STFT). (For ORT 1.11 the LayerNorm is decomposed to opset 16;
  on ggml it's a native op.)

## Milestones
1. **onnx→gguf converter** ✅ DONE — `scripts/convert_matcha_onnx_to_gguf.py`. Extracts all
   385 weights into one gguf + model-card hparams; round-trip-validated (gguf == onnx, PASS;
   ~104 MB f32). ONNX folds Linear weights into anonymous `onnx::MatMul_*` initializers, so the
   converter **traces the graph and gives them semantic names** (97 acoustic + 17 vocos resolved,
   e.g. `/blocks.0/pw1/MatMul`'s weight → `voc.blocks.0.pw1.weight`) — makes the gguf self-describing.
2. **Vocos vocoder (ggml) — ✅ DONE & VALIDATED** — `tools/matcha_vocos_validate.cpp` builds the
   ConvNeXt graph (conv_pre → norm_in → 8× {dw-conv → LN → pw1 → GELU → pw2 → ×gamma → residual}
   → norm_out → head) and matches the ONNX reference (`scripts/gen_vocos_fixture.py`) to
   **rel 3e-4 — F16 conv-kernel precision** (conv_pre, block0, and final magnitude all checked).
   The one bug was an input-layout slip: ggml conv data `[L,IC]` (element `(l,ic)` at `ic*L+l`)
   equals the ONNX `[80,T]` mel buffer directly — no transpose. Linear weights are numpy `[in,out]`
   → ggml `ne=[out,in]`, transposed before `mul_mat`. The iSTFT tail (mag/phase → waveform) is
   separate DSP to wire from `cosyvoice3_hift` (n_fft=512).
3. **iSTFT tail — ✅ DONE & VALIDATED** — head `[514,T]` → `mag = exp(min(log_mag, 9))`,
   `phase` → `re = mag·cos(phase)`, `im = mag·sin(phase)` → overlap-add iSTFT
   (n_fft=512, hop=128, periodic-hann, center, window-sum normalised, center-trimmed) → waveform.
   Implemented in `matcha_vocos_validate.cpp`; matches the numpy reference at **corr 1.000000,
   rel 8e-4** (same 4992 samples). So the **full vocoder (mel → waveform) is numerically correct.**
   (sherpa-onnx does this same iSTFT on CPU via knf::IStft — cheap deterministic tail, no GPU needed.)
4. **gguf loader + hparams** — map tensors into a `MatchaModel` (encoder / dur-pred / CFM / vocos). TODO.
5. **Text frontend** — the bundle ships `tokens.txt` + `lexicon.txt` + jieba/espeak data +
   rule FSTs; reuse RapidSpeech's existing TTS frontend plumbing where possible. TODO.
6. **Acoustic model forward graph (ggml) — the big remaining half** — text encoder (embedding →
   conv prenet → transformer blocks w/ InstanceNorm) → duration predictor → length-regulate →
   CFM decoder UNet solved with 3 Euler ODE steps (InstanceNorm, Snake `x + sin²`/Softplus form,
   `Sin` time-embedding, `RandomNormalLike` seed noise). Needs the same staged ONNX-intermediate
   validation that worked for the vocoder. TODO — this is the bulk of the remaining effort.
7. **End-to-end validate + CUDA-10.2/sm_53 build + benchmark** — deterministic (fixed-noise)
   GPU==CPU and vs the sherpa-onnx reference wav; ASR round-trip; warm time + peak RSS in the
   container; add to the edge-speech-gpu-bench comparison. TODO.

## Status summary
The **vocoder half (mel → waveform) is DONE and numerically validated** — Vocos ConvNeXt network
(rel 3e-4) + iSTFT tail (corr 1.0). What remains is the **acoustic model** (text → mel): a
substantial arch comparable to the existing `openvoice2.cpp` (~2k lines), with the CFM ODE loop as
the piece with no direct precedent. The staged ONNX-intermediate validation method (extract per-stage
reference, compare, fix layout) is proven and carries directly to the acoustic graph.
