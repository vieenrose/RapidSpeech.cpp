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
2. **Vocos vocoder (ggml) — IN PROGRESS** — `tools/matcha_vocos_validate.cpp` is a standalone CPU
   validator: loads the gguf vocos weights, builds the ConvNeXt graph (conv_pre → norm_in → 8×
   {dw-conv → LN → pw1 → GELU → pw2 → ×gamma → residual} → norm_out → head), runs on the reference
   mel fixture (`scripts/gen_vocos_fixture.py`) and compares the magnitude output to ONNX.
   **Status: compiles + runs (harness proven), numeric match NOT yet achieved** (rel err ~1.35 —
   layout/interpretation bugs to debug: per-op ggml `ne` orientation, the head mag/phase split,
   GELU/eps). Layout reference captured in the source header. Next: dump per-layer intermediates
   vs ONNX to localize the mismatch, then wire the validated graph + iSTFT (reuse `cosyvoice3_hift`).
3. **gguf loader + hparams** — map tensors into a `MatchaModel` (encoder / dur-pred / CFM / vocos). TODO.
3. **Text frontend** — the bundle ships `tokens.txt` + `lexicon.txt` + jieba/espeak data +
   rule FSTs; reuse RapidSpeech's existing TTS frontend plumbing where possible. TODO.
4. **Forward graph (ggml)** — encoder → duration/length-regulate → CFM ODE loop (3 steps) →
   vocos. Needs: InstanceNorm (build from existing ops), Snake activation (`x + sin²`/Softplus
   form), `Sin` time-embedding, and an iSTFT (the one genuinely new primitive — compose from
   ggml rfft/irfft or a matmul-DFT, like RapidSpeech's existing CosyVoice3 HiFi head). TODO.
5. **Validate vs ONNX** — deterministic check (fixed noise) GPU==CPU and vs the sherpa-onnx
   reference wav; ASR round-trip. TODO.
6. **CUDA-10.2/sm_53 build + benchmark** — warm time + peak RSS in the container; add to the
   edge-speech-gpu-bench comparison. TODO.

## Effort note
Milestones 2–5 are a substantial arch implementation — comparable in size to the existing
`openvoice2.cpp` (VITS, ~2k lines), with the CFM ODE loop and the iSTFT head as the two pieces
with no direct precedent in the current arch set. This is multi-session work; the converter
(milestone 1) is the validated foundation it builds on.
