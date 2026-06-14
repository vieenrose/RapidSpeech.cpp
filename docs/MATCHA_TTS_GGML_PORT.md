# Matcha-TTS → RapidSpeech.cpp (ggml) port — design & status

> **✅ COMPLETE + OPTIMIZED.** The full pipeline runs end-to-end in ggml (`arch/matcha.cpp`, registered
> as `matcha-tts`) and produces correct speech. Every stage validated vs ONNX: encoder rel ≤1e-4,
> length regulator rel 0, CFM decoder mel **corr 0.999993**, Vocos+iSTFT corr 1.0; end-to-end audio
> (matched durations) **corr 0.952** to the ONNX path. Runs on **CPU or CUDA** (backend-sched). Warm
> synth for 3.18 s of 8 kHz audio (GB10): **CPU 113 ms** (RTF 0.036, 493 MB) / **CUDA 36 ms** (RTF 0.011,
> + ~57 s one-time sm_53→sm_121 JIT) — opt into CUDA with `MATCHA_USE_CUDA=1`. Validators:
> `tools/matcha_*_validate.cpp`, e2e harness `tools/matcha_e2e_test.cpp`. Remaining: a zh-TW/en text
> frontend to produce `phoneme_ids` (currently caller-injected).

## Performance optimization (auto-research)
A spec-extract + adversarial-verify Workflow plus direct profiling drove the optimization. Findings:
- **The iSTFT was 75% of runtime** (730 ms of 971 ms) — it was a naive **O(N²) DFT** (NFFT=512,
  ~198 frames ≈ 26M trig ops). Replaced with a self-contained **radix-2 FFT** (irfft): **730 → 1.8 ms
  (≈400×)**, audio bit-identical (**corr 1.0000000**). This alone cut warm synth **971 → 242 ms**.
- `mish` simplified 8→5 ops (single `exp`); marginal but free.
- **Net: warm synth 971 → 239 ms (4.06×), peak RSS 509 → 493 MB, RTF 0.31 → 0.075, zero quality loss.**
- Attributed breakdown after: phaseA(encoder) ~12 ms, graph-build ~3.6 ms, **decoder+vocos compute
  ~186 ms** (the genuine UNet×3-ODE + 8 ConvNeXt FLOPs), iSTFT ~1.8 ms. The remaining cost is real
  model compute, reducible only by GPU or fewer ODE steps.

### CUDA — now supported (backend-sched refactor)
`PushText` originally used `ggml_graph_compute_with_ctx` (the **CPU-only** legacy compute) with two
6 GB no-data contexts and raw host `->data` writes, so it could *never* run on a GPU backend. It was
**refactored to the backend-agnostic `ggml_backend_sched` path** (`init_compute_ctx` no_alloc graph →
`ggml_set_input` → `ggml_backend_sched_alloc_graph` → flush registered host inputs via
`ggml_backend_tensor_set` → `ggml_backend_sched_graph_compute` → `ggml_backend_tensor_get`). Host-coupled
data (masks, sinusoid time-emb, mu_expanded, the mish/affine scalar consts) became **registered inputs**
(openvoice2 pending-inputs pattern); SnakeBeta `exp(α)`/`exp(−β)` moved **in-graph** (backend-safe).

`matcha-tts` now runs on whatever backend `rs_context` picks. Default stays CPU on the gen1; **opt into
CUDA with `MATCHA_USE_CUDA=1`**. Measured (GB10, warm, RTF for 3.18 s of audio):

| Path | Warm synth | RTF | Notes |
|---|---|---|---|
| old CPU (`graph_compute_with_ctx`) | 239 ms | 0.075 | pre-refactor |
| **new CPU (sched)** | **113 ms** | **0.036** | **2× faster** — drops the 6 GB per-call ctx for sched buffers |
| **new CUDA (sched)** | **36 ms** | **0.011** | + **~57 s one-time JIT** (sm_53 PTX→sm_121 on the GB10) |

Audio correct on every path: CPU-sched vs pre-refactor **corr 0.99999**; CUDA vs CPU **corr 0.996** (the
small decorrelation is CUDA-vs-CPU float reduction order amplified by the phase-sensitive iSTFT).

**Refined Nano-gen1 verdict.** The earlier "CUDA is a dead-end" call was about *the Nano specifically*,
and that nuance stands: **CUDA graphs are gated on `cc ≥ 800` (Ampere)** in this ggml, and the Nano is
**sm_53**, so on the Nano every kernel launches individually — a launch-bound CFM graph on weak Maxwell
may still lose to the A57 CPU (untested on device). The GB10 36 ms shows the *code* works and is fast on
a big GPU, but the GB10 is Blackwell, not the Nano. Honest split:
- **The port is now CUDA-capable and validated** (corr 0.996) — no longer CPU-only.
- **On a real Nano gen1**: no JIT (native sm_53 SASS) so it runs on CUDA immediately, but whether CUDA
  beats the now-113 ms CPU path is an open hardware question.
- **On Ampere+ (Orin)**: cuda-graphs *do* engage (cc ≥ 800) — that's where ggml-CUDA Matcha should win.
- So `rs_context` keeps CPU as the gen1 default (safe, fast, no JIT), with CUDA one env-var away.

### Future work (documented, not done — would risk the validated numerics for ~10–20 ms)
- Reduce the 48 `ggml_cont` + 21 `ggml_transpose` at the conv `[T,C]` ↔ transformer `[C,T]` layout
  seams (each is a real memory copy; runtime count is high since they're inside the 18× resnet / 18×
  transformer loops). Inherent part is unavoidable (convs need `ne0=T`, LayerNorm/matmul prefer
  `ne0=C`); the redundant `cont(transpose(cont(...)))` chains are the removable part.

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
1. **onnx→gguf converter** ✅ DONE — `scripts/convert_matcha_onnx_to_gguf.py`. Extracts **all 474
   learned tensors** into one gguf + model-card hparams; round-trip-validated (gguf == onnx, PASS).
   Two ONNX-export quirks the converter handles:
   - **Anonymous Linear weights** (`onnx::MatMul_*` initializers) → traced to semantic names from
     the consuming node (97 acoustic + 17 vocos, e.g. `voc.blocks.0.pw1.weight`).
   - **Folded learned Constants** — the acoustic encoder's **LayerNorm gamma/beta and relative-
     position embeddings are baked into multi-element `Constant` nodes**, not initializers (128
     such constants in the graph). The converter harvests the 89 that are genuine learned params
     (`model.const.<consumer-path>`), so no weight is lost. (This is why an initializer-only scan
     shows the encoder having attention/FFN convs but "no norms".)
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

## Status summary (honest)
**Done & numerically validated:**
- onnx→gguf converter — all **474 learned tensors** (incl. 89 folded-Constant encoder norms/rel-pos),
  round-trip exact.
- **Full Vocos vocoder, mel → waveform** — ConvNeXt network (rel 3e-4 vs ONNX) + iSTFT tail (corr 1.0).

**Acoustic model (text → mel) — text-encoder DONE, CFM decoder remains:**
- **Encoder front** ✅ VALIDATED (`matcha_encoder_validate.cpp`, rel 2e-5): embedding ×√192 →
  ConvReLUNorm prenet.
- **Transformer encoder layer** ✅ VALIDATED (`matcha_attn_validate.cpp`, rel 9e-5): RoPE attention +
  FFN + both post-norms. The decisive detail (caught by ONNX intermediate shapes `(12,1,2,48)` /
  `(1,2,12,96)`): **n_heads=2, head_dim=96, PARTIAL rotary** — only the first 48 of 96 dims are
  rotated (rotate_half 24/24), last 48 pass through. cos/sin are **baked Constant tables** (no
  theta-match needed). SDPA uses the `openvoice2.cpp` pattern. Attention alone matched **rel 0
  (exact)**. The **6 layers are identical structure** → the encoder transformer is solved (chain the
  validated block ×6).
- **Full text encoder** ✅ VALIDATED (`matcha_full_encoder_validate.cpp`): emb → prenet → **6× RoPE
  transformer** → `proj_m` (μ, rel 1e-4) + `proj_w` (durations logw, rel 2e-5). Confirmed the RoPE
  cos/sin tables are **shared across all 6 layers** (reused the layer-0 baked table → exact). So the
  entire **text → (μ, durations)** path is done.
- **Length regulator** — expand μ by ceil(exp(logw)·length_scale) durations (control logic). TODO.
- **CFM decoder** (**2977 nodes — the remaining bulk**) — fully mapped, sub-blocks below. A 1-D UNet
  `estimator(x_t, μ, t)` → vector field, solved with **3 Euler ODE steps** from `RandomNormalLike`
  seed × `noise_scale`. Structure (hidden 256):
  - **time embedding**: SinusoidalPosEmb(t) → `time_mlp` linear_1 (→1024) → SiLU (Sigmoid·Mul) →
    linear_2 (1024→1024).
  - **ResnetBlock1D** (`block1`/`block2` = Conv1d k3 → **GroupNorm** (done as reshape→InstanceNorm→
    per-group affine) → **Mish** `x·tanh(softplus(x))`); time cond via `mlp` (256←1024) added between
    blocks; `res_conv` k1 skip. Input to first = concat(x_t, μ) → 160 ch.
  - **BasicTransformerBlock** (`norm1`→ self-attn `attn1` to_q/k/v/out → `norm3`→ FF: `net.0` =
    SnakeBeta + proj(256→1024), `net.2` = linear(1024→256)). **SnakeBeta confirmed (numpy)**:
    `x + sin²(exp(log_alpha)·x) / exp(log_beta)` (alpha/beta stored in log space). ggml validator
    `matcha_snake_validate.cpp` (formula correct; a broadcast-div plumbing segfault on the old gguf-
    container ggml CPU build remains to resolve — low-risk, the math is verified).
  - **down/up sample**: downsample = Conv k3 stride 2; upsample = **ConvTranspose**. `final_block`
    (Conv→GroupNorm→Mish) + `final_proj` (256→80).
  - **GroupNorm**: 8 groups, done as reshape `[B,8,-1]` → InstanceNorm → reshape back → affine
    (scale·x+bias). The scale/bias are folded Constants (now captured distinctly — see converter).
  - **time_mlp**: input is a baked `[1,160]` sinusoidal embedding (the 3 ODE-step values are folded
    constants since num_ode_steps is fixed at export) → linear_1 (160→1024) → **SiLU** → linear_2.
  - **ODE**: the 3 Euler steps are **unrolled** in the graph (≈3× the estimator → 2977 nodes); the 3
    copies **share weights via Identity passthroughs**.

  **Converter completeness (fixed this pass):** the decoder exposed two converter bugs — (1) attn
  to_q/k/v weights were first consumed by Identity (the shared ODE copies) → misnamed; (2) GroupNorm
  scale+bias collided to one name → 13 silently dropped. Both fixed (Identity-chain resolution +
  input-index disambiguation); the gguf now carries all 474 weights, 0 collisions, round-trip PASS.
  So **the gguf is now complete for build_cfm**; the remaining work is implementing + staged-validating
  the UNet blocks (ResnetBlock1D, BasicTransformerBlock) and the ODE loop inside `MatchaModel::build_cfm`.

  **Key de-risking (this pass):**
  - **`arch/cosyvoice3_flow.cpp` is a reference CFM/flow-matching decoder in ggml** — it already has
    `mish()`, `linear`, `conv1d`, `conv1d_grouped`, a sinusoidal time-embedding table, `BuildDiTBlock`
    (time-conditioned transformer), and `BuildCFMStepGraph`. build_cfm should reuse these patterns.
  - **ggml has all needed primitives in the full build**: `ggml_softplus`/`mish`/`ggml_log`/`ggml_sin`
    exist there (the earlier standalone-validator harness was the limited path; the arch compiles
    against the same ggml as cosyvoice3_flow, so Mish/Snake are not blockers in matcha.cpp).
  - **ResnetBlock1D math validated in numpy** (`scripts/gen_resnet_ref.py`): Conv→GroupNorm8→Mish,
    time-cond add (Linear∘Mish), block2, residual res_conv. GroupNorm8 = reshape `[8,C/8·T]` →
    normalize per group → `·γ+β`; γ/β are the folded `block.block.1_2.{weight,bias}` `[256]`.
  - **ResnetBlock1D validated in ggml** (`tools/matcha_resnet_validate.cpp`, rel 2.8e-4) — the
    core/most-repeated decoder unit. GroupNorm8 in ggml: `x[T,C] → reshape [T·C/8, 8] → ggml_norm →
    reshape → ·γ+β`. Mish via `x·tanh(log(1+exp(x)))` (this ggml has no `ggml_softplus`).

  **CFM-decoder block specs (both core units now characterised):**
  - **ResnetBlock1D** ✅ ggml-validated (above). Channel dims per UNet stage: down0 160→256,
    down1 256→256, mid 256→256, up 512→256 (512 = 256 skip-concat), final 256→256, final_proj 256→80.
  - **BasicTransformerBlock**: `x += attn1(norm1(x))`; `x += ff(norm3(x))`. norm1/norm3 = LayerNorm
    [256]. attn1 = self-attn, **2 heads × head_dim 64** (inner 128): to_q/k/v 256→128, SDPA
    1/√64, to_out 128→256 — same SDPA pattern as the (validated) encoder attention. ff =
    `net.2(SnakeBeta(net.0.proj(x)))`: proj 256→1024 → SnakeBeta (numpy-validated:
    `x+sin²(e^α x)/e^β`) → linear 1024→256.

  **Remaining build_cfm assembly** (bounded; both blocks + all primitives validated):
  sinusoidal time-emb + time_mlp (Linear→SiLU→Linear); length regulator (expand μ by
  ceil(e^logw·length_scale)); UNet down(×2)/mid(×2)/up(×2 w/ skip-concat) + final; 3-step Euler ODE
  (unrolled, weights shared). Then wire into `matcha.cpp::build_cfm` and validate end-to-end vs ONNX
  mel (noise_scale=0). Estimated several more build-debug cycles — assembly, not new research.

This is genuinely a ~2k-line arch (≈ `openvoice2.cpp`), with the rel-pos attention, the
Constant-folded norms, and the 3-step CFM ODE solver as the intricate pieces. It is **multi-session
work** — not completable in one pass without large amounts of unvalidated code. The proven method
(per-stage ONNX-intermediate extraction → ggml compare → fix → next stage) carries directly over;
the next concrete stage is the embedding+prenet front, then the rel-pos transformer block, each
validated before moving on.

For a **working Matcha on the Nano today**, the validated path is **sherpa-onnx on the cuDNN-free
ORT 1.11** (681 MB / 139 ms, output matches the HF reference) — see edge-speech-gpu-bench. This
ggml port is the lighter-RAM alternative, in progress.
