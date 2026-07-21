# MOSS-Transcribe-Diarize — origin and licence of this directory

Every `.cpp`/`.hpp` in this directory (and `include/`, `third_party/dr_wav.h`)
is **vendored, unmodified**, from:

> **moss-transcribe.cpp** — https://github.com/localai-org/moss-transcribe.cpp
> Licence: MIT (see `LICENSE-moss-transcribe-cpp.txt`, reproduced verbatim)
> Upstream revision: see `UPSTREAM_REV` (`190a569c13b4b247450f2fb3b2a431244e84833e`)

It is a from-scratch C++17 ggml port of
[OpenMOSS MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
(the model and its reference pipeline are Apache-2.0).

## Why vendored unmodified

The whole point of this stage is byte-level identity with the reference PyTorch
pipeline. Every local edit is a chance to break that silently, and it also makes
re-syncing with upstream harder. So the rule for this directory is:

**Do not edit these files.** Adaptation belongs in the RapidSpeech-side wrapper,
not in here. If a change genuinely must happen upstream-side, record it here with
the reason, and re-run the parity gate afterwards.

Local deltas so far: **none.**

## Parity status in THIS tree

Validated 2026-07-21 against the genuine PyTorch pipeline, non-windowed, no
post-processing, on both golden clips (`golden_zh_5min.wav` 318 s,
`golden_en_5min.wav` 300 s):

* PyTorch **f32** vs this code at **f16 GGUF** — byte-identical, CPU and CUDA,
  across transcript text, every timestamp, and every `[Sxx]` speaker tag.
* Verified to still hold when built against **RapidSpeech's** ggml revision
  (`57ea0bc`, 0.11.1) rather than upstream's own (`eced84c`, 0.15.3).

The reference dtype **must be f32**. bf16 — which the model authors' README uses
on CUDA — has an 8-bit mantissa against f16's 10, so a bf16 reference is *less*
precise than the f16 port and diverges at near-ties. PyTorch-f16 diverges too
(different accumulation order). Reproduce with `scripts/80_official_reference.py
--dtype f32` in the distil-vibevoice-asr repo.

## Also consulted (not copied from)

Two other MIT implementations exist and are worth reading when a detail is
ambiguous; where independent ports disagree is exactly where the reference
behaviour needs checking:

* `handy-computer/transcribe.cpp` — MIT — `src/arch/moss/{encoder,decoder,diarize}`
* `CrispStrobe/CrispASR` — MIT — `crispasr_chunk_context_gate.h`, forced alignment

## Build flags are part of the parity contract

`GGML_LLAMAFILE` **must be ON** (CMakeLists forces it when `RS_BUILD_MOSS_TD`).
Upstream forces it; RapidSpeech defaults it OFF. Built with it OFF, the English
golden clip decoded `[291.22]` where the reference has `[291.26]` — 1 timestamp
in 92, with the transcript text and all other markers byte-identical. Different
sgemm kernels, different accumulation order, one near-tie flipped.

That is the general shape of the risk here: parity failures from numerics are
not loud, they are a single token deep in a 3 kB transcript. Diff the whole
output; never eyeball the first few lines.

## Verified in this tree (RapidSpeech ggml `57ea0bc`, `rs-moss-td` binary)

| clip | CUDA | CPU (8 threads) |
|---|---|---|
| `golden_zh_5min.wav` | identical | identical |
| `golden_en_5min.wav` | identical | identical |

all against PyTorch **f32**.
