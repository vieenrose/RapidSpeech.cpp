# matcha-server — persistent Matcha-TTS demo (Jetson Nano gen1 / sm_53)

**Status: demo / reference implementation.** This is the minimal, transport-agnostic
shape of the warm-persistent Matcha-TTS pattern (a stdin/stdout line protocol). The
production integration (iPhO 2.0) wraps this same engine as a **LiveKit Agents TTS
plugin** so it can stream into a room/SIP leg; this demo is the engine-loop reference
that plugin is built from.

Loads the matcha gguf and the CUDA backend **once**, JITs the sm_53 kernels once,
then serves synthesis requests in a loop. The launch-bound first-graph cost (PTX-JIT
+ first-graph build, ~1.5 s on Maxwell) is paid a single time, so every subsequent
request runs at the warm RTF instead of the cold single-shot RTF.

On a real Jetson Nano gen1 (Maxwell sm_53, CUDA 10.2, ggml-CUDA):

| mode | RTF |
|------|-----|
| cold single-shot (`matcha_e2e_test`) | ~1.0 |
| **warm (this service)** | **~0.18–0.22** |

## Frontend note

RapidSpeech now ships a built-in Matcha frontend (`frontend/matcha_frontend.{h,cpp}`,
used by `rs-tts-offline` via `MATCHA_TOKENS` + `MATCHA_LEXICON`) that turns **mixed
Chinese/English** text into phoneme IDs from the model's `tokens.txt` + `lexicon.txt`
(zh-TW and zh, no conversion; English via espeak-ng + the diphthong replacement table,
`-DRS_MATCHA_ESPEAK=ON`). Punctuation maps to tokens and synthesis is split per clause.

This **demo service** is intentionally lower-level: it takes pre-computed phoneme-ID
files, not raw text, so it stays independent of the frontend (drive it from the built-in
frontend, from sherpa-onnx's matcha frontend `--debug=1`, or any other source).

## Protocol

Line-oriented on stdin; one reply line on stdout per request:

```
<ids_file>  <out_wav>  [length_scale]
```

- `ids_file` — little-endian int32 phoneme IDs
- `out_wav` — 8 kHz mono PCM16 output path
- `length_scale` — optional float (default 1.0)

Reply: `OK <out_wav> <nsamples> <synth_ms>` or `ERR <reason>`. EOF ends the service.

## Build

CUDA build for the Nano (cross-compiled on x86):

```sh
scripts/build_jetson_nano_gen1.sh          # produces build-nano/matcha-server
```

Run on the Nano with the GPU backend forced on (matcha defaults to CPU on sm_53):

```sh
MATCHA_USE_CUDA=1 ./matcha-server matcha8k.gguf
```
