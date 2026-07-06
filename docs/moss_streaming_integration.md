# MOSS-TTS-Nano — full streaming integration (RapidSpeech.cpp + codec.cpp)

Status: **all primitives verified**; remaining work is assembly. This is the exact wiring.

## Verified building blocks (this branch)
| piece | tool (verified) | to move into arch |
|---|---|---|
| Global GPT-2 forward (prefill) | `tools/moss_parity.cpp` (MSE 1.4e-6 vs ONNX) | `build_global()` |
| Global decode w/ KV cache | `tools/moss_kv.cpp` (MSE 1.4e-6 vs ONNX decode_step) | KV-cached `decode_step()` |
| Local decoder (16 codebooks) | `tools/moss_local_parity.cpp` (14/16 argmax Q8) | `local_frame()` |
| Q8 GGUF load | `tools/moss_load_test.cpp` (389 tensors) | `Load()` (done) |
| Custom matvec kernel | `tools/moss_cuda/moss_matvec_q8_sm53.cu` (2.6×) | ggml-CUDA `mul_mat_vec_q` swap |

## Codec = codec.cpp (do NOT reimplement)
- `mybigday/codec.cpp` runs the `moss_audio` codec in ggml (CUDA), gguf from `hans00/MOSS-TTS-Nano-GGUF`.
- Use for: **encode**(zh-TW ref audio → 16-channel ref codes) and **decode**(codes → 48 kHz, streaming).
- Codec round-trip already verified faithful (corr 0.95–0.998).

## Generation flow (from modeling_moss_tts_nano.py, traced + verified)
```
prompt (17-wide/pos): col0 = text token, col1..16 = 16 audio codebook tokens
  build_voice_clone_request_rows(ref_codes, text_token_ids)   # user prefix + ref audio codes + assistant prefix
prompt embed per pos = token_embd[text] + Σ_k audio_embd_k[code_k]   (pad=1024 → 0)
prefill(prompt_embed) [KV cache] -> hidden[last]
loop frame:
  # dual-track: predict a text token first
  local_seq = [hidden]                              # pos 0
  h0 = local_transformer(local_seq)[-1]; text_tok = argmax(text_lm_head(h0))
  if text_tok == audio_end_token_id: break
  cur = wte(text_tok)                               # global token_embd
  frame = []
  for k in 0..15:
    local_seq.append(cur)                           # pos k+1
    hk = local_transformer(local_seq)[-1]
    code_k = sample(audio_lm_heads[k](hk))          # greedy/top-k (audio_temp 0.8/top_p 0.95/top_k 25/rep_pen 1.2)
    frame.append(code_k); cur = audio_embd_k(code_k)
  # feed frame back to global
  next17 = [text_tok, frame[0..15]]
  hidden = decode_step(embed(next17))               # KV-cached global step
  emit frame -> codec.cpp streaming decode -> 80ms audio chunk
```

## Wiring into ISpeechModel (arch/moss_tts_nano.cpp)
- `PushReferenceAudio(samples)` → codec.cpp encode → store `ref_codes` in MossState.
- `PushText(text)` → sentencepiece tokenize (tokenizer.model) → build 17-wide prompt rows (+ref_codes) → prefill → set state ready. For long text: clause-split (see PrimeTTS `_split_clauses`), one prompt per clause.
- `GetAudioOutput()` → run the frame loop above, each frame → codec.cpp decode → return 80 ms PCM chunk. Streaming: emit as produced.
- KV cache: persistent per-layer buffers (see moss_kv.cpp). Both global (12L) and local (1L, own small KV) caches.

## Build order (Nano)
1. Swap custom matvec kernel into ggml-CUDA `mul_mat_vec_q` (RTF 0.91→~0.35).
2. Link codec.cpp (or vendor its codec op) for encode/decode.
3. sentencepiece for tokenizer.
4. Build `moss-tts-nano` arch on Nano w/ the patched ggml-CUDA (sm_53).
5. Gate: zh-TW CER (X-ASR), RTF<1, TTFF (~100-200ms), streaming chunk cadence.

## Perf targets (measured this session, Jetson Nano gen1 GPU)
- Q8 full-12L: RTF 0.86 (stock) → ~0.35 (custom kernel). 4L-student: 0.50 → ~0.21. Floor ~0.12.
- zh-TW accent: free via zh-TW reference clip (台湾腔), no retrain.
- Weights: `Luigi/MOSS-TTS-Nano-100M-GGUF` (Q8), codec: `hans00/MOSS-TTS-Nano-GGUF`.
