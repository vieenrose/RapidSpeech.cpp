# FunASR-Nano 使用说明（离线 ASR，可选 LLM 二次解码）

FunASR-Nano 是轻量离线 ASR：SANM 编码器 + CTC 快速解码头，另带一个 Qwen3 LLM
解码器做二次改写（rescore）。可以只跑 CTC（快），也可以叠加 LLM（更准）。

---

## 1. 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-asr-offline（离线）、build/rs-asr-vad-online（流式切段 + 二次解码）
```

## 2. 转换 GGUF

```bash
# 完整模型（CTC + LLM）
python scripts/convert_hf_to_gguf.py \
    --model-dir /path/to/FunASRNano \
    --output    funasr-nano-fp16.gguf --out-type f16

# 只要 CTC（更小，跳过 LLM 权重）
python scripts/convert_hf_to_gguf.py \
    --model-dir /path/to/FunASRNano \
    --output    funasr-nano-ctc.gguf --out-type f16 --without_llm
```

转换器会同时写入 CTC 词表（`ctc.tokenizer.ggml.tokens`）和 LLM 词表。

## 3. 离线使用

```bash
# CTC 快速解码（默认）
./build/rs-asr-offline -m funasr-nano-fp16.gguf -w audio.wav -t 4 --gpu true

# 加 VAD 切段 / CAMPPlus 说话人分离，同 SenseVoice
./build/rs-asr-offline -m funasr-nano-fp16.gguf -v silero_vad_v6.gguf \
    -s campplus.gguf -w audio.wav
```

## 4. 二次解码（CTC → LLM rescore）

流式切段 CLI `rs-asr-vad-online` 支持 2-pass：先 CTC 出快结果，静音端点后再用 LLM 改写：

```bash
./build/rs-asr-vad-online -m funasr-nano-fp16.gguf -v silero_vad_v6.gguf \
    -w audio.wav --two-pass --ctc-precheck
# 或麦克风：--mic
```

- `--two-pass`：CTC 快速 partial + LLM 终稿 rescore。
- `--ctc-precheck`：LLM 前先用 CTC 判静音，跳过空段，减少幻觉。

## 5. 量化

```bash
./build/rs-quantize funasr-nano-fp16.gguf funasr-nano-q4_k_m.gguf q4_k_m
```
