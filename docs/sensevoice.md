# SenseVoice-Small 使用说明（离线 ASR）

SenseVoice-Small 是多语言离线 ASR（SANM 编码器 + CTC），中英日韩粤，带情感/事件标签。
在本项目中通过 `rs-asr-offline` 驱动，可选叠加 VAD 切段与 CAMPPlus 说话人聚类。

---

## 1. 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-asr-offline
```

## 2. 转换 GGUF

从 FunASR/ModelScope 下载 SenseVoiceSmall（含 `config.yaml`、`model.pt`、`am.mvn`）：

```bash
python scripts/convert_hf_to_gguf.py \
    --model-dir /path/to/SenseVoiceSmall \
    --output    sensevoice-small-f16.gguf \
    --out-type  f16
```

- CMVN（`am.mvn`）、词表和前端参数（HAMMING 窗、LFR m=7/n=6）都写进 GGUF。
- `--out-type f32` 保留全精度；CTC 输出投影始终留 f32。

## 3. 使用

```bash
# 整段识别（不切段）
./build/rs-asr-offline -m sensevoice-small-f16.gguf -w audio.wav -t 4 --gpu true

# VAD 切段 + 时间戳（Silero 或 FireRedVAD，按 GGUF arch 自动识别）
./build/rs-asr-offline -m sensevoice-small-f16.gguf -v silero_vad_v6.gguf -w audio.wav

# 再加 CAMPPlus 说话人分离，每段带说话人标签
./build/rs-asr-offline -m sensevoice-small-f16.gguf -v silero_vad_v6.gguf \
    -s campplus.gguf -w audio.wav
```

常用参数见 `rs-asr-offline -h`（`--vad-threshold`、`--silence-ms`、`--spk-cluster`
spectral/ahc、`--num-speakers` 等）。

## 4. 量化

```bash
./build/rs-quantize sensevoice-small-f16.gguf sensevoice-small-q4_k_m.gguf q4_k_m
```

编码器主体可量化到 Q3_K 以上；过低精度（q2_k_m）会掉字，见项目量化说明。
