# CAMPPlus 使用说明（说话人识别 / 声纹）

CAMPPlus 提取说话人嵌入（声纹向量），用于说话人确认（1:1 比对）和说话人分离
（ASR 每段打说话人标签）。

---

## 1. 转换 GGUF

```bash
# 从本地 PyTorch 权重
python scripts/convert_campplus_to_gguf.py --pt-file campplus_cn_common.bin --output campplus.gguf
# 或从 HuggingFace
python scripts/convert_campplus_to_gguf.py --hf <repo_id> --output campplus.gguf
```

## 2. 声纹注册 + 比对（rs-speaker-id）

```bash
./build/rs-speaker-id --model campplus.gguf \
    --enroll alice=alice.wav --enroll bob=bob.wav \
    --query unknown.wav [--threads N] [--cpu]
```

- 每个 `--enroll NAME=PATH` 注册一条参考声纹（任意采样率，内部重采样到 16k 单声道）。
- query 音频与所有注册项算余弦相似度并打印，最佳匹配高亮。
- `--dump-emb FILE` 可导出嵌入向量。

## 3. 与 ASR 联用做说话人分离

把 CAMPPlus 作为 `-s` 传给 `rs-asr-offline`，每个语音段自动打说话人标签：

```bash
./build/rs-asr-offline -m asr.gguf -v silero_vad_v6.gguf -s campplus.gguf -w meeting.wav \
    --spk-cluster spectral      # 谱聚类自动估计说话人数（默认）
# 或
    --spk-cluster ahc --spk-threshold 0.5   # 层次聚类，阈值越高说话人越多
    --num-speakers 3            # 指定固定说话人数，跳过自动估计
```
