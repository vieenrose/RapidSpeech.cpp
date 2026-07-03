# VAD 使用说明（Silero VAD / FireRedVAD）

VAD（语音活动检测）用于把长音频切成语音段，供 ASR 逐段识别。本项目支持两种 VAD，
GGUF 通用，`rs-asr-offline` / `rs-asr-vad-online` 会按 `general.architecture` 自动识别。

| 模型 | arch | 特点 |
|------|------|------|
| Silero VAD | `silero-vad` | 通用、轻量，实时端点触发 |
| FireRedVAD | `firered-vad` | DFSMN，流式，段间合并 |

---

## 1. 转换 GGUF

```bash
# Silero
python scripts/convert_silero_to_gguf.py --model-dir /path/to/silero_vad --output silero_vad_v6.gguf
# FireRedVAD（含 model.pth.tar + cmvn.ark）
python scripts/convert_fireredvad_to_gguf.py --model-dir /path/to/FireRedVAD --output firered-vad.gguf
```

## 2. 与 ASR 联用

VAD 作为 `-v` 传给 ASR CLI，切段 + 时间戳自动完成：

```bash
./build/rs-asr-offline -m asr.gguf -v silero_vad_v6.gguf -w audio.wav
./build/rs-asr-offline -m asr.gguf -v firered-vad.gguf   -w audio.wav
```

常用切段参数（`rs-asr-offline -h`）：

| 参数 | 说明 |
|------|------|
| `--vad-threshold <f>` | 语音概率阈值（默认 0.5） |
| `--silence-ms <ms>` | 切段静音时长；Silero=端点触发，FireRed=相邻段合并间隙（默认 600） |
| `--speech-pad-ms <ms>` | 语音起点前的预留 padding（默认 200） |
| `--prob-smooth <f>` | VAD 概率 EMA 平滑，仅 Silero（默认 0.3） |

## 3. C API

VAD 也可通过 `rs_vad_*` C API 单独使用（`rs_vad_init_from_file` / detect），见
`include/rapidspeech.h`。
