# OpenVoice2 / MeloTTS 使用说明（多语言 TTS）

OpenVoice2 是基于 MeloTTS 的多语言 TTS（当前 GGUF 支持中/英）。基础模型合成语音，
可选的 tone-color converter 做音色转换。

---

## 1. 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-tts-offline
```

## 2. 转换 GGUF

从 MeloTTS 权重转换，每种语言一个基础 GGUF：

```bash
python scripts/convert_openvoice2_v2.py \
    --lang EN --model-dir /path/to/MeloTTS-English \
    --output openvoice2-base-en.gguf --f16
python scripts/convert_openvoice2_v2.py \
    --lang ZH --model-dir /path/to/MeloTTS-Chinese \
    --output openvoice2-base-zh.gguf --f16
```

## 3. 基本合成

```bash
# 英文
./build/rs-tts-offline -m openvoice2-base-en.gguf \
    -t "Hello, welcome to RapidSpeech!" --lang English -o out_en.wav -t 4

# 中文（需额外提供 1024 维 ZH BERT GGUF）
./build/rs-tts-offline -m openvoice2-base-zh.gguf \
    -t "今天天气真好。" --lang Chinese --bert zh-bert.gguf -o out_zh.wav
```

- `--lang` 需用完整语言名（`English` / `Chinese`），传 `zh`/`en` 会 fallback 到默认
  说话人导致近似静音。
- 中文分支需要 `--bert` 提供 1024 维 ZH BERT 特征。

## 4. 音色克隆（可选）

用 `--ref` 提供参考音频克隆音色（需配套 speech tokenizer + CAM++）：

```bash
./build/rs-tts-offline -m openvoice2-base-zh.gguf --bert zh-bert.gguf \
    --ref speaker.wav --speech-tokenizer spk-tok.gguf --campplus campplus.gguf \
    -t "克隆这个音色说话。" --lang Chinese -o clone.wav
```

其余可调项见 `rs-tts-offline -h`（`--length-scale` 语速、`--peak`/`--gain` 归一化、
`--seed` 等）。
