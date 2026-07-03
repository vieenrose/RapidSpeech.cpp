# IndexTTS-2 使用说明（音色克隆 + 情感控制）

IndexTTS-2 是带**情感控制**的零样本语音克隆 TTS。本文介绍如何用 `rs-tts-offline`
驱动它的音色克隆与 4 种情感控制模式。

---

## 1. 准备

### 1.1 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-tts-offline
```

### 1.2 模型文件

| 文件 | 用途 | 必需 |
|------|------|------|
| `indextts2.gguf` | 主模型（GPT-AR + S2Mel + W2V-BERT + 语义编码器 + CAMPPlus + BPE） | 是 |
| `indextts2-bigvgan.gguf` | BigVGAN-v2 声码器（mel→波形），单独 GGUF | 是 |
| `indextts2-qwen-emo-f16.gguf` | Qwen3-0.6B 情感分类器 | 仅模式 3 |

转换脚本（PyTorch → GGUF）：

```bash
# 主模型 + BigVGAN
python scripts/convert_indextts2_to_gguf.py \
    --indextts2-dir <checkpoints> --bigvgan-dir <bigvgan_v2_22khz> \
    --maskgct-dir <maskgct> --campplus-bin <campplus_cn_common.bin> \
    --w2v-bert-dir <w2v-bert-2.0> --output-dir <out> --dtype f16

# Qwen 情感模型（仅模式 3 需要）
python scripts/convert_qwen3_emotion_to_gguf.py \
    --model-dir <checkpoints>/qwen0.6bemo4-merge \
    --output <out>/indextts2-qwen-emo-f16.gguf --out-type f16
```

---

## 2. 基本用法（音色克隆）

最少需要：主模型 + BigVGAN + 一段参考音频（提供音色）。

```bash
./build/rs-tts-offline \
    -m indextts2.gguf \
    --bigvgan-gguf indextts2-bigvgan.gguf \
    --ref speaker.wav \
    -t "今天天气真好，我们一起去公园散步吧。" \
    -o out.wav --seed 42
```

- `--ref`：**音色**参考音频（任意采样率，内部重采样到 16k/22.05k）。决定克隆出的音色。
- 输出 24kHz（实际 22.05kHz）WAV。
- 不加任何 `--emo-*` 时即**模式 0**：情感跟随音色参考音频本身。

---

## 3. 情感控制：4 种模式

情感与音色是**两条独立通路**。音色始终来自 `--ref`；情感由下列模式之一决定。
模式可由所给参数自动判定，也可用 `--emo-mode <0..3>` 强制。

`--emo-alpha`（情感权重，0~1，默认 1.0；官方 webui 默认 0.65）控制情感强度。

### 模式 0 — 跟随音色（默认）

不传任何情感参数。情感取自 `--ref` 音频自身。

```bash
./build/rs-tts-offline -m indextts2.gguf --bigvgan-gguf indextts2-bigvgan.gguf \
    --ref speaker.wav -t "文本" -o out.wav
```

### 模式 1 — 独立情感参考音频

用**另一段音频**的情绪，音色仍来自 `--ref`。

```bash
./build/rs-tts-offline -m indextts2.gguf --bigvgan-gguf indextts2-bigvgan.gguf \
    --ref speaker.wav \
    --emo-audio angry_voice.wav \
    --emo-alpha 0.8 \
    -t "你怎么能这样对我！" -o out.wav
```

### 模式 2 — 8 维情感向量

手动给 8 个情感的权重，顺序固定：

```
happy(高兴), angry(愤怒), sad(悲伤), afraid(恐惧),
disgusted(反感), melancholic(低落), surprised(惊讶), calm(自然)
```

```bash
# 愤怒为主
./build/rs-tts-offline -m indextts2.gguf --bigvgan-gguf indextts2-bigvgan.gguf \
    --ref speaker.wav \
    --emo-vector "0,1.0,0,0,0,0,0,0" \
    --emo-alpha 0.8 \
    -t "你怎么能这样对我" -o out.wav
```

- 内部会对向量做 bias 去强调 + 总和≤0.8 归一化（与官方 `normalize_emo_vec` 一致），
  再按 `--emo-alpha` 缩放。
- `--emo-random`：在情感原型库里随机选（默认按音色 style 余弦匹配最相似原型）。

### 模式 3 — 情感文本（Qwen 情感模型）

用一句中文描述情绪，由 Qwen3 情感分类器转成 8 维向量。**需要 `--qwen-emo-gguf`**。

```bash
./build/rs-tts-offline -m indextts2.gguf --bigvgan-gguf indextts2-bigvgan.gguf \
    --qwen-emo-gguf indextts2-qwen-emo-f16.gguf \
    --ref speaker.wav \
    --emo-text "非常愤怒，怒火中烧" \
    --emo-alpha 0.8 \
    -t "你怎么能这样对我" -o out.wav
```

- `--emo-text` 为空时回退用正文做情感判断。
- 实测：`愤怒`→angry 0.85、`开心`→happy 0.95、`悲伤`→sad 0.99、`恐惧`→afraid 0.90。
- 若未提供 `--qwen-emo-gguf`，会打印 warning 并优雅回退到音色情感（模式 0），不崩溃。

---

## 4. CLI 参数速查（情感相关）

| 参数 | 说明 |
|------|------|
| `--ref <wav>` | 音色参考音频（克隆音色，必需） |
| `--bigvgan-gguf <path>` | BigVGAN 声码器 GGUF（必需） |
| `--emo-audio <wav>` | 模式 1：独立情感参考音频 |
| `--emo-vector "a,b,..,h"` | 模式 2：8 维情感向量 |
| `--emo-text <text>` | 模式 3：情感描述文本 |
| `--qwen-emo-gguf <path>` | 模式 3：Qwen 情感模型 GGUF |
| `--emo-alpha <f>` | 情感权重 0~1（默认 1.0，webui 默认 0.65） |
| `--emo-random` | 模式 2：随机选情感原型 |
| `--emo-mode <0..3>` | 强制指定模式（默认按所给参数自动判定） |
| `--emo-test-dir <dir>` | 调试：dump `base_vec/emo_vec_src/emovec/emovec_mat/style/wvec` |
| `--seed <n>` | 采样随机种子（默认 42） |

模式自动判定优先级：`--emo-text` → 3，`--emo-vector` → 2，`--emo-audio` → 1，否则 0。

---

## 5. C API

```c
#include "rapidspeech.h"

rs_set_tts_params(ctx, /*instruct*/NULL, /*language*/NULL, /*seed*/42);
rs_push_reference_audio(ctx, spk_pcm, n, sr);          // 音色

// 选一种情感方式：
rs_push_emotion_audio(ctx, emo_pcm, n, sr);            // 模式 1
float vec[8] = {0,1,0,0,0,0,0,0};
rs_set_emotion(ctx, RS_EMO_FROM_VECTOR, 0.8f, vec, /*random*/false,
               /*apply_bias*/true, NULL);              // 模式 2
rs_set_emotion(ctx, RS_EMO_FROM_TEXT, 0.8f, NULL, false, false,
               "非常愤怒");                             // 模式 3（需先加载 qwen gguf，见下）

rs_push_text(ctx, "要合成的文本");
while (rs_process(ctx) > 0) { float* pcm; int n = rs_get_audio_output(ctx, &pcm); /* ... */ }
```

模式枚举：`RS_EMO_FROM_SPEAKER=0 / _AUDIO=1 / _VECTOR=2 / _TEXT=3`。

---

## 6. 环境变量

| 变量 | 作用 |
|------|------|
| `RS_INDEXTTS2_BIGVGAN_GGUF` | BigVGAN GGUF 路径（等价 `--bigvgan-gguf`） |
| `RS_INDEXTTS2_QWEN_EMO_GGUF` | Qwen 情感 GGUF 路径（等价 `--qwen-emo-gguf`，懒加载） |
| `RS_INDEXTTS2_EMO_TEST_DIR` | dump 情感中间量做对齐验证（等价 `--emo-test-dir`） |
| `RS_INDEXTTS2_DUMP_PROD_DIR` | dump `spk_cond_emb/S_ref` 等生产中间量 |

---

## 7. 验证 / 对齐 PyTorch

- 情感向量路径 `emovec_mat` 对 `feat1.pt`/`feat2.pt` **bit-exact**（SNR 212 dB）。
- Qwen 情感输出与 PyTorch `QwenEmotion.inference` 主导情感一致、量级吻合。
- PT 参考导出（在 IndexTTS-2 的 PyTorch 环境运行）：

```bash
python scripts/dump_indextts2_emotion_ref.py \
    --indextts2-dir <checkpoints> --src-root <IndexTTS-2 源码> \
    --spk-audio speaker.wav --emo-text "愤怒" --emo-alpha 0.8 \
    --out-dir /tmp/emoref
# 再用 --emo-test-dir 跑 C++，对比同名 base_vec/emo_vec_src/emovec/emovec_mat
```

---

## 8. 常见问题

- **没声音 / "BigVGAN not loaded"**：忘了 `--bigvgan-gguf`（或对应环境变量）。
- **模式 3 不生效**：缺 `--qwen-emo-gguf`；会回退到模式 0 并打印 warning。
- **情感太弱/太强**：调 `--emo-alpha`（0.6~0.8 通常自然，1.0 最强）。
- **想要可复现结果**：固定 `--seed`。

---

## 9. 性能 / 实时率（RTF）

macOS 默认走**融合 Metal BigVGAN 声码器**（单 command buffer，绕过 ggml 图的 ~467 个
CPU/Metal 分片；卷积用 im2col + MPSMatrixMultiplication GEMM），声码器从 ~54s 降到
**~0.8s**，数值与 PyTorch 一致（37 dB）。

调速手段：
- `--s2mel-steps <n>`：CFM 扩散步数，默认 6。**6 步听感基本无损，3-4 步略有损失但可接受**
  （越低越快）；25 步为上游默认（最稳但最慢）。
- `RS_INDEXTTS2_BIGVGAN_GGML=1`：强制回退旧 ggml 声码器路径（排错用，很慢）。

**音量**：IndexTTS-2/BigVGAN 原始输出幅度偏低（与 PyTorch 一致，rms~0.008）。CLI 默认对输出
做**峰值归一化**到 0.95（约 10× 增益）使其正常响度；相关选项：
- `--peak <v>`：峰值归一化目标（默认 0.95，`0` 关闭）
- `--gain <x>`：额外线性增益（想更响，如 `--gain 1.3`）
- `--no-normalize`：保留原始幅度（数值验证/自行后处理时用）

现状（M1，2.79s 音频）：整体 RTF 约 2.9（3步）～3.7（6步），原始 ~17，约 **4.5-6× 提升**。
各阶段：AR ~2.2s（去 KV cont）/ 条件编码 ~1s / S2Mel(3步) ~1.5s / BigVGAN ~0.8s（im2col+MPS
GEMM）+ 一次性 Metal 内核编译 ~数秒（常驻进程摊销后稳态更快，3 步可低于 RTF 3）。

> 说明：BigVGAN 已优化到位（54s→0.8s）。AR/S2Mel 是 kernel-launch/compute-bound，要再降需写
> 融合 Metal GPT/DiT 内核（大工程）；M1 上此规模模型 RTF<1 不现实，但 3 步 + 常驻服务可到 ~2-3。
