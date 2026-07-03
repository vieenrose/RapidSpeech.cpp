# X-ASR 使用说明（Zipformer2 流式/非流式 ASR）

X-ASR 是基于 icefall/k2 的**中英文 Zipformer2 transducer** ASR 模型（~160M 参数，
带标点和大小写）。**一份 GGUF 同时支持两种模式**：

- **离线（offline）**：整段音频全上下文前向，精度最高。
- **流式（streaming）**：分 chunk 增量解码，逐层保留左上下文缓存，亚秒级出字。

上游模型：<https://github.com/Gilgamesh-J/X-ASR>。

---

## 1. 准备

### 1.1 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-asr-offline（离线）、build/rs-asr-online（真流式）
```

### 1.2 转换 GGUF

> **重要**：HuggingFace 上的 `streaming_exp/pretrained.pt` 是**旧的无标点
> checkpoint**，与部署 ONNX 的权重不同（词表偏移、预测不一致）。带标点的部署权重
> 必须**从官方 ONNX 图中提取**，而不是用 pretrained.pt。

从 [GilgameshWind/X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en)
下载 `deployment/models/chunk-160ms-model/`（encoder/decoder/joiner-160ms.onnx +
tokens.txt），然后：

```bash
# 1) 从 ONNX 提取部署权重（需要 onnx 包；建议独立 venv，见脚本头注释）
python scripts/xasr_ref/extract_onnx_weights.py \
    --model-dir /path/to/X-ASR --suffix 160ms
# → 生成 /path/to/X-ASR/deployed_weights/*.npy

# 2) 打包 GGUF（默认从 deployed_weights 目录读取）
python scripts/convert_xasr_to_gguf.py \
    --weights /path/to/X-ASR/deployed_weights \
    --tokens  /path/to/X-ASR/tokens.txt \
    --output  /path/to/X-ASR/xasr-f32.gguf --out-type f32
```

`--out-type f16` 产出更小的 f16 权重；转换器会把 host 读取的表
（decoder.embedding/conv、chunkwise_conv_scale）和大词表投影强制留在 f32。

### 1.3 量化（可选，推荐）

```bash
./build/rs-quantize xasr-f32.gguf xasr-q4_k_m.gguf q4_k_m
```

| 精度 | 大小 | 说明 |
|------|------|------|
| f32 | 585 MB | 基准 |
| f16 | 305 MB | 与 f32 转写一致 |
| q8_0 | 170 MB | 与 f32 **逐字一致** |
| q4_k_m | **99.5 MB** | 离线一致，流式偶有标点差异 |

`rs-quantize` 会自动跳过 host 读取的表和小卷积核，只量化 19 层编码器的
FF/attention 投影（占参数量主体）。CPU 上量化约提速 14%。

---

## 2. 离线识别

注册后离线走通用的 `rs-asr-offline`（无需 VAD 即整段识别）：

```bash
./build/rs-asr-offline -m xasr-q4_k_m.gguf -w audio.wav --gpu true
```

输出带标点和大小写，例如
`国家发展改革委等部门发布关于开展重点行业节能降碳改造攻坚三年行动的通知，…`。

---

## 3. 流式识别（rs-asr-online）

真正的分 chunk 流式，逐 chunk 出 partial（单行刷新），跨 chunk 延续假设：

```bash
# WAV：实时速率回放 + live partial（加 --fast 则尽快跑完）
./build/rs-asr-online -m xasr-q4_k_m.gguf -w audio.wav --chunk-len 32

# 麦克风
./build/rs-asr-online -m xasr-q4_k_m.gguf --mic --chunk-len 16
```

常用参数：

| 参数 | 说明 |
|------|------|
| `-m` | GGUF 模型路径 |
| `-w <wav>` / `--mic` | 输入源：WAV 文件或麦克风（二选一） |
| `--chunk-len N` | chunk 长度（fbank 帧，10ms/帧），**必须是 16 的倍数** |
| `--fast` | WAV 模式下不按实时速率、尽快跑完 |
| `--cpu` | 强制 CPU 后端（默认用 GPU） |
| `-t N` | CPU 线程数 |
| `-V` | 打印框架/ggml 日志与 per-chunk 计时（默认静默，只显示转写） |

输出约定：转写文本走 **stdout**，提示/统计走 **stderr**，重定向/管道时
（`2>/dev/null`）得到纯文本；partial 会按终端宽度截断（`…` 前缀），最终整行定稿。

### chunk 大小 ↔ 延迟/吞吐权衡

`--chunk-len` 对应官方的 160/320/…ms 档，是**延迟与吞吐的权衡**——chunk 越大，
每次出字的延迟越高，但单位音频的算子调度开销越小、RTF 越低：

| `--chunk-len` | 首字延迟 | RTF（M1 Metal, f16, 参考值） |
|:---:|:---:|:---:|
| 16 | ~160 ms | ~0.15 |
| 32 | ~320 ms | ~0.08 |
| 96 | ~960 ms | ~0.04 |
| 192 | ~1920 ms | ~0.04 |

`chunk-len 16` 与官方 sherpa-onnx 160ms 流式模型**逐字一致**。追求吞吐/省电时用
大 chunk + 量化；追求低延迟时用小 chunk。

---

## 4. 实现要点与验证

- **前端**：kaldi fbank，POVEY 窗、`snip_edges=false`、mel 20–7600 Hz、无 LFR/CMVN、
  PCM 落在 [-1,1]（对齐 sherpa-onnx）。模型自持 `AudioProcessor`
  （`use_external_frontend`）。
- **可移植性**：编码器图每 chunk 重建（与离线路径同款 fresh-graph 模式），缓存状态走
  主机内存——这是本仓库唯一验证过在 **CPU / Metal / CUDA / Vulkan** 都正确的方式；
  持久化图会因各后端 scheduler 的 buffer 复用语义不同而在某些后端出错。
- **解码**：greedy transducer（predictor = embedding + 分组卷积 k=2 + ReLU，在 host
  上算；joiner 分段批量图）。
- **对拍**：`scripts/xasr_ref/` 提供 PyTorch 参考（`dump_offline.py --ckpt deployed`）
  与逐级 SNR 对比（`compare_snr.py`）。C++ 侧用环境变量 `RS_XASR_DUMP_DIR` 落盘各级
  张量、`RS_XASR_LOAD_FBANK` 覆盖前端输入。编码器各级 SNR 约 66 dB（f32），token 与
  官方一致。
