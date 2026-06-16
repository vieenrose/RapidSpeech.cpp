# X-ASR zh-en streaming zipformer2 transducer — ggml port (WIP)

Port of the k2-fsa / sherpa-onnx **X-ASR zh-en streaming zipformer2 transducer**
(https://huggingface.co/GilgameshWind/X-ASR-zh-en) to RapidSpeech.cpp's ggml
backend, targeting CUDA on the Jetson Nano gen1 (sm_53). Motivation: on a 4 GB
Nano the onnxruntime-CUDA path OOMs (cuDNN ~782 MB / cuBLAS-only ~2.8 GB),
whereas ggml-CUDA fits in a few hundred MB — see `edge-speech-gpu-bench` and
`docs/CUDA_10.2_JETSON_NANO.md`.

## Status

| Stage | State |
|---|---|
| ONNX→GGUF converter | ✅ `scripts/xasr/convert_xasr_to_gguf.py` (922 tensors, f16/f32) |
| Parity harness | ✅ `scripts/xasr/dump_reference.py` (onnxruntime gold dumps) |
| Arch scaffold (`XAsrZipformer2`) load + tensor map | ✅ loads all 922 tensors |
| `encoder_embed` forward | ⏳ |
| zipformer2 layers + downsample/out_combiner | ⏳ |
| streaming caches | ⏳ |
| decoder + joiner + greedy transducer search | ⏳ |
| parity vs sherpa-onnx + CUDA (RS_FORCE_CC=530) | ⏳ |

## Architecture (960 ms variant)

zipformer2, 6 stacks / 19 layers. `num_layers=2,2,4,5,4,2`,
`dims=192,256,512,768,512,256`, `cnn_kernels=31,31,15,15,15,31`,
`left_context=256,128,64,32,64,128`, `query_head_dim=32`, `value_head_dim=12`,
`num_heads=4,4,4,8,4,4`, `decode_chunk_len=96`, `T=109`. encoder_embed
(Conv2dSubsampling 1→8→32→128 + ConvNeXt + BiasNorm) → stacks → 768-dim →
`encoder_proj`→512 = `encoder_out`. Stateless decoder (embedding + Conv1d,
context_size=2) + joiner (Add→Tanh→Linear) → 5000 logits.

## Convert

```
python scripts/xasr/convert_xasr_to_gguf.py \
  --model-dir <dir-with-encoder/decoder/joiner-960ms.onnx> \
  --encoder encoder-960ms.onnx --decoder decoder-960ms.onnx \
  --joiner joiner-960ms.onnx --tokens tokens.txt \
  --output models/x-asr-zh-en-960ms-f16.gguf --f16
```

Linear weights are anonymous `onnx::MatMul_*` in the encoder ONNX; the converter
recovers them by tracing bias→Add→MatMul, and maps the bias-less `linear_pos`
by node-index proximity. GGUF tensor names follow the icefall module names.

## Build (Jetson Nano gen1 / L4T r32.7 container)

Use `scripts/build_jetson_nano_gen1_native.sh` (CUDA) — applies
`patches/ggml-cuda-10.2-sm53.patch` and uses gcc-8. The arch self-registers
under `general.architecture = "XAsrZipformer2"`.
