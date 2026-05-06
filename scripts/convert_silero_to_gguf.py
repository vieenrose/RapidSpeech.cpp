#!/usr/bin/env python3
"""
Convert Silero VAD from safetensors to GGUF format.

Data layout for ggml_conv_1d compatibility:
- 3D conv weights (PyTorch [OC,IC,K]):
    Reshape to 2D [OC, K*IC] in C-order.
    GGML reads ne=[K*IC, OC]; internally reshapes to [K,IC,OC].
    im2col output [L, K*IC] @ [K*IC, OC] -> [L, OC]. Works!
- 2D LSTM weights (PyTorch [4H,H]):
    Store as [H, 4H] in C-order (transpose from [4H,H]).
    GGML reads ne=[4H, H]; ggml_mul_mat computes A^T @ B = [H,4H] @ B = F.linear result. Works!
- 1D biases: stored as-is.
"""
import argparse
import numpy as np
from safetensors.torch import load_file
from pathlib import Path
from gguf import GGUFWriter


def convert_silero_to_gguf(model_dir: str, output: str, out_type: str = "f32"):
    model_path = Path(model_dir)
    if model_path.is_file():
        pass  # already a file path
    else:
        model_path = model_path / "silero_vad_16k.safetensors"
    print(f"Loading from {model_path}")
    state_dict = load_file(str(model_path))

    writer = GGUFWriter(output, "silero-vad")

    # Metadata
    writer.add_string("general.architecture", "silero-vad")
    writer.add_string("general.name", "silero-vad")
    writer.add_int32("vad.sample_rate", 16000)
    writer.add_int32("vad.window_size", 512)
    writer.add_float32("vad.threshold", 0.5)
    writer.add_int32("vad.min_speech_ms", 250)
    writer.add_int32("vad.min_silence_ms", 100)
    writer.add_int32("vad.context_size_samples", 64)

    for name, tensor in state_dict.items():
        if name.endswith(("Loss", "loss")):
            continue

        data = tensor.detach().float().numpy()
        pt_shape = list(data.shape)

        if data.ndim == 3:
            # 3D conv weights: store as [OC, IC, K] in C-order (same as PyTorch).
            # GGML reads ne=[K, IC, OC]. In ggml_conv_1d, a->ne[1]=IC, b->ne[1]=IC.
            # GGML_ASSERT(b->ne[1] == a->ne[1]) -> IC == IC. ✓
            # Weight reshape: (a->ne[0]*a->ne[1], a->ne[2]) = (K*IC, OC). ✓
            # im2col output: [N, OL, IC*K]. matmul: [N*OL, IC*K] @ [OC, IC*K] -> [N*OL, OC]. ✓
            OC, IC, K = data.shape
            data = data.copy()  # Already [OC, IC, K] in C-order
        elif data.ndim == 2:
            # 2D LSTM weights: keep as [4H, H] in C-order (PyTorch native).
            # GGML reads ne=[H, 4H]; ggml_mul_mat computes A^T @ B = [4H, H] @ B.
            # For LSTM: mul_mat(W_ih, x) where W_ih ne=[H,4H], x ne=[H,1]
            # -> A^T @ B = [4H, H] @ [H, 1] = [4H, 1]. Correct = F.linear result.
            data = data.copy()

        writer.add_tensor(name, data.astype(np.float32))
        stored_shape = list(data.shape)
        cpp_ne = list(reversed(stored_shape))
        print(f"  {name}: PT={pt_shape} -> stored={stored_shape} -> C++_ne={cpp_ne}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nSuccessfully wrote GGUF to {output}")
    print(f"3D conv weights stored as 2D [IC*K, OC]. ggml_conv_1d reads ne=[OC, IC*K] and reshapes to [IC*K, OC] for matmul.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--out-type", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()
    convert_silero_to_gguf(args.model_dir, args.output, args.out_type)
