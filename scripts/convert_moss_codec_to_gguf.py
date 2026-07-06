#!/usr/bin/env python3
"""Convert MOSS-Audio-Tokenizer-Nano DECODER + RVQ quantizer to GGUF.

On-device we only need the decode path (tokens -> 48kHz waveform); the encoder is
skipped (reference audio is pre-encoded offline). Projections use PyTorch weight_norm
(parametrizations.weight.original0 = g magnitude, .original1 = v direction) which we
reconstruct: w = g * v / ||v||_(dims!=0). Big 2D weights -> Q8_0, small -> F32.

Usage: python convert_moss_codec_to_gguf.py --src codec/ --out moss_codec.gguf
"""
import argparse, json, collections
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize

Q8, F32 = GGMLQuantizationType.Q8_0, GGMLQuantizationType.F32

def wnorm(g, v):
    # g:[out,1,...], v:[out,in,...(kernel)]  -> reconstruct, squeeze trailing kernel=1
    norm = torch.sqrt((v.float() ** 2).sum(dim=tuple(range(1, v.ndim)), keepdim=True))
    w = g.float() * v.float() / norm
    while w.ndim > 2 and w.shape[-1] == 1:
        w = w.squeeze(-1)
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/luigi/moss-port/codec")
    ap.add_argument("--out", default="/home/luigi/moss-port/moss_codec.gguf")
    a = ap.parse_args()
    src = Path(a.src)
    cfg = json.load(open(src / "config.json"))
    st = safe_open(str(src / "model-00001-of-00001.safetensors"), "pt")
    keys = [k for k in st.keys() if not k.startswith("encoder.")]  # decoder + quantizer only

    # reconstruct weight_norm pairs, pass through everything else
    tensors = {}
    handled = set()
    for k in keys:
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            g = st.get_tensor(k); v = st.get_tensor(base + ".parametrizations.weight.original1")
            tensors[base + ".weight"] = wnorm(g, v)
            handled.add(k); handled.add(base + ".parametrizations.weight.original1")
    for k in keys:
        if k in handled:
            continue
        t = st.get_tensor(k)
        # squeeze conv kernel=1 -> 2D linear where applicable
        while t.ndim > 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        tensors[k] = t

    w = GGUFWriter(a.out, "moss_audio_tokenizer_nano")
    w.add_uint32("codec.sample_rate", cfg["sample_rate"])
    w.add_uint32("codec.downsample_rate", cfg["downsample_rate"])
    w.add_uint32("codec.n_codebooks", 16)
    w.add_uint32("codec.code_dim", cfg.get("code_dim", 8))

    nq = nf = 0
    for name in sorted(tensors):
        arr = tensors[name].detach().to(torch.float32).numpy()
        if arr.ndim == 2 and name.endswith(".weight") and arr.shape[-1] % 32 == 0 and arr.shape[0] % 32 == 0:
            w.add_tensor("codec." + name, quantize(arr, Q8), raw_dtype=Q8); nq += 1
        else:
            w.add_tensor("codec." + name, arr, raw_dtype=F32); nf += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {len(tensors)} tensors ({nq} Q8_0, {nf} F32)")

if __name__ == "__main__":
    main()
