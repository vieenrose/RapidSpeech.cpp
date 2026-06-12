#!/usr/bin/env python3
"""Merge the jetson-tts 8 kHz distilled vocoder (Vocos8k student) into an
openvoice2 GGUF produced by convert_openvoice2_v2.py.

Replaces the stock 44.1 kHz HiFi-GAN `vocoder.*` tensors with the distilled
student (from the jetson-tts flowdec GGUF, see
github.com/vieenrose/jetson-tts ggml_offload/convert_onnx_to_gguf.py) and
rewrites the audio metadata:
    openvoice2.sample_rate   -> 8000
    openvoice2.hop_length    -> 64           (iSTFT hop @ 125 Hz frames)
    openvoice2.vocoder_type  -> "vocos8k"
    openvoice2.resample_scale-> 125 / (44100/512)

Tensor conventions for the C++ vocos8k branch:
  * conv kernels (conv_pre, dw, istft bases) stored F16 (ggml conv ops);
  * pw1/pw2/head Linear weights stored PRE-TRANSPOSED so the runtime can
    ggml_mul_mat() directly (ne = [in, out]);
  * everything else F32.

Usage:
  python scripts/convert_melo8k_merge.py \
      --base /tmp/ov2/openvoice2-base-zh.gguf \
      --student /path/to/flowdec-f32.gguf \
      --out models/openvoice2-melo8k-zh.gguf
"""
import argparse

import numpy as np
import gguf

RESAMPLE_SCALE = 125.0 / (44100.0 / 512.0)


def reader_kv(r):
    out = {}
    for key, field in r.fields.items():
        out[key] = field
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = gguf.GGUFReader(args.base)
    stud = gguf.GGUFReader(args.student)

    w = gguf.GGUFWriter(args.out, "openvoice2")

    # ---- metadata: copy base KVs with overrides --------------------------
    overrides = {
        "openvoice2.sample_rate": 8000,
        "openvoice2.hop_length": 64,
    }
    for key, field in base.fields.items():
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                   "general.architecture"):
            continue
        if key in overrides:
            w.add_int32(key, overrides[key])
            continue
        val = field.contents()
        if isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, int):
            w.add_int32(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        elif isinstance(val, list) and val and isinstance(val[0], int):
            w.add_array(key, val)
        # (skip exotic types; none expected from their converter)
    w.add_string("openvoice2.vocoder_type", "vocos8k")
    w.add_float32("openvoice2.resample_scale", RESAMPLE_SCALE)

    # ---- tensors ----------------------------------------------------------
    n_base = n_stud = 0
    for t in base.tensors:
        if t.name.startswith("vocoder."):
            continue  # drop the 44.1 kHz HiFi-GAN
        arr = t.data  # already in gguf on-disk layout
        w.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
        n_base += 1

    LINEAR = ("pw1.weight", "pw2.weight", "head.weight")
    for t in stud.tensors:
        if not t.name.startswith("dec.student."):
            continue
        name = "vocoder." + t.name[len("dec."):]      # vocoder.student.*
        # special-case the two non-student names the C++ cond block expects
        if ".cond." in t.name or ".conv_pre." in t.name:
            name = t.name.replace("dec.student.", "vocoder.")
        arr = np.asarray(t.data)
        # de-quantize nothing: student gguf is f32/f16 only
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            arr = arr.view(np.float16) if arr.dtype != np.float16 else arr
        shape = tuple(int(d) for d in t.shape)         # ne order
        arr = arr.reshape(shape[::-1])                 # back to numpy order
        if ".istft." in name:
            # CUDA conv_transpose_1d requires F32 kernels (CPU wants F16 —
            # the C++ branch casts in-graph for CPU-only runs)
            arr = arr.astype(np.float32)
        if any(name.endswith(s) for s in LINEAR):
            # stored [in,out] numpy in the student gguf -> we want runtime
            # ne=[in,out], i.e. numpy [out,in]
            arr = np.ascontiguousarray(arr.T.astype(np.float32))
        w.add_tensor(name, np.ascontiguousarray(arr))
        n_stud += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"merged: {n_base} base tensors + {n_stud} student tensors -> {args.out}")


if __name__ == "__main__":
    main()
