#!/usr/bin/env python3
"""
Convert a 3D-Speaker / FunAudioLLM CAM++ speaker-embedding checkpoint
(PyTorch state_dict, e.g. `campplus_cn_common.bin`) to a single GGUF file
compatible with RapidSpeech.cpp's `CAMPPlusModel` arch (arch/campplus.cpp).

CAM++ = FCM 2-D conv head + xvector TDNN body (CAMDenseTDNN blocks) →
        StatsPool → dense(1024→192) → L2-normalized 192-d speaker embedding.

Output layout:
  general.architecture = "campplus"
  campplus.*           — hparams (sample_rate, n_mels, channel dims, blocks …)
  Tensors (names the CAMPPlusModel graph looks up):
    FCM head (identical to the PyTorch names, `head.*`):
      head.conv1.weight, head.bn1.{weight,bias,running_mean,running_var}
      head.conv2.weight, head.bn2.*
      head.layer{1,2}.{0,1}.conv{1,2}.weight, .bn{1,2}.*
      head.layer{1,2}.0.shortcut.0.weight, .shortcut.1.*   (down-sample blocks)
    xvector body (`xvector.*` → `xv.*`, BN modules renamed):
      xv.tdnn.linear.weight,  xv.tdnn.nl.bn.*
      xv.transit{1,2,3}.linear.weight, xv.transit{1,2,3}.nl.bn.*
      xv.dense.linear.weight, xv.dense.nl.bn.*        (dense BN is affine=False)
      xv.out_nl.bn.*                                  (bare BN, no linear)
      xv.block{1,2,3}.tdnnd<i>.nonl1.bn.*
      xv.block{1,2,3}.tdnnd<i>.l1.weight
      xv.block{1,2,3}.tdnnd<i>.nonl2.bn.*
      xv.block{1,2,3}.tdnnd<i>.cam.ll.weight
      xv.block{1,2,3}.tdnnd<i>.cam.l1.{weight,bias}
      xv.block{1,2,3}.tdnnd<i>.cam.l2.{weight,bias}

BN strategy: the runtime folds each BatchNorm's (running_mean, running_var,
weight, bias) into a per-channel (gamma, beta) pair at Load() time, so this
converter writes the *raw* BN tensors — do NOT fold here. `num_batches_tracked`
buffers are dropped. Affine=False BNs (xv.dense.nl.bn) only carry
running_mean/running_var, which the runtime handles.

Memory-layout convention (same as convert_moss_td_to_gguf.py): gguf sets ggml
ne = reversed(numpy.shape). PyTorch weights are already in the C-order that
makes that reversal land ne0 on the fastest axis, so pass arrays through
directly — do NOT pre-reverse. conv2d weight [OC,IC,KH,KW] → ne=[KW,KH,IC,OC];
conv1d weight [OC,IC,K] → ne=[K,IC,OC]; both are exactly what ggml_conv_{1,2}d
expect.

Usage:
  python scripts/convert_campplus_to_gguf.py \
      --pt-file /tmp/claude-1001/campplus_cn_common.bin \
      --output  campplus.gguf [--fp32]

Requires: torch, numpy, gguf.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Please install torch", file=sys.stderr)
    raise

try:
    from gguf import GGUFWriter
except ImportError:
    print("Please install gguf: pip install gguf", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# PyTorch state_dict key  ->  GGUF tensor name (what campplus.cpp looks up).
# Returns None to drop the tensor.
# ---------------------------------------------------------------------------

def map_name(k: str) -> str | None:
    if k.endswith("num_batches_tracked"):
        return None

    # FCM head: PyTorch names are identical to the runtime's.
    if k.startswith("head."):
        return k

    if k.startswith("xvector."):
        n = "xv." + k[len("xvector."):]
        # CAM gate sub-convs.
        n = n.replace("cam_layer.linear_local.", "cam.ll.")
        n = n.replace("cam_layer.linear1.", "cam.l1.")
        n = n.replace("cam_layer.linear2.", "cam.l2.")
        # BatchNorm module renames.
        n = n.replace("out_nonlinear.batchnorm.", "out_nl.bn.")
        n = n.replace("nonlinear1.batchnorm.", "nonl1.bn.")
        n = n.replace("nonlinear2.batchnorm.", "nonl2.bn.")
        n = n.replace("nonlinear.batchnorm.", "nl.bn.")
        # Dense-layer bottleneck: the only remaining `linear1` after the CAM
        # renames above.
        n = n.replace(".linear1.", ".l1.")
        return n

    # Anything else (unexpected) is skipped.
    return None


def _store(writer: GGUFWriter, name: str, t: np.ndarray, use_f16: bool):
    t = np.ascontiguousarray(t)
    eff_ndim = sum(1 for d in t.shape if d > 1)
    # F16 for multi-dim conv/linear kernels; F32 for BN params and biases.
    if use_f16 and eff_ndim >= 2 and not name.endswith(".bias"):
        t = t.astype(np.float16)
    else:
        t = t.astype(np.float32)
    writer.add_tensor(name, t)


# Tensors the CAMPPlusModel::MapTensors sanity check (and graph) require.
def required_names(n1: int, n2: int, n3: int) -> set[str]:
    req = {
        "head.conv1.weight", "head.conv2.weight",
        "xv.tdnn.linear.weight", "xv.dense.linear.weight",
        "xv.out_nl.bn.running_mean", "xv.out_nl.bn.running_var",
    }
    for b, n in ((1, n1), (2, n2), (3, n3)):
        req.add(f"xv.block{b}.tdnnd1.l1.weight")
        for i in range(1, n + 1):
            req.add(f"xv.block{b}.tdnnd{i}.l1.weight")
            req.add(f"xv.block{b}.tdnnd{i}.cam.ll.weight")
    return req


def convert(pt_file: Path, output: Path, use_f16: bool):
    print(f"Loading {pt_file}")
    sd = torch.load(str(pt_file), weights_only=True, map_location="cpu")
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Derive block layer counts from the checkpoint (robust to variants).
    import re
    blk_layers = {1: 0, 2: 0, 3: 0}
    for k in sd.keys():
        m = re.match(r"xvector\.block(\d+)\.tdnnd(\d+)\.", k)
        if m:
            b, i = int(m.group(1)), int(m.group(2))
            if b in blk_layers:
                blk_layers[b] = max(blk_layers[b], i)
    n1, n2, n3 = blk_layers[1], blk_layers[2], blk_layers[3]
    print(f"Dense block layer counts: block1={n1} block2={n2} block3={n3}")

    writer = GGUFWriter(str(output), "campplus")  # sets general.architecture
    writer.add_string("general.name", "CAMPlus-speaker-embedding")

    # --- Hparams (campplus.cpp LoadHParams reads these; all have defaults) ---
    writer.add_uint32("campplus.sample_rate",      16000)
    writer.add_uint32("campplus.n_mels",           80)
    writer.add_uint32("campplus.fcm_in_channels",  1)
    writer.add_uint32("campplus.fcm_out_channels", 32)
    writer.add_uint32("campplus.tdnn_in_channels", 320)
    writer.add_uint32("campplus.tdnn_out_channels", 128)
    writer.add_uint32("campplus.tdnn_kernel",      5)
    writer.add_uint32("campplus.tdnn_stride",      2)
    writer.add_uint32("campplus.growth_rate",      32)
    writer.add_uint32("campplus.bn_channels",      128)
    writer.add_uint32("campplus.cam_kernel",       3)
    writer.add_uint32("campplus.block1_layers",    n1)
    writer.add_uint32("campplus.block1_dilation",  1)
    writer.add_uint32("campplus.block2_layers",    n2)
    writer.add_uint32("campplus.block2_dilation",  2)
    writer.add_uint32("campplus.block3_layers",    n3)
    writer.add_uint32("campplus.block3_dilation",  2)
    writer.add_uint32("campplus.transit1_out",     256)
    writer.add_uint32("campplus.transit2_out",     512)
    writer.add_uint32("campplus.transit3_out",     512)
    writer.add_uint32("campplus.stats_pool_dim",   1024)
    writer.add_uint32("campplus.embed_dim",        192)
    writer.add_float32("campplus.bn_eps",          1e-5)

    # --- Tensors ---
    produced = set()
    n_written = 0
    n_skipped = 0
    for k, v in sd.items():
        name = map_name(k)
        if name is None:
            n_skipped += 1
            continue
        arr = v.detach().to(torch.float32).cpu().numpy()
        _store(writer, name, arr, use_f16)
        produced.add(name)
        n_written += 1

    # --- Validate coverage against runtime requirements ---
    req = required_names(n1, n2, n3)
    missing = sorted(req - produced)
    if missing:
        print("ERROR: required tensors not produced:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        writer.close()
        sys.exit(1)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {n_written} tensors ({n_skipped} skipped) -> {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-file", type=Path,
                    default=Path("/tmp/claude-1001/campplus_cn_common.bin"),
                    help="PyTorch CAM++ state_dict checkpoint")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output GGUF path")
    ap.add_argument("--fp32", action="store_true",
                    help="Store all tensors as F32 (default: F16 conv/linear "
                         "kernels, F32 BN/bias)")
    args = ap.parse_args()

    if not args.pt_file.exists():
        print(f"Checkpoint not found: {args.pt_file}", file=sys.stderr)
        sys.exit(1)
    convert(args.pt_file, args.output, use_f16=not args.fp32)


if __name__ == "__main__":
    main()
