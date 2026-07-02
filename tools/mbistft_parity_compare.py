#!/usr/bin/env python3
"""Compare mbistft-parity C++ dumps against the PyTorch parity refs.

Usage:
  python tools/mbistft_parity_compare.py \
      --refs /home/luigi/mbvits_run/parity_refs \
      --cpp  <outdir of mbistft-parity>

Gate: cosine similarity > 0.99 for EVERY module of EVERY utterance.
The C++ dumps are raw f32 already in the refs' (PyTorch) memory order.
"""
import argparse
import json
import os
import sys

import numpy as np

MODULES = ["emb", "attn0", "attn1", "attn2", "attn3", "attn4", "attn5",
           "enc", "m_p", "logs_p", "logw", "w_ceil", "m_p_exp", "logs_p_exp",
           "z_p", "z_flow", "o_mb", "wav"]


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="/home/luigi/mbvits_run/parity_refs")
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--thresh", type=float, default=0.99)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.refs, "parity_manifest.json")))
    utts = manifest["utterances"]

    per_module_min = {m: 1.0 for m in MODULES}
    failures = []
    print(f"{'utt':>5} " + " ".join(f"{m:>10}" for m in MODULES))
    for u in utts:
        idx = u["idx"]
        row = []
        for m in MODULES:
            ref = np.load(os.path.join(args.refs, f"utt{idx:02d}_{m}.npy"))
            cpp_path = os.path.join(args.cpp, f"utt{idx:02d}_{m}.bin")
            if not os.path.exists(cpp_path):
                row.append(float("nan"))
                failures.append((idx, m, "missing"))
                continue
            cpp = np.fromfile(cpp_path, dtype=np.float32)
            if cpp.size != ref.size:
                row.append(float("nan"))
                failures.append((idx, m, f"size {cpp.size} vs {ref.size}"))
                continue
            c = cosine(ref, cpp)
            row.append(c)
            per_module_min[m] = min(per_module_min[m], c)
            if c <= args.thresh:
                failures.append((idx, m, f"cos={c:.6f}"))
        print(f"utt{idx:02d} " + " ".join(
            (f"{c:10.6f}" if c == c else f"{'MISS':>10}") for c in row))

    print("\n=== per-module MIN cosine across all utts ===")
    for m in MODULES:
        flag = "OK " if per_module_min[m] > args.thresh else "FAIL"
        print(f"  {flag} {m:12s} {per_module_min[m]:.6f}")

    if failures:
        print(f"\nFAILED: {len(failures)} module checks below {args.thresh}")
        for idx, m, why in failures[:40]:
            print(f"  utt{idx:02d} {m}: {why}")
        sys.exit(1)
    print(f"\nPASS: all modules of all {len(utts)} utts cosine > {args.thresh}")


if __name__ == "__main__":
    main()
